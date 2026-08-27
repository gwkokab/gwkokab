# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The NumPyro sampler backend.

:class:`NumpyroBase` supplies ``driver`` for analyses run with NumPyro's NUTS. Chains
are run in batches sized to the available devices and each batch is written to the
output HDF5 as it finishes, so a long run leaves usable partial output rather than
nothing.

The backend is chosen at runtime from ``sampler_cfg.json``'s ``sampler_name``, not by
the console script.

See Also
--------
gwkokab.analysis.core.flowMC_base : The alternative sampler backend.
"""

import warnings
from collections.abc import Callable
from typing import Any, Dict, List

import h5py
import jax
import numpy as np
import numpyro
from jax import random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS

from gwkokab.analysis.core.analysis_base import analysis_base_arg_parser, AnalysisBase
from gwkokab.analysis.core.inference_io import NumpyroGlobalConfig, NumpyroMCMCConfig
from gwkokab.analysis.core.utils import read_from_hdf5, write_to_hdf5
from gwkokab.analysis.utils.literals import (
    CHAIN_GROUP_FORMAT,
    INFERENCE_OUTPUT_FILENAME,
    SAMPLES_GROUP_NAME,
)
from gwkokab.models.utils import JointDistribution
from gwkokab.utils.exceptions import LoggedUserWarning


def _save_inference_data(samples: Any, start_chain_idx: int) -> None:
    """Append one batch of chains to the output HDF5.

    Samples are written twice over: flattened into the cumulative ``samples`` dataset,
    and separately per chain under ``chains/chain_<n>``, so that both pooled posteriors
    and per-chain diagnostics are available afterwards.

    Parameters
    ----------
    samples : Any
        The batch's samples, grouped by chain and keyed by site name.
    start_chain_idx : int
        Index of the first chain in this batch, so chain groups are numbered
        consistently across batches.
    """
    samples_per_chain = np.stack([samples[key] for key in samples.keys()], axis=-1)
    combined_samples = np.concatenate(samples_per_chain, axis=0)

    with h5py.File(INFERENCE_OUTPUT_FILENAME, "a") as f:
        if start_chain_idx == 0:
            samples_to_save = combined_samples
        else:
            previous_samples = read_from_hdf5(f, dataset_path=SAMPLES_GROUP_NAME)
            samples_to_save = np.concatenate(
                [previous_samples, combined_samples], axis=0
            )

        write_to_hdf5(
            f,
            dataset_path=SAMPLES_GROUP_NAME,
            data=samples_to_save,
        )

        n_chains = samples_per_chain.shape[0]
        for i in range(n_chains):
            chain_number = start_chain_idx + i
            write_to_hdf5(
                f,
                dataset_path="chains/"
                + CHAIN_GROUP_FORMAT.format(chain_id=chain_number),
                data=samples_per_chain[i],
            )


def _run_mcmc(
    key: PRNGKeyArray,
    kernel: numpyro.infer.NUTS,
    mcmc_cfg: NumpyroMCMCConfig,
    data: Dict[str, Any],
):
    """Run every chain, in device-sized batches, saving as it goes.

    The chain method is reconciled with the devices actually present: with more than one
    device anything but ``"parallel"`` is overridden, and a batch size of one falls back
    to ``"sequential"``. Chains are then run ``batch_size`` at a time, with any remainder
    run in a final short batch.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key, split once per batch.
    kernel : numpyro.infer.NUTS
        The configured NUTS kernel.
    mcmc_cfg : NumpyroMCMCConfig
        Chain counts, warmup, thinning and progress settings.
    data : Dict[str, Any]
        Keyword arguments passed to the NumPyro model on each run.
    """
    n_devices = jax.device_count()
    chain_method = mcmc_cfg.chain_method
    if chain_method != "parallel" and n_devices > 1:
        warnings.warn(
            f"Multiple devices detected ({n_devices}), but chain_method is set to "
            f"'{chain_method}'. Overriding to 'parallel'.",
            LoggedUserWarning,
        )
        chain_method = "parallel"
    else:
        logger.info(f"Using chain method: '{chain_method}' with {n_devices} device(s).")

    n_chains = mcmc_cfg.num_chains
    batch_size: int = (
        n_chains if chain_method == "vectorized" else min(n_chains, n_devices)
    )

    if batch_size == 1:
        chain_method = "sequential"
        logger.info("Batch size of 1 detected. Switching to 'sequential' chain method.")

    n_batches = n_chains // batch_size

    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_cfg.num_warmup,
        num_samples=mcmc_cfg.num_samples,
        thinning=mcmc_cfg.thinning,
        num_chains=batch_size,
        chain_method=chain_method,
        progress_bar=mcmc_cfg.progress_bar,
        progress_rate=mcmc_cfg.progress_rate,
        jit_model_args=mcmc_cfg.jit_model_args,
    )

    def _run_batch_and_save(key: PRNGKeyArray, chain_idx: int) -> PRNGKeyArray:
        """Run one batch of MCMC chains, print the summary, and save the data.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key; split so the returned key is fresh for the next batch.
        chain_idx : int
            Index of the first chain in this batch.

        Returns
        -------
        PRNGKeyArray
            The advanced key.
        """
        key, subkey = jrd.split(key)
        mcmc.run(subkey, **data)
        samples = mcmc.get_samples(group_by_chain=True)
        print_summary(samples)
        _save_inference_data(samples=samples, start_chain_idx=chain_idx)
        return key

    chain_idx: int = 0
    for _ in range(n_batches):
        key = _run_batch_and_save(key, chain_idx)
        chain_idx += batch_size

    if (remaining_chains := n_chains - n_batches * batch_size) > 0:
        mcmc.num_chains = remaining_chains
        _run_batch_and_save(key, chain_idx)


class NumpyroBase(AnalysisBase):
    """Sampler mixin that runs the analysis with NumPyro's NUTS.

    Mixed with a data-representation base, which supplies ``run`` and ``read_data``, and
    with a model family's ``Core`` class. Selected at runtime by ``sampler_cfg.json``.
    """

    def driver(
        self,
        *,
        logpdf: Callable[[Array, Dict[str, Any]], Array],
        priors: JointDistribution,
        data: Dict[str, Any],
        labels: List[str],
    ) -> None:
        """Run NUTS over the given model and data.

        The sampler configuration is written to the output HDF5 before sampling starts, so
        the run can be reconstructed from the output file alone. The debugging flags
        ``debug_nans``, ``profile_memory`` and ``check_leaks`` each wrap the run
        differently; they are mutually exclusive, the first set winning.

        Parameters
        ----------
        logpdf : Callable[[Array, Dict[str, Any]], Array]
            The NumPyro model built by :mod:`gwkokab.inference`.
        priors : JointDistribution
            Unused; NumPyro draws each hyper-parameter at its own ``sample`` site, so the
            priors are already baked into the model.
        data : Dict[str, Any]
            Keyword arguments passed to the model.
        labels : List[str]
            Unused; NumPyro's sites carry their own names.
        """
        del labels
        del priors

        sampler_cfg: NumpyroGlobalConfig = self.sampler_cfg

        logger.info("Initializing NUTS sampler with provided configuration.")
        kernel = NUTS(
            logpdf,
            step_size=sampler_cfg.kernel.step_size,
            inverse_mass_matrix=sampler_cfg.kernel.inverse_mass_matrix,
            adapt_step_size=sampler_cfg.kernel.adapt_step_size,
            adapt_mass_matrix=sampler_cfg.kernel.adapt_mass_matrix,
            dense_mass=sampler_cfg.kernel.dense_mass,
            target_accept_prob=sampler_cfg.kernel.target_accept_prob,
            max_tree_depth=sampler_cfg.kernel.max_tree_depth,
            find_heuristic_step_size=sampler_cfg.kernel.find_heuristic_step_size,
            forward_mode_differentiation=sampler_cfg.kernel.forward_mode_differentiation,
            regularize_mass_matrix=sampler_cfg.kernel.regularize_mass_matrix,
        )
        logger.success("NUTS Kernel initialized.")

        logger.info("Saving sampler configuration to HDF5.")
        sampler_cfg.write_to_hdf5(INFERENCE_OUTPUT_FILENAME)
        logger.success("Sampler configuration saved.")

        mcmc_cfg = sampler_cfg.mcmc

        if self.debug_nans:
            with jax.debug_nans(True):
                _run_mcmc(self.rng_key, kernel, mcmc_cfg, data)
        elif self.profile_memory:
            _run_mcmc(self.rng_key, kernel, mcmc_cfg, data)

            import datetime

            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{self.analysis_name}_memory_{time}.prof"
            jax.profiler.save_device_memory_profile(filename)
        elif self.check_leaks:
            with jax.checking_leaks():
                _run_mcmc(self.rng_key, kernel, mcmc_cfg, data)
        else:
            _run_mcmc(self.rng_key, kernel, mcmc_cfg, data)

        logger.success("Sampling and data saving complete.")


numpyro_arg_parser = analysis_base_arg_parser
