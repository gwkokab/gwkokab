# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


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

from gwkokab.analysis.core.analysis_base import AnalysisBase, guru_arg_parser
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
        """Runs a batch of MCMC chains, prints summary, and saves the data."""
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
    def driver(
        self,
        *,
        logpdf: Callable[[Array, Dict[str, Any]], Array],
        priors: JointDistribution,
        data: Dict[str, Any],
        labels: List[str],
    ) -> None:
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
        with h5py.File(INFERENCE_OUTPUT_FILENAME, "a") as f:
            write_to_hdf5(
                f,
                dataset_path="sampler_cfg",
                attrs={"sampler_name": "numpyro"},
            )
            write_to_hdf5(
                f,
                dataset_path="sampler_cfg/kernel",
                attrs={
                    "adapt_mass_matrix": sampler_cfg.kernel.adapt_mass_matrix,
                    "adapt_step_size": sampler_cfg.kernel.adapt_step_size,
                    "dense_mass": sampler_cfg.kernel.dense_mass,
                    "find_heuristic_step_size": sampler_cfg.kernel.find_heuristic_step_size,
                    "forward_mode_differentiation": sampler_cfg.kernel.forward_mode_differentiation,
                    "inverse_mass_matrix": sampler_cfg.kernel.inverse_mass_matrix,
                    "max_tree_depth": sampler_cfg.kernel.max_tree_depth,
                    "regularize_mass_matrix": sampler_cfg.kernel.regularize_mass_matrix,
                    "step_size": sampler_cfg.kernel.step_size,
                    "target_accept_prob": sampler_cfg.kernel.target_accept_prob,
                },
            )
            write_to_hdf5(
                f,
                dataset_path="sampler_cfg/mcmc",
                attrs={
                    "chain_method": sampler_cfg.mcmc.chain_method,
                    "jit_model_args": sampler_cfg.mcmc.jit_model_args,
                    "num_chains": sampler_cfg.mcmc.num_chains,
                    "num_samples": sampler_cfg.mcmc.num_samples,
                    "num_warmup": sampler_cfg.mcmc.num_warmup,
                    "progress_bar": sampler_cfg.mcmc.progress_bar,
                    "progress_rate": sampler_cfg.mcmc.progress_rate,
                    "thinning": sampler_cfg.mcmc.thinning,
                },
            )
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


numpyro_arg_parser = guru_arg_parser
