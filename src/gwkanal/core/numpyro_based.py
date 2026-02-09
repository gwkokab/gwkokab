# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
from collections.abc import Callable
from typing import Any, Dict, List, Tuple, Union

import jax
import numpy as np
import numpyro
from jax import random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS

from gwkanal.core.guru import Guru, guru_arg_parser
from gwkanal.core.utils import PRNGKeyMixin
from gwkanal.utils.common import read_json
from gwkanal.utils.literals import INFERENCE_DIRECTORY, POSTERIOR_SAMPLES_FILENAME
from gwkokab.models.utils import JointDistribution
from gwkokab.utils.tools import error_if, warn_if


_INFERENCE_DIRECTORY = "numpyro_" + INFERENCE_DIRECTORY


def _save_inference_data(samples: Any, start_chain_idx: int, labels: List[str]) -> None:
    os.makedirs(_INFERENCE_DIRECTORY, exist_ok=True)

    samples_per_chain = np.stack([samples[key] for key in labels], axis=-1)
    combined_samples = np.concat(samples_per_chain, axis=0)

    header = " ".join(labels)

    if start_chain_idx == 0:
        np.savetxt(
            _INFERENCE_DIRECTORY + "/" + POSTERIOR_SAMPLES_FILENAME,
            combined_samples,
            header=header,
        )
    else:
        with open(_INFERENCE_DIRECTORY + "/" + POSTERIOR_SAMPLES_FILENAME, "a") as f:
            np.savetxt(f, combined_samples)

    n_chains = samples_per_chain.shape[0]

    for i in range(n_chains):
        np.savetxt(
            _INFERENCE_DIRECTORY + f"/chain_{start_chain_idx + i}.dat",
            samples_per_chain[i],
            header=header,
            comments="#",
            delimiter=" ",
        )


def _run_mcmc(
    key: PRNGKeyArray,
    kernel: numpyro.infer.NUTS,
    mcmc_cfg: Dict[str, Any],
    data: Dict[str, Any],
    labels: List[str],
):
    n_devices = jax.device_count()
    if (chain_method := mcmc_cfg.pop("chain_method")) != "parallel" and n_devices > 1:
        warn_if(
            True,
            msg=f"Multiple devices detected ({n_devices}), but chain_method is set to "
            f"'{chain_method}'. Overriding to 'parallel'.",
        )
        chain_method = "parallel"

    n_chains = mcmc_cfg.pop("num_chains", 1)
    batch_size: int = (
        n_chains if chain_method == "vectorized" else min(n_chains, n_devices)
    )
    n_batches = n_chains // batch_size

    if batch_size == 1:
        chain_method = "sequential"

    mcmc = MCMC(kernel, num_chains=batch_size, chain_method=chain_method, **mcmc_cfg)

    def _run_batch_and_save(key: PRNGKeyArray, chain_idx: int) -> PRNGKeyArray:
        """Runs a batch of MCMC chains, prints summary, and saves the data."""
        key, subkey = jrd.split(key)
        mcmc.run(subkey, **data)
        samples = mcmc.get_samples(group_by_chain=True)
        print_summary(samples)
        _save_inference_data(samples=samples, start_chain_idx=chain_idx, labels=labels)
        return key

    chain_idx: int = 0
    for _ in range(n_batches):
        key = _run_batch_and_save(key, chain_idx)
        chain_idx += batch_size

    if (remaining_chains := n_chains - n_batches * batch_size) > 0:
        mcmc.num_chains = remaining_chains
        _run_batch_and_save(key, chain_idx)


class NumpyroBased(Guru, PRNGKeyMixin):
    output_directory: str = _INFERENCE_DIRECTORY

    def driver(
        self,
        *,
        logpdf: Callable[[Array, Dict[str, Any]], Array],
        priors: JointDistribution,
        data: Dict[str, Any],
        labels: List[str],
    ) -> None:
        del priors

        sampler_cfg = read_json(self.sampler_settings_filename)

        error_if(
            (kernel_cfg := sampler_cfg.pop("kernel", None)) is None,
            msg="Kernel configuration not found in sampler settings.",
        )
        error_if(
            (mcmc_cfg := sampler_cfg.pop("mcmc", None)) is None,
            msg="MCMC configuration not found in sampler settings.",
        )

        dense_mass: Union[List[Tuple[str, ...]], bool] = kernel_cfg.pop(
            "dense_mass", False
        )

        if isinstance(dense_mass, list):
            for i in range(len(dense_mass)):
                dense_mass[i] = tuple(dense_mass[i])

        kernel = NUTS(logpdf, dense_mass=dense_mass, **kernel_cfg)

        if self.debug_nans:
            with jax.debug_nans(True):
                _run_mcmc(self.rng_key, kernel, mcmc_cfg, data, labels)
        elif self.profile_memory:
            _run_mcmc(self.rng_key, kernel, mcmc_cfg, data, labels)

            import datetime

            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{self.analysis_name}_memory_{time}.prof"
            jax.profiler.save_device_memory_profile(filename)
        elif self.check_leaks:
            with jax.checking_leaks():
                _run_mcmc(self.rng_key, kernel, mcmc_cfg, data, labels)
        else:
            _run_mcmc(self.rng_key, kernel, mcmc_cfg, data, labels)

        logger.info("Sampling and data saving complete.")


numpyro_arg_parser = guru_arg_parser
