# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
from collections.abc import Callable
from typing import Any, Dict, List

import arviz as az
import jax
import numpy as np
import numpyro
import pandas as pd
from jax import random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger
from numpyro.infer import MCMC, NUTS

from gwkokab.models.utils import JointDistribution
from gwkokab.utils.tools import warn_if
from kokab.core.guru import Guru, guru_arg_parser
from kokab.utils.common import read_json
from kokab.utils.literals import INFERENCE_DIRECTORY, POSTERIOR_SAMPLES_FILENAME


_INFERENCE_DIRECTORY = "numpyro_" + INFERENCE_DIRECTORY


def _save_inference_data(
    inference_data: az.InferenceData, start_chain_idx: int = 0
) -> None:
    os.makedirs(_INFERENCE_DIRECTORY, exist_ok=True)

    header = list(inference_data.posterior.data_vars.keys())  # type: ignore

    posterior_samples = inference_data.posterior.stack(  # type: ignore
        sample=("chain", "draw")
    ).to_dataframe()

    if start_chain_idx == 0:
        np.savetxt(
            _INFERENCE_DIRECTORY + "/" + POSTERIOR_SAMPLES_FILENAME,
            np.column_stack([posterior_samples[key] for key in header]),
            header=" ".join(header),
        )
    else:
        with open(_INFERENCE_DIRECTORY + "/" + POSTERIOR_SAMPLES_FILENAME, "a") as f:
            np.savetxt(
                f,
                np.column_stack([posterior_samples[key] for key in header]),
            )

    summary = az.summary(inference_data)

    pd.DataFrame(summary).to_json(
        _INFERENCE_DIRECTORY + "/posterior_summary.json", indent=4
    )

    posterior_data = np.permute_dims(
        np.asarray(inference_data.posterior.to_dataarray()),  # type: ignore
        (1, 2, 0),  # (variable, chain, draw) -> (chain, draw, variable)
    )

    n_chains = posterior_data.shape[0]

    for i in range(n_chains):
        np.savetxt(
            _INFERENCE_DIRECTORY + f"/chain_{start_chain_idx + i}.dat",
            posterior_data[i],
            header=" ".join(header),
            comments="#",
            delimiter=" ",
        )


def _run_mcmc(
    key: PRNGKeyArray,
    kernel: numpyro.infer.NUTS,
    mcmc_kwargs: Dict[str, Any],
    data: Dict[str, Any],
):
    n_devices = jax.device_count()
    if (
        chain_method := mcmc_kwargs.pop("chain_method")
    ) != "parallel" and n_devices > 1:
        warn_if(
            True,
            msg=f"Multiple devices detected ({n_devices}), but chain_method is set to "
            f"'{chain_method}'. Overriding to 'parallel'.",
        )
        chain_method = "parallel"

    n_chains = mcmc_kwargs.pop("num_chains", 1)
    batch_size: int = min(n_chains, n_devices)
    n_batches = n_chains // batch_size

    mcmc = MCMC(kernel, num_chains=batch_size, chain_method=chain_method, **mcmc_kwargs)

    chain_idx: int = 0
    for _ in range(n_batches):
        key, subkey = jrd.split(key)
        mcmc.run(subkey, **data)
        inference_data = az.from_numpyro(mcmc)
        _save_inference_data(inference_data, start_chain_idx=chain_idx)
        chain_idx += batch_size

    if n_batches * batch_size < n_chains:
        mcmc.num_chains = n_chains - n_batches * batch_size
        key, subkey = jrd.split(key)
        mcmc.run(subkey, **data)
        inference_data = az.from_numpyro(mcmc)
        _save_inference_data(inference_data, start_chain_idx=chain_idx)


class NumpyroBased(Guru):
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
        del labels

        sampler_config = read_json(self.sampler_settings_filename)

        kernel = NUTS(logpdf, **sampler_config["kernel"])

        if self.debug_nans:
            with jax.debug_nans(True):
                _run_mcmc(self.rng_key, kernel, sampler_config["mcmc"], data)
        elif self.profile_memory:
            _run_mcmc(self.rng_key, kernel, sampler_config["mcmc"], data)

            import datetime

            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{self.analysis_name}_memory_{time}.prof"
            jax.profiler.save_device_memory_profile(filename)
        elif self.check_leaks:
            with jax.checking_leaks():
                _run_mcmc(self.rng_key, kernel, sampler_config["mcmc"], data)
        else:
            _run_mcmc(self.rng_key, kernel, sampler_config["mcmc"], data)

        logger.info("Sampling and data saving complete.")


numpyro_arg_parser = guru_arg_parser
