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
from jaxtyping import Array
from loguru import logger
from numpyro.infer import MCMC, NUTS

from gwkokab.models.utils import JointDistribution
from kokab.utils.common import read_json
from kokab.utils.guru import Guru, guru_arg_parser
from kokab.utils.literals import INFERENCE_DIRECTORY


def save_inference_data(mcmc: numpyro.infer.MCMC) -> None:
    os.makedirs(INFERENCE_DIRECTORY, exist_ok=True)

    inference_data = az.from_numpyro(mcmc)

    header = list(inference_data.posterior.data_vars.keys())

    posterior_samples = mcmc.get_samples()
    np.savetxt(
        INFERENCE_DIRECTORY + "/samples.dat",
        np.column_stack([posterior_samples[key] for key in header]),
        header=" ".join(header),
    )

    summary = az.summary(inference_data)

    pd.DataFrame(summary).to_json(
        INFERENCE_DIRECTORY + "/posterior_summary.json", indent=4
    )

    posterior_data = np.permute_dims(
        np.asarray(inference_data.posterior.to_dataarray()),
        (1, 2, 0),  # (variable, chain, draw) -> (chain, draw, variable)
    )

    n_chains = posterior_data.shape[0]

    for i in range(n_chains):
        np.savetxt(
            INFERENCE_DIRECTORY + f"/chain_{i}.dat",
            posterior_data[i],
            header=" ".join(header),
            comments="#",
            delimiter=" ",
        )


class NumpyroBased(Guru):
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

        def f() -> MCMC:
            kernel = NUTS(logpdf, **sampler_config["kernel"])
            mcmc = MCMC(kernel, **sampler_config["mcmc"])
            mcmc.run(self.rng_key, **data)
            return mcmc

        if self.debug_nans:
            with jax.debug_nans(True):
                mcmc = f()
        elif self.profile_memory:
            mcmc = f()

            import datetime

            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{self.analysis_name}_memory_{time}.prof"
            jax.profiler.save_device_memory_profile(filename)
        elif self.check_leaks:
            with jax.checking_leaks():
                mcmc = f()
        else:
            mcmc = f()

        save_inference_data(mcmc)

        logger.info("Sampling and data saving complete.")


numpyro_arg_parser = guru_arg_parser
