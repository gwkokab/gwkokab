# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from typing import Union

import jax
import numpy as np
from jaxtyping import Array
from loguru import logger
from numpyro.distributions import Distribution
from numpyro.distributions.distribution import enable_validation
from scipy.stats import multivariate_normal

from gwkanal.core.inference_io import AnalyticalPELoader, PoissonMeanEstimationLoader
from gwkokab.inference import analytical_likelihood

from .flowMC_based import FlowMCBased
from .guru import guru_arg_parser as guru_parser


class Monk(FlowMCBased):
    def __init__(
        self,
        model: Union[Distribution, Callable[..., Distribution]],
        data_loader: AnalyticalPELoader,
        seed: int,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_samples: int,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
        analysis_name: str = "",
    ) -> None:
        """

        Parameters
        ----------
        model : Union[Distribution, Callable[..., Distribution]]
            model to be used in the Monk class. It can be a Distribution or a callable
            that returns a Distribution.
        data_loader : AnalyticalPELoader
            data loader for the analytical PE data.
        seed : int
            seed for the random number generator.
        prior_filename : str
            path to the JSON file containing the prior distributions.
        poisson_mean_filename : str
            path to the JSON file containing the Poisson mean configuration.
        flowMC_settings_filename : str
            path to the JSON file containing the flowMC settings.
        debug_nans : bool, optional
            If True, checks for NaNs in each computation. See details in the
            [documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug_nans.html#jax.debug_nans),
            by default False
        profile_memory : bool, optional
            If True, enables memory profiling, by default False
        check_leaks : bool, optional
            If True, checks for JAX Tracer leaks. See details in the
            [documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.checking_leaks.html#jax.checking_leaks),
            by default False
        analysis_name : str, optional
            Name of the analysis, by default ""
        """
        assert all(letter.isalpha() or letter == "_" for letter in analysis_name), (
            "Analysis name must be alphabetic characters only."
        )
        self.data_loader = data_loader
        self.n_samples = n_samples
        self.set_rng_key(seed=seed)

        super().__init__(
            analysis_name=analysis_name or model.__name__,
            check_leaks=check_leaks,
            debug_nans=debug_nans,
            model=model,
            poisson_mean_filename=poisson_mean_filename,
            prior_filename=prior_filename,
            profile_memory=profile_memory,
            sampler_settings_filename=sampler_settings_filename,
        )

    def run(self) -> None:
        """Runs the Monk analysis."""
        constants, priors, variables, variables_index = self.classify_model_parameters()

        data = self.data_loader.load(self.parameters)

        mean_stack: Array = jax.device_put(data["mean"])
        limits_stack = jax.device_put(data["limits"])
        cov_stack = data["cov"]
        scale_tril_stack: Array = jax.device_put(np.linalg.cholesky(cov_stack))
        # del cov_stack  # We don't need the covariance matrices anymore

        n_events = mean_stack.shape[0]

        logger.debug("mean_stack.shape: {shape}", shape=mean_stack.shape)
        logger.debug("scale_tril_stack.shape: {shape}", shape=scale_tril_stack.shape)
        logger.debug("limits_stack.shape: {shape}", shape=limits_stack.shape)

        logger.info("Parsing Poisson mean configuration and initializing estimator.")
        pmean_loader = PoissonMeanEstimationLoader.from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        _, poisson_mean_estimator, T_obs, variance_of_poisson_mean_estimator = (
            pmean_loader.get_estimators()
        )

        logpdf = analytical_likelihood(
            self.model,
            priors,
            constants,
            variables_index,
            poisson_mean_estimator,
            self.rng_key,
            n_events,
            self.n_samples,
        )

        lower_bounds = jax.lax.dynamic_index_in_dim(limits_stack, 0, 1, keepdims=False)
        upper_bounds = jax.lax.dynamic_index_in_dim(limits_stack, 1, 1, keepdims=False)

        lower_cdf = np.array(
            [
                multivariate_normal.cdf(
                    lower_bounds[i], mean=mean_stack[i], cov=cov_stack[i]
                )
                for i in range(n_events)
            ]
        )

        upper_cdf = np.array(
            [
                multivariate_normal.cdf(
                    upper_bounds[i], mean=mean_stack[i], cov=cov_stack[i]
                )
                for i in range(n_events)
            ]
        )

        ln_offsets = -np.log(np.maximum(upper_cdf - lower_cdf, 1e-10))

        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "mean_stack": mean_stack,
                "scale_tril_stack": scale_tril_stack,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "T_obs": T_obs,
                "ln_offsets": ln_offsets,
            },
            labels=sorted(variables.keys()),
        )


def monk_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Monk script.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """

    # Global enable validation for all distributions
    enable_validation()

    parser = guru_parser(parser)

    monk_group = parser.add_argument_group("Monk Options")
    monk_group.add_argument(
        "--data-loader-cfg",
        type=str,
        required=True,
        help="Path to JSON configuration for the AnalyticalPELoader.",
    )

    likelihood_group = parser.add_argument_group("Likelihood Options")
    likelihood_group.add_argument(
        "--n-samples",
        help="Number of samples to draw from the multivariate normal distribution for each "
        "event to compute the likelihood",
        default=10_000,
        type=int,
    )

    return parser
