# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from typing import Union

import jax
import numpy as np
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
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_samples: int,
        n_mom_samples: int,
        max_iter_mean: int,
        max_iter_cov: int,
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
        self.n_mom_samples = n_mom_samples
        self.max_iter_mean = max_iter_mean
        self.max_iter_cov = max_iter_cov

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

        scale_stack = data["scale"]
        mean_stack = data["mean"] * scale_stack
        limits_stack = data["limits"]
        cov_stack = data["cov"] * np.apply_along_axis(
            lambda x: np.outer(x, x), 1, scale_stack
        )
        scale_tril_stack = np.linalg.cholesky(cov_stack)

        n_events = mean_stack.shape[0]

        logger.info("Parsing Poisson mean configuration and initializing estimator.")
        pmean_loader = PoissonMeanEstimationLoader.from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        _, poisson_mean_estimator, _, pmean_kwargs = pmean_loader.get_estimators()

        logpdf = analytical_likelihood(
            self.model,
            priors,
            constants,
            variables_index,
            poisson_mean_estimator,
            self.data_loader.analytical_to_model_coord_fn,
            self.rng_key,
            n_events,
            self.n_samples,
            self.n_mom_samples,
            self.max_iter_mean,
            self.max_iter_cov,
        )

        lower_bounds = jax.lax.dynamic_index_in_dim(limits_stack, 0, 1, keepdims=False)
        upper_bounds = jax.lax.dynamic_index_in_dim(limits_stack, 1, 1, keepdims=False)

        lower_cdf = np.array(
            [
                multivariate_normal.cdf(
                    scale_stack[i] * lower_bounds[i],
                    mean=mean_stack[i],
                    cov=cov_stack[i],
                )
                for i in range(n_events)
            ]
        )

        upper_cdf = np.array(
            [
                multivariate_normal.cdf(
                    scale_stack[i] * upper_bounds[i],
                    mean=mean_stack[i],
                    cov=cov_stack[i],
                )
                for i in range(n_events)
            ]
        )

        log_det_scale = np.sum(np.log(scale_stack), axis=1)

        ln_offsets = log_det_scale - np.log(np.maximum(upper_cdf - lower_cdf, 1e-10))

        logger.info("ln_offsets.shape: {shape}", shape=ln_offsets.shape)
        logger.info("lower_bounds.shape: {shape}", shape=lower_bounds.shape)
        logger.info("mean_stack.shape: {shape}", shape=mean_stack.shape)
        logger.info("scale_stack.shape: {shape}", shape=scale_stack.shape)
        logger.info("scale_tril_stack.shape: {shape}", shape=scale_tril_stack.shape)
        logger.info("upper_bounds.shape: {shape}", shape=upper_bounds.shape)

        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "ln_offsets": jax.device_put(ln_offsets),
                "lower_bounds": jax.device_put(lower_bounds),
                "mean_stack": jax.device_put(mean_stack),
                "pmean_kwargs": jax.device_put(pmean_kwargs),
                "scale_stack": jax.device_put(scale_stack),
                "scale_tril_stack": jax.device_put(scale_tril_stack),
                "upper_bounds": jax.device_put(upper_bounds),
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

    enable_validation()
    parser = guru_parser(parser)

    # Monk Options
    monk = parser.add_argument_group("Monk Options")
    monk.add_argument(
        "--data-loader-cfg",
        type=str,
        required=True,
        help="Path to JSON config for AnalyticalPELoader.",
    )

    tune = parser.add_argument_group("Tuning Options")
    tune.add_argument(
        "--n-samples",
        type=int,
        default=1_000,
        help="Number of samples of Multivariate Normal per event during likelihood estimation.",
    )
    tune.add_argument(
        "--n-mom-samples",
        type=int,
        default=1_000,
        help="Number of samples of Multivariate Normal per event during Moment of Matching estimation.",
    )
    tune.add_argument(
        "--max-iter-mean",
        type=int,
        default=10,
        help="Max iterations for Moment of Matching mean estimation.",
    )
    tune.add_argument(
        "--max-iter-cov",
        type=int,
        default=4,
        help="Max iterations for Moment of Matching covariance estimation.",
    )

    return parser
