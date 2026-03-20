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

from gwkanal.core.inference_io import AnalyticalPELoader, PoissonMeanEstimationLoader
from gwkokab.inference import analytical_likelihood

from .flowMC_based import FlowMCBased
from .guru import guru_arg_parser as guru_parser


def _multivariate_normal_samples(
    N: int,
    mean: np.ndarray,
    cov: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Generate samples from a multivariate normal distribution.

    Parameters
    ----------
    N : int
        Number of samples to generate.
    mean : np.ndarray
        Mean of the multivariate normal distribution.
    cov : np.ndarray
        Covariance matrix of the multivariate normal distribution.
    low : np.ndarray
        Lower bound for the samples.
    high : np.ndarray
        Upper bound for the samples.
    scale : np.ndarray
        Scale factors to avoid numerical issues when sampling.

    Returns
    -------
    np.ndarray
        Samples drawn from the multivariate normal distribution.
    int
        Total number of samples generated.
    """
    samples = np.zeros((N, mean.shape[0]))
    mask = np.zeros(N, dtype=bool)
    N_total = 0
    while not np.all(mask):
        n_invalid = np.sum(~mask)
        N_total += n_invalid
        new_samples = (
            np.random.multivariate_normal(
                mean, cov, size=n_invalid, check_valid="raise"
            )
            / scale
        )
        samples[~mask] = new_samples
        mask = np.all((samples >= low) & (samples <= high), axis=1)
    return samples, N_total


class Monk(FlowMCBased):
    def __init__(
        self,
        model: Union[Distribution, Callable[..., Distribution]],
        data_loader: AnalyticalPELoader,
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

        scale = data["scale"]
        lower_bound = data["lower_bound"]
        upper_bound = data["upper_bound"]
        scaled_mean = list(map(lambda x, y: x * y, data["mean"], scale))
        scaled_cov = list(map(lambda x, y: x * np.outer(y, y), data["cov"], scale))

        n_events = len(scaled_mean)

        logger.info("Parsing Poisson mean configuration and initializing estimator.")
        pmean_loader = PoissonMeanEstimationLoader.from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        _, poisson_mean_estimator, _, pmean_kwargs = pmean_loader.get_estimators()

        logpdf = analytical_likelihood(
            self.model, priors, constants, variables_index, poisson_mean_estimator
        )

        ln_offsets = np.sum(np.log(scale), axis=1)

        samples = []
        for i in range(n_events):
            event_samples, n_total = _multivariate_normal_samples(
                self.n_samples,
                scaled_mean[i],
                scaled_cov[i],
                lower_bound[i],
                upper_bound[i],
                scale[i],
            )

            ln_offsets -= np.log(n_total)

            logger.info(
                "Event {i}: Generated {n_samples} samples for event {i} with {n_total} total attempts.",
                n_samples=self.n_samples,
                i=i + 1,
                n_total=n_total,
            )
            samples.append(event_samples)

        samples_stack = np.stack(samples, axis=1)

        transformed_samples = self.data_loader.analytical_to_model_coord_fn(
            samples_stack
        )

        ln_offsets += (
            self.data_loader.log_abs_det_jacobian_analytical_to_model_coord_fn(
                samples_stack, transformed_samples
            )
        )

        logger.info("ln_offsets.shape: {shape}", shape=ln_offsets.shape)
        logger.info("samples_stack.shape: {shape}", shape=transformed_samples.shape)

        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "ln_offsets": jax.device_put(ln_offsets),
                "pmean_kwargs": jax.device_put(pmean_kwargs),
                "samples_stack": jax.device_put(transformed_samples),
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

    return parser
