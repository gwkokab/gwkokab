# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import Union

import h5py
import jax
import numpy as np
from loguru import logger
from numpyro.distributions import Distribution
from numpyro.distributions.distribution import enable_validation

from gwkanal.core.inference_io import AnalyticalPELoader, PoissonMeanEstimationLoader
from gwkanal.core.utils import SampleTransformer
from gwkokab.inference import analytical_likelihood

from .flowMC_based import FlowMCBased
from .guru import guru_arg_parser as guru_parser


def _save_samples_to_hdf5(
    filename: str,
    event_filenames: tuple[Path, ...],
    samples: np.ndarray,
    transformed_samples: np.ndarray,
    ln_offsets: np.ndarray,
) -> None:
    """Save the generated samples and related data to an HDF5 file.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file to save the data to.
    event_filenames : tuple[Path, ...]
        Tuple of original filenames corresponding to each event.
    samples : np.ndarray
        The original samples in analytical coordinates.
    transformed_samples : np.ndarray
        The transformed samples in model coordinates.
    ln_offsets : np.ndarray
        The log offsets for each event.
    """
    opts = {
        "compression": "gzip",
        "compression_opts": 9,
        "shuffle": True,
    }

    with h5py.File(filename, "w") as f:
        for i, fname in enumerate(event_filenames):
            event_group = f.create_group(fname.stem)
            event_group.attrs["original_filename"] = str(fname)
            event_group.create_dataset("samples", data=samples[i], **opts)
            event_group.create_dataset(
                "transformed_samples",
                data=transformed_samples[i],
                **opts,
            )
            event_group.create_dataset("ln_offsets", data=ln_offsets[i], **opts)


def _multivariate_normal_samples(
    transform: SampleTransformer,
    N: int,
    mean: np.ndarray,
    cov: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate samples from a multivariate normal distribution.

    Parameters
    ----------
    transform : SampleTransformer
        The transformation to apply to the samples after generation.
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
    np.ndarray
        Transformed samples after applying the provided transformation.
    int
        Total number of samples generated.
    """
    samples = np.zeros((N, mean.shape[0]))
    transformed_samples = transform.transform(samples)
    mask = np.zeros(N, dtype=bool)
    N_total = 0

    mean *= scale
    cov *= np.outer(scale, scale)

    while not np.all(mask):
        n_invalid = np.sum(~mask)
        N_total += n_invalid
        new_samples = (
            np.random.multivariate_normal(
                mean, cov, size=n_invalid, check_valid="raise"
            )
            / scale
        )
        new_transformed_samples = transform.transform(new_samples)
        samples[~mask] = new_samples
        transformed_samples[~mask] = new_transformed_samples
        mask = np.all((samples >= low) & (samples <= high), axis=1)
        mask &= transform.check(samples, transformed_samples)
    return samples, transformed_samples, N_total


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
        mean = data["mean"]
        cov = data["cov"]

        n_events = len(mean)

        logger.info("Parsing Poisson mean configuration and initializing estimator.")
        pmean_loader = PoissonMeanEstimationLoader.from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        _, poisson_mean_estimator, _, pmean_kwargs = pmean_loader.get_estimators()

        logpdf = analytical_likelihood(
            self.model, priors, constants, variables_index, poisson_mean_estimator
        )

        samples = []
        transformed_samples = []
        total_samples = []

        for i in range(n_events):
            event_samples, event_transformed_samples, n_total = (
                _multivariate_normal_samples(
                    self.data_loader.sample_transformer,
                    self.n_samples,
                    mean[i],
                    cov[i],
                    lower_bound[i],
                    upper_bound[i],
                    scale[i],
                )
            )

            logger.info(
                "Generated {n_samples} samples for event '{event_name}' with a total of {n_total} samples drawn to account for bounds and transformations.",
                n_samples=self.n_samples,
                event_name=self.data_loader.event_paths[i].stem,
                n_total=n_total,
            )
            samples.append(event_samples)
            transformed_samples.append(event_transformed_samples)
            total_samples.append(n_total)

        samples_stack = np.stack(samples, axis=0)
        transformed_samples_stack = np.stack(transformed_samples, axis=0)

        ln_offsets = self.data_loader.sample_transformer.log_abs_det_jacobian(
            samples_stack, transformed_samples_stack
        )

        logger.info("ln_offsets.shape: {shape}", shape=ln_offsets.shape)
        logger.info(
            "samples_stack.shape: {shape}", shape=transformed_samples_stack.shape
        )

        filename = "monk_samples.hdf5"

        _save_samples_to_hdf5(
            filename=filename,
            event_filenames=self.data_loader.event_paths,
            samples=samples_stack,
            transformed_samples=transformed_samples_stack,
            ln_offsets=ln_offsets,
        )

        logger.info(
            "Saved generated samples and related data to '{filename}'.",
            filename=filename,
        )

        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "ln_offsets": jax.device_put(ln_offsets),
                "pmean_kwargs": jax.device_put(pmean_kwargs),
                "samples_stack": jax.device_put(transformed_samples_stack),
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
