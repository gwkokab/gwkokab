# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from typing import List, Tuple, Union

import h5py
import jax
from jax import numpy as jnp
from jaxtyping import Array
from loguru import logger
from numpyro.distributions import Distribution
from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import analytical_likelihood
from gwkokab.utils.tools import error_if
from kokab.utils.poisson_mean_parser import read_pmean

from .flowMC_based import FlowMCBased
from .guru import guru_arg_parser as guru_parser


def _read_mean_covariances(filename: str) -> Tuple[List[Array], List[Array]]:
    """Reads means and covariances from a file.

    Args:
        filename (str): The path to the file containing means and covariances.

    Returns:
        Tuple[List[Array], List[Array]]: A tuple containing two lists:
            - list_of_means: List of mean arrays.
            - list_of_covariances: List of covariance arrays.
    """
    assert filename.endswith(".hdf5") or filename.endswith(".h5"), (
        "The filename must end with '.hdf5' or '.h5'."
    )

    logger.info("Reading means and covariances from {filename}", filename=filename)

    list_of_means = []
    list_of_covariances = []

    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if not key.startswith("event_"):
                continue

            group = f[key]
            error_if(
                "mean" not in group,
                msg=f"Key 'mean' not found in group {key} of file {filename}.",
            )
            error_if(
                "cov" not in group,
                msg=f"Key 'cov' not found in group {key} of file {filename}.",
            )

            mean = group["mean"][:]
            cov = group["cov"][:]

            n_dim = mean.shape[0]
            error_if(
                cov.shape != (n_dim, n_dim),
                msg=f"Covariance shape {cov.shape} does not match mean shape "
                f"{mean.shape} in group {key} of file {filename}.",
            )

            list_of_means.append(mean)
            list_of_covariances.append(cov)

    return list_of_means, list_of_covariances


class Monk(FlowMCBased):
    def __init__(
        self,
        model: Union[Distribution, Callable[..., Distribution]],
        data_filename: str,
        seed: int,
        prior_filename: str,
        selection_fn_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_samples: int,
        max_iter_mean: int,
        max_iter_cov: int,
        n_vi_steps: int,
        learning_rate: float,
        batch_size: int,
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
        data_filename : str
            path to the HDF5 file containing the data.
        seed : int
            seed for the random number generator.
        prior_filename : str
            path to the JSON file containing the prior distributions.
        selection_fn_filename : str
            path to the JSON file containing the selection function.
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
        self.data_filename = data_filename
        self.n_samples = n_samples
        self.max_iter_mean = max_iter_mean
        self.max_iter_cov = max_iter_cov
        self.n_vi_steps = n_vi_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
            selection_fn_filename=selection_fn_filename,
        )

    def run(self) -> None:
        """Runs the Monk analysis."""
        _, dist_fn, priors, variables, variables_index = self.bake_model()

        list_of_means, list_of_covariances = _read_mean_covariances(self.data_filename)

        n_events = len(list_of_means)
        mean_stack: Array = jax.block_until_ready(
            jax.device_put(jnp.stack(list_of_means, axis=0), may_alias=True)
        )
        cov_stack: Array = jax.block_until_ready(
            jax.device_put(jnp.stack(list_of_covariances, axis=0), may_alias=True)
        )

        logger.debug("mean_stack.shape: {shape}", shape=mean_stack.shape)
        logger.debug("cov_stack.shape: {shape}", shape=cov_stack.shape)

        ERate_fn = read_pmean(
            self.rng_key,
            self.parameters,
            self.poisson_mean_filename,
            self.selection_fn_filename,
        )

        logpdf = analytical_likelihood(
            dist_fn,
            priors,
            variables_index,
            ERate_fn,
            self.rng_key,
            n_events=n_events,
            n_samples=self.n_samples,
            max_iter_mean=self.max_iter_mean,
            max_iter_cov=self.max_iter_cov,
            n_vi_steps=self.n_vi_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
        )

        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={"mean_stack": mean_stack, "cov_stack": cov_stack},
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
        "--data-filename",
        help="Path to the HDF5 file containing the data.",
        type=str,
        required=True,
    )

    likelihood_group = parser.add_argument_group("Likelihood Options")
    likelihood_group.add_argument(
        "--n-samples",
        help="Number of samples to draw from the multivariate normal distribution for each "
        "event to compute the likelihood, by default 10_000",
        default=10_000,
        type=int,
    )
    likelihood_group.add_argument(
        "--max-iter-mean",
        help="Maximum number of iterations for the fitting process of the mean, by default 10",
        default=10,
        type=int,
    )
    likelihood_group.add_argument(
        "--max-iter-cov",
        help="Maximum number of iterations for the fitting process of the covariance, by default 3",
        default=3,
        type=int,
    )
    likelihood_group.add_argument(
        "--n-vi-steps",
        help="Number of steps for the variational inference",
        default=5,
        type=int,
    )
    likelihood_group.add_argument(
        "--learning-rate",
        help="Learning rate for the variational inference, by default 0.01",
        default=0.01,
        type=float,
    )
    likelihood_group.add_argument(
        "--batch-size",
        help="Batch size for the `jax.lax.map` used in the likelihood computation, by default 1000",
        default=1_000,
        type=int,
    )

    return parser
