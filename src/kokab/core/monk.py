# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from typing import List, Tuple, Union

import h5py
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array
from loguru import logger
from numpyro.distributions import Distribution
from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import analytical_likelihood
from gwkokab.poisson_mean import get_selection_fn_and_poisson_mean_estimator
from gwkokab.utils.tools import error_if
from kokab.utils.common import read_json

from .flowMC_based import FlowMCBased
from .guru import guru_arg_parser as guru_parser


def _read_mean_covariances(filename: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Reads means and covariances from a file.

    Args:
        filename (str): The path to the file containing means and covariances.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple containing two lists:
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
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_samples: int,
        minimum_mc_error: float,
        n_checkpoints: int,
        n_max_steps: int,
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
        self.minimum_mc_error = minimum_mc_error
        self.n_checkpoints = n_checkpoints
        self.n_max_steps = n_max_steps
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
        _, dist_fn, priors, variables, variables_index = self.bake_model()

        list_of_means, list_of_covariances = _read_mean_covariances(self.data_filename)

        n_events = len(list_of_means)
        mean_stack: Array = jax.block_until_ready(
            jax.device_put(jnp.stack(list_of_means, axis=0))
        )
        cov_stack = np.stack(list_of_covariances, axis=0)
        scale_tril_stack: Array = jax.device_put(jnp.linalg.cholesky(cov_stack))
        del cov_stack  # We don't need the covariance matrices anymore

        logger.debug("mean_stack.shape: {shape}", shape=mean_stack.shape)
        logger.debug("scale_tril_stack.shape: {shape}", shape=scale_tril_stack.shape)

        pmean_config = read_json(self.poisson_mean_filename)
        _, poisson_mean_estimator, T_obs, _ = (
            get_selection_fn_and_poisson_mean_estimator(
                key=self.rng_key, parameters=self.parameters, **pmean_config
            )
        )

        logpdf = analytical_likelihood(
            dist_fn,
            priors,
            variables_index,
            poisson_mean_estimator,
            self.rng_key,
            n_events=n_events,
            n_samples=self.n_samples,
            minimum_mc_error=self.minimum_mc_error,
            n_checkpoints=self.n_checkpoints,
            n_max_steps=self.n_max_steps,
        )

        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "mean_stack": mean_stack,
                "scale_tril_stack": scale_tril_stack,
                "T_obs": T_obs,
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
        "--data-filename",
        help="Path to the HDF5 file containing the data.",
        type=str,
        required=True,
    )

    likelihood_group = parser.add_argument_group("Likelihood Options")
    likelihood_group.add_argument(
        "--n-samples",
        help="Number of samples to draw from the multivariate normal distribution for each "
        "event to compute the likelihood",
        default=10_000,
        type=int,
    )
    likelihood_group.add_argument(
        "--minimum-mc-error",
        help="Minimum Monte Carlo error for the likelihood computation.",
        default=0.01,
        type=float,
    )
    likelihood_group.add_argument(
        "--n-checkpoints",
        help="Number of checkpoints to save during the optimization process.",
        default=5,
        type=int,
    )
    likelihood_group.add_argument(
        "--n-max-steps",
        help="Maximum number of steps until minimum Monte Carlo error is reached.",
        default=10,
        type=int,
    )

    return parser
