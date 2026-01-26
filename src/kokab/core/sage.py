# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple, Union

import jax
import numpy as np
import tqdm
from jaxtyping import Array, ArrayLike
from loguru import logger
from numpyro._typing import DistributionLike
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import (
    variance_of_single_event_likelihood,
)
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean import get_selection_fn_and_poisson_mean_estimator
from gwkokab.utils.tools import batch_and_remainder, warn_if
from kokab.core.inference_io import DiscreteParameterEstimationLoader
from kokab.utils.common import read_json
from kokab.utils.literals import POSTERIOR_SAMPLES_FILENAME

from ..utils.jenks import pad_and_stack
from .guru import Guru


class Sage(Guru):
    def __init__(
        self,
        likelihood_fn: Callable[
            [
                Callable[..., DistributionLike],
                JointDistribution,
                Dict[str, DistributionLike],
                Dict[str, int],
                ArrayLike,
                Callable[[ScaledMixture], Array],
                Optional[List[Callable[..., Array]]],
                Dict[str, Array],
            ],
            Callable,
        ],
        model: Union[Distribution, Callable[..., Distribution]],
        where_fns: Optional[List[Callable[..., Array]]],
        data_loader: DiscreteParameterEstimationLoader,
        seed: int,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        variance_cut_threshold: Optional[float],
        n_buckets: Optional[int],
        threshold: float,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
        analysis_name: str = "",
    ) -> None:
        """
        Parameters
        ----------
        model : Union[Distribution, Callable[..., Distribution]]
            model to be used in the Sage class. It can be a Distribution or a callable
            that returns a Distribution.
        where_fns : Optional[List[Callable[..., Array]]]
            List of functions to apply to the data before passing it to the model.
        data_loader : DiscreteParameterEstimationLoader
            Data loader to load the data for the analysis.
        seed : int
            Seed for the random number generator.
        prior_filename : str
            Path to the JSON file containing the prior distributions.
        poisson_mean_filename : str
            Path to the JSON file containing the Poisson mean configuration.
        sampler_settings_filename : str
            Path to the JSON file containing the sampler settings.
        n_buckets : int
            Number of buckets to use for padding and stacking the data.
        threshold : float
            Threshold for padding and stacking the data.
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
            "Analysis name must contain only letters and underscores."
        )
        self.likelihood_fn = likelihood_fn
        self.n_buckets = n_buckets
        self.data_loader = data_loader
        self.threshold = threshold
        self.where_fns = where_fns
        self.set_rng_key(seed=seed)

        super().__init__(
            analysis_name=analysis_name,
            check_leaks=check_leaks,
            debug_nans=debug_nans,
            model=model,
            poisson_mean_filename=poisson_mean_filename,
            prior_filename=prior_filename,
            profile_memory=profile_memory,
            sampler_settings_filename=sampler_settings_filename,
            variance_cut_threshold=variance_cut_threshold,
        )

    def read_data(
        self,
    ) -> Tuple[int, float, Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...]]:
        parameters = [p.value if isinstance(p, P) else p for p in self.parameters]
        data, log_ref_priors = self.data_loader.load(parameters, self.seed)
        assert len(data) == len(log_ref_priors), (
            "Data and log reference priors must have the same length"
        )
        _data_group, _log_ref_priors_group, _masks_group = pad_and_stack(
            data, log_ref_priors, n_buckets=self.n_buckets, threshold=self.threshold
        )
        if self.n_buckets is None:
            self.n_buckets = len(_data_group)
            logger.info(
                "Number of buckets not specified. Using the best number of buckets: {n_buckets}.",
                n_buckets=self.n_buckets,
            )
        elif self.n_buckets != len(_data_group):
            warn_if(
                True,
                msg=f"Specified number of buckets ({self.n_buckets}) is different from "
                f"the best number of buckets ({len(_data_group)}). Using the best number"
                " of buckets.",
            )
            self.n_buckets = len(_data_group)

        for i in range(self.n_buckets):
            _log_ref_priors_group[i] = np.where(  # type: ignore
                _masks_group[i], _log_ref_priors_group[i], 0.0
            )

        _data_group = tuple(_data_group)
        _log_ref_priors_group = tuple(_log_ref_priors_group)
        _masks_group = tuple(_masks_group)

        data_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_data_group)
        )
        log_ref_priors_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_log_ref_priors_group)
        )
        masks_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_masks_group)
        )

        logger.debug(
            "data_group.shape: {shape}",
            shape=", ".join([str(d.shape) for d in data_group]),
        )
        logger.debug(
            "log_ref_priors_group.shape: {shape}",
            shape=", ".join([str(d.shape) for d in log_ref_priors_group]),
        )
        logger.debug(
            "masks_group.shape: {shape}",
            shape=", ".join([str(d.shape) for d in masks_group]),
        )

        n_events = len(data)
        sum_log_size = sum([np.log(d.shape[0]) for d in data])
        log_constants = -sum_log_size  # -Σ log(M_i)

        return n_events, log_constants, data_group, log_ref_priors_group, masks_group

    def run(self) -> None:
        constants, priors, variables, variables_index = self.classify_model_parameters()

        n_events, log_constants, data_group, log_ref_priors_group, masks_group = (
            self.read_data()
        )

        pmean_config = read_json(self.poisson_mean_filename)
        _, poisson_mean_estimator, T_obs, variance_of_poisson_mean_estimator = (
            get_selection_fn_and_poisson_mean_estimator(
                key=self.rng_key, parameters=self.parameters, **pmean_config
            )
        )

        log_constants += n_events * np.log(T_obs)  # type: ignore

        logpdf = self.likelihood_fn(
            dist_fn=self.model,
            priors=priors,
            variables=variables,
            variables_index=variables_index,
            log_constants=log_constants,
            poisson_mean_estimator=poisson_mean_estimator,
            where_fns=self.where_fns,
            constants=constants,  # type: ignore
        )

        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "data_group": data_group,
                "log_ref_priors_group": log_ref_priors_group,
                "masks_group": masks_group,
            },
            labels=sorted(variables.keys()),
        )

        if self.variance_cut_threshold is not None:
            samples: Array = jax.block_until_ready(
                jax.device_put(
                    np.loadtxt(
                        f"{self.output_directory}/{POSTERIOR_SAMPLES_FILENAME}",
                        skiprows=1,
                        delimiter=" ",
                    )
                )
            )

            def compute_variance(sample: Array) -> Array:
                scaled_mixture = self.model(
                    **constants,
                    **{var: sample[variables_index[var]] for var in variables_index},
                )
                variance = variance_of_single_event_likelihood(
                    scaled_mixture,
                    self.n_buckets,
                    data_group,
                    log_ref_priors_group,
                    masks_group,
                ) + variance_of_poisson_mean_estimator(scaled_mixture)
                return variance

            mask = None
            max_variance = 0.0
            min_variance = float("inf")

            batched_samples, remainder_samples = batch_and_remainder(
                samples, batch_size=500
            )
            n_batches = batched_samples.shape[0]
            compute_variance_jit = jax.jit(jax.vmap(compute_variance))
            for i in tqdm.tqdm(
                range(n_batches + int(remainder_samples.shape[0] > 0)),
                desc="Computing variance of likelihood estimator",
            ):
                batch_sample = (
                    remainder_samples if i == n_batches else batched_samples[i]
                )

                variance = jax.device_get(compute_variance_jit(batch_sample))
                variance = np.nan_to_num(variance, nan=float("inf"))  # type: ignore

                if mask is None:
                    mask = variance < self.variance_cut_threshold
                else:
                    mask = np.concatenate(
                        (mask, variance < self.variance_cut_threshold), axis=0
                    )  # type: ignore

                max_variance = max(max_variance, np.max(variance))  # type: ignore
                min_variance = min(min_variance, np.min(variance))  # type: ignore
            logger.info(
                "Variance of the likelihood estimator ranges from {min_variance} to {max_variance}.",
                min_variance=min_variance,
                max_variance=max_variance,
            )
            assert mask is not None, "Mask should not be None here."
            n_saved = np.sum(mask)
            logger.info(
                "{n_saved} samples filtered out of {total_samples} with variance above the threshold of {threshold}.",
                n_saved=n_saved,
                total_samples=samples.shape[0],
                threshold=self.variance_cut_threshold,
            )
            np.savetxt(
                f"{self.output_directory}/variance_filtered_{POSTERIOR_SAMPLES_FILENAME}",
                samples[mask],
                header=" ".join(sorted(variables.keys())),
            )


def sage_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Sage script.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """

    sage_group = parser.add_argument_group("Sage Options")
    sage_group.add_argument(
        "--data-loader-cfg",
        type=str,
        required=True,
        help="Path to the JSON configuration file for the DiscreteParameterEstimationLoader.",
    )

    optm_group = parser.add_argument_group("Optimization Options")
    optm_group.add_argument(
        "--n-buckets",
        help="Number of buckets for the data arrays to be split into. "
        "This is useful for large datasets to avoid memory issues. "
        "See https://github.com/gwkokab/gwkokab/issues/568 for more details.",
        type=int,
        default=None,
    )
    optm_group.add_argument(
        "--threshold",
        help="Threshold to determine best number of buckets, if the number of buckets "
        "is not specified. It should be between 0 and 100.",
        type=float,
        default=3.0,
    )

    return parser
