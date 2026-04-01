# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import warnings
from argparse import ArgumentParser
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple, Union

import jax
import numpy as np
from jaxtyping import Array, ArrayLike
from loguru import logger
from numpyro.distributions import Distribution

from gwkanal.core.inference_io import DiscretePELoader, PoissonMeanEstimationLoader
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from gwkokab.utils.exceptions import LoggedUserWarning, LoggedValueError

from ..utils.jenks import pad_and_stack
from .guru import Guru


class Sage(Guru):
    def __init__(
        self,
        likelihood_fn: Callable[
            [
                Callable[..., Distribution],
                JointDistribution,
                Dict[str, Distribution],
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
        data_loader: DiscretePELoader,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        variance_cut_threshold: float,
        n_buckets: Optional[int],
        threshold: float,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
        analysis_name: str = "",
    ) -> None:
        if not (all(letter.isalpha() or letter == "_" for letter in analysis_name)):
            raise LoggedValueError(
                "Analysis name must contain only letters and underscores.",
            )

        self.likelihood_fn = likelihood_fn
        self.n_buckets = n_buckets
        self.data_loader = data_loader
        self.threshold = threshold
        self.where_fns = where_fns

        logger.info(
            f"Initializing Sage class for analysis identifier: '{analysis_name}'"
        )

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
    ) -> Tuple[Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...]]:
        parameters = [p.value if isinstance(p, P) else p for p in self.parameters]

        data, log_ref_priors = self.data_loader.load(parameters, self.seed)

        if len(data) != len(log_ref_priors):
            raise LoggedValueError(
                "Number of data events does not match number of log reference priors.",
            )

        logger.info("Commencing data partitioning into buckets.")
        _data_group, _log_ref_priors_group, _masks_group = pad_and_stack(
            data, log_ref_priors, n_buckets=self.n_buckets, threshold=self.threshold
        )

        if self.n_buckets is None:
            self.n_buckets = len(_data_group)
            logger.info(
                f"Automatic bucket determination completed. Optimal buckets: {self.n_buckets}"
            )
        elif self.n_buckets != len(_data_group):
            overridden_buckets = len(_data_group)
            warnings.warn(
                f"Specified n_buckets ({self.n_buckets}) differs from partitioning results. "
                f"Overriding to {overridden_buckets} buckets for computational alignment.",
                LoggedUserWarning,
            )
            self.n_buckets = overridden_buckets

        for i in range(self.n_buckets):
            _log_ref_priors_group[i] = np.where(  # type: ignore
                _masks_group[i], _log_ref_priors_group[i], 0.0
            )

        _data_group = tuple(_data_group)
        _log_ref_priors_group = tuple(_log_ref_priors_group)
        _masks_group = tuple(_masks_group)

        # Monitor device placement
        primary_devices = jax.devices()
        logger.info(f"Staging data groups to JAX devices: {primary_devices}")

        data_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_data_group)
        )
        log_ref_priors_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_log_ref_priors_group)
        )
        masks_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_masks_group)
        )

        for i, group in enumerate(data_group):
            mask_count = np.sum(_masks_group[i] == 0)
            logger.debug(
                f"Bucket {i}: Shape {group.shape} | Padding elements: {mask_count}"
            )

        return data_group, log_ref_priors_group, masks_group

    def run(self) -> None:
        model_name = getattr(self.model, "__name__", str(self.model))
        logger.info(f"Starting inference pipeline for model: {model_name}")

        constants, priors, variables, variables_index = self.classify_model_parameters()
        logger.debug(
            f"Parameter classification: {len(variables)} variables, {len(constants)} constants."
        )

        data_group, log_ref_priors_group, masks_group = self.read_data()

        logger.info("Parsing Poisson mean configuration and initializing estimator.")
        pmean_loader = PoissonMeanEstimationLoader.from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        _, poisson_mean_estimator, pmean_kwargs = pmean_loader.get_estimators()

        logger.info(
            "Constructing likelihood function and preparing for sampler execution."
        )
        logpdf = self.likelihood_fn(
            dist_fn=self.model,
            priors=priors,
            variables=variables,
            variables_index=variables_index,
            poisson_mean_estimator=poisson_mean_estimator,
            where_fns=self.where_fns,
            constants=constants,  # type: ignore
            variance_cut_threshold=self.variance_cut_threshold,
        )
        logger.success("Likelihood function construction completed successfully.")

        N_pes = np.array(
            [np.count_nonzero(batched_masks, axis=-1) for batched_masks in masks_group],
            dtype=int,
        )
        logger.info(f"Event counts per bucket (N_pe): {N_pes}")

        logger.info(
            "Initiating sampler execution with prepared likelihood and data groups."
        )
        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "data_group": data_group,
                "log_ref_priors_group": log_ref_priors_group,
                "masks_group": masks_group,
                "pmean_kwargs": pmean_kwargs,
                "N_pes": N_pes,
            },
            labels=sorted(variables.keys()),
        )

        logger.success(
            "Inference pipeline completed successfully for model: {model_name}",
            model_name=model_name,
        )


def sage_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Sage
    script.
    """
    sage_group = parser.add_argument_group("Sage Configuration")
    sage_group.add_argument(
        "--data-loader-cfg",
        type=str,
        required=True,
        help="Path to JSON configuration for the DiscreteParameterEstimationLoader.",
    )

    optm_group = parser.add_argument_group("Performance Tuning Options")
    optm_group.add_argument(
        "--n-buckets",
        help="Manually specify the number of data buckets for memory management. "
        "See https://github.com/kokabsc/gwkokab/issues/568 for more details.",
        type=int,
        default=None,
    )
    optm_group.add_argument(
        "--threshold",
        help="Threshold (0-100) for automatic bucket optimization via Jenks natural breaks.",
        type=float,
        default=3.0,
    )

    return parser
