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
from numpyro.distributions import Distribution

from gwkanal.core.inference_io import DiscretePELoader, PoissonMeanEstimationLoader
from gwkanal.utils.literals import POSTERIOR_SAMPLES_FILENAME
from gwkokab.inference.poissonlikelihood_utils import (
    variance_of_single_event_likelihood,
)
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from gwkokab.utils.tools import batch_and_remainder, error_if, warn_if

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
        error_if(
            not (all(letter.isalpha() or letter == "_" for letter in analysis_name)),
            msg="Analysis name must contain only letters and underscores.",
        )

        self.likelihood_fn = likelihood_fn
        self.n_buckets = n_buckets
        self.data_loader = data_loader
        self.threshold = threshold
        self.where_fns = where_fns
        self.set_rng_key(seed=seed)

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
    ) -> Tuple[int, float, Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...]]:
        parameters = [p.value if isinstance(p, P) else p for p in self.parameters]

        data, log_ref_priors = self.data_loader.load(parameters, self.seed)
        sum_log_size = sum([np.log(d.shape[0]) for d in data])
        log_constants = -sum_log_size

        n_events = len(data)
        error_if(
            len(data) != len(log_ref_priors),
            AssertionError,
            msg="Number of data events does not match number of log reference priors.",
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
            warn_if(
                True,
                msg=f"Specified n_buckets ({self.n_buckets}) differs from partitioning results. "
                f"Overriding to {overridden_buckets} buckets for computational alignment.",
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

        return n_events, log_constants, data_group, log_ref_priors_group, masks_group

    def run(self) -> None:
        model_name = getattr(self.model, "__name__", str(self.model))
        logger.info(f"Starting inference pipeline for model: {model_name}")

        constants, priors, variables, variables_index = self.classify_model_parameters()
        logger.debug(
            f"Parameter classification: {len(variables)} variables, {len(constants)} constants."
        )

        n_events, log_constants, data_group, log_ref_priors_group, masks_group = (
            self.read_data()
        )

        logger.info("Parsing Poisson mean configuration and initializing estimator.")
        pmean_loader = PoissonMeanEstimationLoader.from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        _, poisson_mean_estimator, variance_of_poisson_mean_estimator, pmean_kwargs = (
            pmean_loader.get_estimators()
        )

        log_constants += n_events * np.log(pmean_kwargs["T_obs"])

        logger.info(
            "Constructing likelihood function and preparing for sampler execution."
        )
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
                "pmean_kwargs": pmean_kwargs,
            },
            labels=sorted(variables.keys()),
        )

        if self.variance_cut_threshold is not None:
            logger.info(
                f"Post-processing: Applying variance cut filtering (Threshold: {self.variance_cut_threshold})"
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

            samples_path = f"{self.output_directory}/{POSTERIOR_SAMPLES_FILENAME}"
            try:
                raw_samples = np.loadtxt(samples_path, skiprows=1, delimiter=" ")
                samples: Array = jax.block_until_ready(jax.device_put(raw_samples))
            except Exception as e:
                logger.error(
                    f"Post-processing failed: Unable to load posterior samples for filtering. Error: {e}"
                )
                return

            batched_samples, remainder_samples = batch_and_remainder(
                samples, batch_size=100
            )
            n_batches = batched_samples.shape[0]
            compute_variance_jit = jax.jit(jax.vmap(compute_variance))

            total_iters = n_batches + int(remainder_samples.shape[0] > 0)

            for i in tqdm.tqdm(
                range(total_iters),
                desc="Estimating Likelihood Variance",
            ):
                batch_sample = (
                    remainder_samples if i == n_batches else batched_samples[i]
                )
                variance = jax.device_get(compute_variance_jit(batch_sample))
                variance = np.nan_to_num(variance, nan=float("inf"))

                if mask is None:
                    mask = variance < self.variance_cut_threshold
                else:
                    mask = np.concatenate(
                        (mask, variance < self.variance_cut_threshold), axis=0
                    )

                max_variance = max(max_variance, np.max(variance))
                min_variance = min(min_variance, np.min(variance))

            logger.info(
                f"Variance estimation range: Min={min_variance:.4e}, Max={max_variance:.4e}"
            )

            assert mask is not None, "Error generating variance filter mask."
            n_kept = np.sum(mask)
            total_samples = samples.shape[0]

            logger.info(
                f"Filtering complete. Samples retained: {n_kept}/{total_samples}. "
                f"Samples rejected: {total_samples - n_kept}."
            )

            out_file = f"{self.output_directory}/variance_filtered_{POSTERIOR_SAMPLES_FILENAME}"
            np.savetxt(
                out_file,
                samples[mask],
                header=" ".join(sorted(variables.keys())),
            )
            logger.info(f"Filtered posterior samples saved to: {out_file}")


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
        "See https://github.com/gwkokab/gwkokab/issues/568 for more details.",
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
