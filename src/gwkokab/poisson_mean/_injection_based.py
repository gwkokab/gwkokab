# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import List, Optional, Tuple

import jax
import numpy as np
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from loguru import logger

from ..models.utils import ScaledMixture
from ..parameters import Parameters as P
from ._injection_based_helper import apply_injection_prior, load_injection_data


def poisson_mean_from_sensitivity_injections(
    key: PRNGKeyArray,
    parameters: List[str],
    filename: str,
    batch_size: Optional[int] = None,
    far_cut: float = 1.0,
    snr_cut: float = 10.0,
) -> Tuple[
    Optional[Callable[[Array], Array]],
    Callable[[ScaledMixture], Array],
    float | Array,
    Callable[[ScaledMixture], Array],
]:
    del key  # Unused.

    injections_dict = load_injection_data(filename, 1.0 / far_cut, snr_cut)

    _PARAM_MAPPING = {
        "mass_1": P.PRIMARY_MASS_SOURCE,
        "mass_2": P.SECONDARY_MASS_SOURCE,
        "mass1_source": P.PRIMARY_MASS_SOURCE,
        "mass2_source": P.SECONDARY_MASS_SOURCE,
        "redshift": P.REDSHIFT,
        "spin1x": P.PRIMARY_SPIN_X,
        "spin1y": P.PRIMARY_SPIN_Y,
        "spin1z": P.PRIMARY_SPIN_Z,
        "spin2x": P.SECONDARY_SPIN_X,
        "spin2y": P.SECONDARY_SPIN_Y,
        "spin2z": P.SECONDARY_SPIN_Z,
        "z": P.REDSHIFT,
    }

    injections_dict = {_PARAM_MAPPING.get(k, k): v for k, v in injections_dict.items()}
    injections_dict = apply_injection_prior(injections_dict, parameters)

    samples = jnp.stack(
        [jnp.asarray(injections_dict[param]) for param in parameters],
        axis=-1,
    )
    log_weights = np.log(injections_dict["prior"])
    analysis_time_years = injections_dict["analysis_time"]
    total_injections = injections_dict["total_generated"]

    logger.debug("Analysis time (years): {}", analysis_time_years)
    logger.debug(
        "Found {} out of {} injections with FAR < {} and SNR > {}",
        samples.shape[0],
        total_injections,
        far_cut,
        snr_cut,
    )

    def _poisson_mean(scaled_mixture: ScaledMixture) -> Array:
        model_log_prob = jax.lax.map(
            scaled_mixture.log_prob,
            samples,
            batch_size=batch_size,
        )

        log_prob = model_log_prob - log_weights

        safe_log_prob = jnp.where(
            jnp.isneginf(log_prob) | jnp.isnan(log_prob),
            -jnp.inf,
            log_prob,
        )

        logsumexp_log_prob = jnn.logsumexp(
            safe_log_prob,
            where=~jnp.isneginf(safe_log_prob),
            axis=-1,
        )

        # (T / n_total) * exp(log Σ exp(log p(θ_i|λ) - log w_i))
        return (analysis_time_years * jnp.exp(logsumexp_log_prob)) / total_injections

    def _variance_of_estimator(scaled_mixture: ScaledMixture) -> Array:
        """See equation 9 and 11 of https://arxiv.org/abs/2406.16813."""
        model_log_prob = jax.lax.map(
            scaled_mixture.log_prob,
            samples,
            batch_size=batch_size,
        )

        log_prob = model_log_prob - log_weights

        safe_log_prob = jnp.where(
            jnp.isneginf(log_prob) | jnp.isnan(log_prob),
            -jnp.inf,
            log_prob,
        )

        logsumexp_log_prob = jnn.logsumexp(
            safe_log_prob,
            where=~jnp.isneginf(safe_log_prob),
            axis=-1,
        )
        logsumexp_log_prob2 = jnn.logsumexp(
            2.0 * safe_log_prob,
            where=~jnp.isneginf(safe_log_prob),
            axis=-1,
        )

        term2 = jnp.exp(
            2.0 * jnp.log(analysis_time_years)
            - 3.0 * jnp.log(total_injections)
            + 2.0 * logsumexp_log_prob
        )
        term1 = jnp.exp(
            2.0 * jnp.log(analysis_time_years)
            - 2.0 * jnp.log(total_injections)
            + logsumexp_log_prob2
        )
        return term1 - term2

    return None, _poisson_mean, analysis_time_years, _variance_of_estimator
