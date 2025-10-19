# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import List, Optional, Tuple

import equinox as eqx
import h5py
import jax
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from numpyro._typing import DistributionT

from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean._injection_based import (
    load_o1o2o3_or_endO_injection_data,
    load_o1o2o3o4a_injection_data,
)
from gwkokab.utils.tools import batch_and_remainder


def custom_poisson_mean_estimator(
    key: PRNGKeyArray,
    parameters: Sequence[str],
    filename: str,
    batch_size: Optional[int] = None,
    far_cut: float = 1.0,
    snr_cut: float = 10.0,
    ifar_pipelines: Sequence[str] | None = None,
) -> Tuple[
    Optional[Callable[[Array], Array]], Callable[[ScaledMixture], Array], float | Array
]:
    """Custom injection-based Poisson mean estimator which ignores eccentricity."""
    del key  # Unused.

    parameters_without_ecc = list(
        filter(lambda p: p != P.ECCENTRICITY.value, parameters)
    )

    with h5py.File(filename, "r") as f:
        is_o1o2o3o4a = "events" in f

    if is_o1o2o3o4a:
        del ifar_pipelines  # Unused.
        # θ_i, log w_i, T, N_total
        samples, log_weights, analysis_time_years, total_injections = (
            load_o1o2o3o4a_injection_data(
                filename,
                parameters_without_ecc,
                far_cut,
                snr_cut,
            )
        )
    else:
        # θ_i, log w_i, T, N_total
        samples, log_weights, analysis_time_years, total_injections = (
            load_o1o2o3_or_endO_injection_data(
                filename,
                parameters_without_ecc,
                far_cut,
                snr_cut,
                ifar_pipelines,
            )
        )

    n_accepted = log_weights.shape[0]

    def _poisson_mean(scaled_mixture: ScaledMixture) -> Array:
        scaled_mixture_without_ecc: List[DistributionT] = list(
            scaled_mixture.component_distributions[0].marginal_distributions
        )
        # eccentricity is the second last distribution in both powerlawpeak and
        # bp2pfull models
        scaled_mixture_without_ecc.pop(-2)
        new_scaled_mixture = ScaledMixture(
            log_scales=scaled_mixture.log_scales,
            component_distributions=[
                JointDistribution(
                    *scaled_mixture_without_ecc,
                    validate_args=scaled_mixture.component_distributions[
                        0
                    ]._validate_args,
                )
            ],
            validate_args=scaled_mixture._validate_args,
        )
        log_prob_fn = eqx.filter_jit(eqx.filter_vmap(new_scaled_mixture.log_prob))

        def _f(carry_logsumexp: Array, data: Tuple[Array, Array]) -> Tuple[Array, None]:
            log_weights, samples = data
            # log p(θ_i|λ)
            model_log_prob = log_prob_fn(samples).reshape(log_weights.shape[0])
            # log p(θ_i|λ) - log w_i
            log_prob = model_log_prob - log_weights
            safe_log_prob = jnp.where(
                jnp.isneginf(log_prob) | jnp.isnan(log_prob),
                -jnp.inf,
                log_prob,
            )

            partial_logsumexp = jnn.logsumexp(
                safe_log_prob,
                where=~jnp.isneginf(safe_log_prob),
                axis=-1,
            )
            safe_carry_logsumexp = jnp.where(
                jnp.isneginf(carry_logsumexp) | jnp.isnan(carry_logsumexp),
                -jnp.inf,
                carry_logsumexp,
            )
            return jnp.logaddexp(safe_carry_logsumexp, partial_logsumexp), None

        initial_logprob = jnp.asarray(-jnp.inf)
        if batch_size is None or n_accepted <= batch_size:
            # If the number of accepted injections is less than or equal to the batch size,
            # we can process them all at once.
            log_prob, _ = _f(initial_logprob, (log_weights, samples))
        else:
            batched_log_weights, remainder_log_weights = batch_and_remainder(
                log_weights, batch_size
            )
            batched_samples, remainder_samples = batch_and_remainder(
                samples, batch_size
            )
            batched_logprob, _ = jax.lax.scan(
                _f,
                initial_logprob,
                (batched_log_weights, batched_samples),
            )
            log_prob, _ = _f(
                batched_logprob, (remainder_log_weights, remainder_samples)
            )

        # (T / n_total) * exp(log Σ exp(log p(θ_i|λ) - log w_i))
        return (analysis_time_years / total_injections) * jnp.exp(log_prob)

    return None, _poisson_mean, analysis_time_years
