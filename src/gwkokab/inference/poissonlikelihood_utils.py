# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Any, Dict, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions.distribution import Distribution


__all__ = [
    "discrete_poisson_likelihood_fn",
    "analytical_poisson_likelihood_fn",
]


def variance_tapering_fn(variance: Array, threshold: Array) -> Array:
    r"""Tapering function that penalizes high variance values more aggressively as they
    exceed the threshold.

    This is a modified implementation of the original variance tapering function
    available in Equation (12) of
    `PixelPop: High Resolution Nonparameteric Inference of Gravitational-Wave Populations in Multiple Dimensions <https://arxiv.org/abs/2406.16813>`_

    .. math::

        \mathcal{T}[\sigma^2_{\ln \hat{\mathcal{L}}}]=100(\sigma^2_{\ln \hat{\mathcal{L}}}-\sigma^2_{\mathrm{threshold}})^2

    Parameters
    ----------
    variance : Array
        Variance of the Poisson likelihood estimator.
    threshold : Array
        Threshold above which the variance will be penalized.

    Returns
    -------
    Array
        Tapering factor that can be applied to the log-likelihood to penalize high variance values.
    """
    return jnp.where(
        variance < threshold,
        jnp.zeros_like(variance),
        100.0 * jnp.square(variance - threshold),
    )


def discrete_poisson_likelihood_fn(
    model_instance: Distribution,
    poisson_mean_estimator: Callable[..., Tuple[Array, Array]],
    data_group: Tuple[Array, ...],
    log_ref_priors_group: Tuple[Array, ...],
    masks_group: Tuple[Array, ...],
    pmean_kwargs: Dict[str, Any],
    N_pes: Tuple[Array, ...],
    variance_cut_threshold: float | None,
) -> Array:

    n_events = sum([masks_group.shape[0] for masks_group in data_group])

    total_log_likelihood = -jnp.sum(
        jnp.asarray([jnp.log(N_pe).sum() for N_pe in N_pes])
    )  # - Σ log(M_i)
    pe_variance = jnp.zeros(())

    # Σ log Σ exp (log p(ω|data_n) - log π_n)
    for batched_data, batched_log_ref_priors, batched_mask, N_pe in zip(
        data_group, log_ref_priors_group, masks_group, N_pes
    ):
        feasible_point = model_instance.support.feasible_like(batched_data[0])

        safe_data = jnp.where(
            batched_mask[..., jnp.newaxis],
            batched_data,
            feasible_point,
        )

        # log p(ω|data_n)
        batch_model_log_prob: Array = model_instance.log_prob(safe_data)

        # log p(ω|data_n) - log π_n
        log_prob = batch_model_log_prob - batched_log_ref_priors
        log_prob = jnp.where(batched_mask, log_prob, -jnp.inf)

        # log Σ exp (log p(ω|data_n) - log π_n)
        log_prob_sum = jax.nn.logsumexp(log_prob, axis=-1)
        log_prob_sum_2 = jax.nn.logsumexp(2.0 * log_prob, axis=-1)

        total_log_likelihood += log_prob_sum.sum(axis=0, initial=0.0)

        pe_variance += (jnp.exp(log_prob_sum_2 - 2.0 * log_prob_sum) - 1.0 / N_pe).sum()

    # μ = E_{Ω|Λ}[VT(ω)]
    expected_rate, expected_rate_variance = poisson_mean_estimator(
        model_instance, **pmean_kwargs
    )
    # log L(ω) = -μ + Σ log Σ exp (log p(ω|data_n) - log π_n) - Σ log(M_i)
    log_likelihood = (
        total_log_likelihood - expected_rate + n_events * jnp.log(pmean_kwargs["T_obs"])
    )
    if variance_cut_threshold is not None:
        total_variance = jnp.nan_to_num(
            pe_variance + expected_rate_variance,
            nan=jnp.inf,
            posinf=jnp.inf,
            neginf=jnp.inf,
        )

        variance_tapering_factor = variance_tapering_fn(
            total_variance, variance_cut_threshold
        )
        log_likelihood -= variance_tapering_factor

    return log_likelihood


def analytical_poisson_likelihood_fn(
    model_instance: Distribution,
    poisson_mean_estimator: Callable[..., tuple[Array, Array]],
    samples_stack: Array,
    ln_offsets: Array,
    pmean_kwargs: Dict[str, Any],
    variance_cut_threshold: float | None,
) -> Array:
    mask = model_instance.support.check(samples_stack)

    def compute_event_log_prob(samples):
        return model_instance.log_prob(samples)

    log_prob_model = jax.vmap(compute_event_log_prob)(samples_stack)

    total_ln_l = jnp.sum(
        jax.nn.logsumexp(log_prob_model + ln_offsets, axis=1, where=mask)
    )

    expected_rates, expected_rate_variance = poisson_mean_estimator(
        model_instance, **pmean_kwargs
    )

    n_events, _, _ = samples_stack.shape
    T_obs = pmean_kwargs["T_obs"]

    log_likelihood = total_ln_l + n_events * jnp.log(T_obs) - expected_rates

    if variance_cut_threshold is not None:
        total_variance = jnp.nan_to_num(
            expected_rate_variance,
            nan=jnp.inf,
            posinf=jnp.inf,
            neginf=jnp.inf,
        )

        variance_tapering_factor = variance_tapering_fn(
            total_variance, variance_cut_threshold
        )
        log_likelihood -= variance_tapering_factor

    return log_likelihood
