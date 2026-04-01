# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Any, Dict, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions.distribution import Distribution


def discrete_poisson_likelihood_fn(
    model_instance: Distribution,
    poisson_mean_estimator: Callable[..., Tuple[Array, Array]],
    data_group: Tuple[Array, ...],
    log_ref_priors_group: Tuple[Array, ...],
    masks_group: Tuple[Array, ...],
    pmean_kwargs: Dict[str, Any],
    N_pes: Tuple[Array, ...],
) -> Tuple[Array, Array]:

    n_events = sum([masks_group.shape[0] for masks_group in data_group])

    total_log_likelihood = -jnp.sum(jnp.log(jnp.asarray(N_pes)))  # - Σ log(M_i)
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

    total_variance = jnp.nan_to_num(
        pe_variance + expected_rate_variance,
        nan=jnp.inf,
        posinf=jnp.inf,
        neginf=jnp.inf,
    )

    return log_likelihood, total_variance
