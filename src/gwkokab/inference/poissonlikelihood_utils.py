# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array
from numpyro._typing import DistributionT


def variance_of_single_event_likelihood(
    model_instance: DistributionT,
    n_buckets: int,
    data_group: Tuple[Array, ...],
    log_ref_priors_group: Tuple[Array, ...],
    masks_group: Tuple[Array, ...],
) -> Array:
    """See equation 9 and 10 in https://arxiv.org/abs/2406.16813."""

    @jax.jit
    def _variance_of_single_event_likelihood(*args: Array):
        data_group = args[0:n_buckets]
        log_ref_priors_group = args[n_buckets : 2 * n_buckets]
        masks_group = args[2 * n_buckets : 3 * n_buckets]

        variance = jnp.zeros(())
        # Σ log Σ exp (log p(θ|data_n) - log π_n)
        for batched_data, batched_log_ref_priors, batched_masks in zip(
            data_group, log_ref_priors_group, masks_group
        ):
            safe_data = jnp.where(
                jnp.expand_dims(batched_masks, axis=-1),
                batched_data,
                model_instance.support.feasible_like(batched_data),
            )
            safe_log_ref_prior = jnp.where(batched_masks, batched_log_ref_priors, 0.0)

            n_events_per_bucket, n_samples, _ = batched_data.shape
            batched_model_log_prob = jax.vmap(
                jax.vmap(model_instance.log_prob, axis_size=n_samples),
                axis_size=n_events_per_bucket,
            )(safe_data)  # type: ignore
            safe_model_log_prob = jnp.where(
                batched_masks, batched_model_log_prob, -jnp.inf
            )
            batched_log_prob: Array = safe_model_log_prob - safe_log_ref_prior
            safe_batched_log_prob = jnp.where(
                batched_masks & (~jnp.isnan(batched_log_prob)),
                batched_log_prob,
                -jnp.inf,
            )
            log_prob_sum = jax.nn.logsumexp(
                safe_batched_log_prob,
                axis=-1,
                where=~jnp.isneginf(safe_batched_log_prob),
            )
            log_prob_sum_2 = jax.nn.logsumexp(
                2.0 * safe_batched_log_prob,
                axis=-1,
                where=~jnp.isneginf(safe_batched_log_prob),
            )
            safe_prob_sum = jnp.where(
                jnp.isneginf(log_prob_sum), 0.0, jnp.exp(log_prob_sum)
            )
            safe_prob_sum_2 = jnp.where(
                jnp.isneginf(log_prob_sum_2), 0.0, jnp.exp(log_prob_sum_2)
            )

            N_pe = jnp.count_nonzero(batched_masks, axis=-1)

            variance += (safe_prob_sum_2 / safe_prob_sum**2 - 1.0 / N_pe).sum()

        return variance

    return _variance_of_single_event_likelihood(
        *data_group, *log_ref_priors_group, *masks_group
    )
