# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from collections.abc import Callable
from typing import Dict, List, Optional

import jax
import numpyro
from jax import Array, numpy as jnp
from jaxtyping import ArrayLike
from numpyro._typing import DistributionLike
from numpyro.distributions import Distribution

from ..models.utils import JointDistribution
from ..poisson_mean import PoissonMean


__all__ = ["numpyro_poisson_likelihood"]


def numpyro_poisson_likelihood(
    dist_fn: Callable[..., DistributionLike],
    priors: JointDistribution,
    variables: Dict[str, DistributionLike],
    variables_index: Dict[str, int],
    log_constants: ArrayLike,
    ERate_obj: PoissonMean,
    where_fns: Optional[List[Callable[..., Array]]],
    constants: Dict[str, Array],
) -> Callable[[List[Array], List[Array], List[Array]], Array]:
    del priors
    del where_fns
    del constants

    def likelihood_fn(
        data_group: List[Array],
        log_ref_priors_group: List[Array],
        masks_group: List[Array],
    ):
        variables_samples = [
            numpyro.sample(parameter_name, prior_dist)
            for parameter_name, prior_dist in sorted(
                variables.items(), key=lambda x: x[0]
            )
        ]
        mapped_params = {
            name: variables_samples[i] for name, i in variables_index.items()
        }

        model_instance: Distribution = dist_fn(**mapped_params, validate_args=True)

        # μ = E_{θ|Λ}[VT(θ)]
        expected_rates = ERate_obj(model_instance)

        @ft.partial(jax.vmap, in_axes=(0, 0, 0))
        def single_event_fn(data: Array, log_ref_prior: Array, mask: Array) -> Array:
            safe_data = jnp.where(
                jnp.expand_dims(mask, axis=-1),
                data,
                model_instance.support.feasible_like(data),
            )
            safe_log_ref_prior = jnp.where(mask, log_ref_prior, 0.0)

            # TODO(Qazalbash): investigate effects of `equinox.filter_vmap` and `equinox.filter_jit`
            # log p(θ|data_n)
            model_log_prob = jax.jit(jax.vmap(jax.jit(model_instance.log_prob)))(
                safe_data
            )
            safe_model_log_prob = jnp.where(mask, model_log_prob, -jnp.inf)

            # log p(θ|data_n) - log π_n
            log_prob: Array = safe_model_log_prob - safe_log_ref_prior
            log_prob = jnp.where(mask & (~jnp.isnan(log_prob)), log_prob, -jnp.inf)

            # log Σ exp (log p(θ|data_n) - log π_n)
            log_prob_sum = jax.nn.logsumexp(
                log_prob,
                axis=-1,
                where=(~jnp.isneginf(log_prob)) & mask,
            )
            return log_prob_sum

        total_log_likelihood = log_constants  # - Σ log(M_i)
        # Σ log Σ exp (log p(θ|data_n) - log π_n)
        for batched_data, batched_log_ref_priors, batched_masks in zip(
            data_group, log_ref_priors_group, masks_group
        ):
            total_log_likelihood += single_event_fn(
                batched_data, batched_log_ref_priors, batched_masks
            )

        # - μ + Σ log Σ exp (log p(θ|data_n) - log π_n) - Σ log(M_i)
        numpyro.factor("log_likelihood", total_log_likelihood - expected_rates)

    return likelihood_fn  # type: ignore
