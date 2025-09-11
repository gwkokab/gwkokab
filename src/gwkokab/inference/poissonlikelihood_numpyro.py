# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Dict, List, Optional

import equinox as eqx
import jax
import numpyro
from jax import Array, numpy as jnp
from jaxtyping import ArrayLike
from numpyro._typing import DistributionT
from numpyro.distributions import Distribution

from ..models.utils import JointDistribution
from ..poisson_mean import PoissonMean


__all__ = ["numpyro_poisson_likelihood"]


@eqx.filter_jit
def _batch_log_prob(
    model_instance: DistributionT,
    batched_data: Array,
    batched_log_ref_priors: Array,
    batched_masks: Array,
) -> Array:
    safe_data = jnp.where(
        jnp.expand_dims(batched_masks, axis=-1),
        batched_data,
        model_instance.support.feasible_like(batched_data),
    )
    safe_log_ref_prior = jnp.where(batched_masks, batched_log_ref_priors, 0.0)

    batched_model_log_prob = jax.vmap(jax.vmap(model_instance.log_prob))(safe_data)  # type: ignore
    safe_model_log_prob = jnp.where(batched_masks, batched_model_log_prob, -jnp.inf)
    batched_log_prob: Array = safe_model_log_prob - safe_log_ref_prior
    batched_log_prob = jnp.where(
        batched_masks & (~jnp.isnan(batched_log_prob)),
        batched_log_prob,
        -jnp.inf,
    )
    log_prob_sum = jax.nn.logsumexp(
        batched_log_prob,
        axis=-1,
        where=~jnp.isneginf(batched_log_prob),
    )
    safe_log_prob_sum = jnp.where(jnp.isneginf(log_prob_sum), -jnp.inf, log_prob_sum)
    return jnp.sum(safe_log_prob_sum, axis=-1)


def numpyro_poisson_likelihood(
    dist_fn: Callable[..., DistributionT],
    priors: JointDistribution,
    variables: Dict[str, DistributionT],
    variables_index: Dict[str, int],
    log_constants: ArrayLike,
    ERate_obj: PoissonMean,
    where_fns: Optional[List[Callable[..., Array]]],
    constants: Dict[str, Array],
) -> Callable[[List[Array], List[Array], List[Array]], Array]:
    del priors

    def log_likelihood_fn(
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

        total_log_likelihood = log_constants  # - Σ log(M_i)
        # Σ log Σ exp (log p(θ|data_n) - log π_n)
        for batched_data, batched_log_ref_priors, batched_masks in zip(
            data_group, log_ref_priors_group, masks_group
        ):
            total_log_likelihood += _batch_log_prob(
                model_instance, batched_data, batched_log_ref_priors, batched_masks
            )

        log_likelihood = total_log_likelihood - expected_rates

        if where_fns is not None and len(where_fns) > 0:
            mask = where_fns[0](**constants, **mapped_params)
            for where_fn in where_fns[1:]:
                mask = mask & where_fn(**constants, **mapped_params)
            log_likelihood = jnp.where(
                mask,
                log_likelihood,
                -jnp.inf,  # type: ignore
            )

        # - μ + Σ log Σ exp (log p(θ|data_n) - log π_n) - Σ log(M_i)
        numpyro.factor("log_likelihood", log_likelihood)

    return log_likelihood_fn  # type: ignore
