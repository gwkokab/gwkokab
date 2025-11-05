# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Dict, List, Optional, Tuple

import equinox as eqx
import jax
import numpyro
from jax import Array, numpy as jnp
from jaxtyping import ArrayLike
from numpyro._typing import DistributionT
from numpyro.distributions import Distribution

from ..models.utils import JointDistribution, LazyJointDistribution, ScaledMixture


__all__ = ["numpyro_poisson_likelihood"]


def numpyro_poisson_likelihood(
    dist_fn: Callable[..., DistributionT],
    priors: JointDistribution,
    variables: Dict[str, DistributionT],
    variables_index: Dict[str, int],
    log_constants: ArrayLike,
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    where_fns: Optional[List[Callable[..., Array]]],
    constants: Dict[str, Array],
) -> Callable[[List[Array], List[Array], List[Array]], Array]:
    is_lazy_prior = isinstance(priors, LazyJointDistribution)
    if is_lazy_prior:
        dependencies = priors.dependencies
        partial_order = priors.partial_order
    del priors

    def log_likelihood_fn(
        data_group: Tuple[Array, ...],
        log_ref_priors_group: Tuple[Array, ...],
        masks_group: Tuple[Array, ...],
    ):
        if is_lazy_prior:
            partial_variables_samples = [
                numpyro.sample(parameter_name, prior_dist)
                if isinstance(prior_dist, Distribution)
                else (parameter_name, prior_dist)
                for parameter_name, prior_dist in sorted(
                    variables.items(), key=lambda x: x[0]
                )
            ]
            for i in partial_order:
                kwargs = {
                    k: partial_variables_samples[v] for k, v in dependencies[i].items()
                }
                parameter_name, prior_dist_fn = partial_variables_samples[i]
                if isinstance(prior_dist_fn, jax.tree_util.Partial):
                    prior_dist = prior_dist_fn.func(
                        *prior_dist_fn.args, **prior_dist_fn.keywords, **kwargs
                    )  # type: ignore
                partial_variables_samples[i] = numpyro.sample(
                    parameter_name, prior_dist
                )
            variables_samples = partial_variables_samples  # type: ignore
        else:
            variables_samples = [
                numpyro.sample(parameter_name, prior_dist)
                for parameter_name, prior_dist in sorted(
                    variables.items(), key=lambda x: x[0]
                )
            ]
        mapped_params = {
            name: variables_samples[i] for name, i in variables_index.items()
        }

        model_instance: DistributionT = dist_fn(**mapped_params, validate_args=True)

        # μ = E_{θ|Λ}[VT(θ)]
        expected_rates = eqx.filter_jit(poisson_mean_estimator)(model_instance)

        n_buckets = len(masks_group)

        @jax.jit
        def _total_log_likelihood_fn(*args: Array):
            data_group = args[0:n_buckets]
            log_ref_priors_group = args[n_buckets : 2 * n_buckets]
            masks_group = args[2 * n_buckets : 3 * n_buckets]

            total_log_likelihood = log_constants  # - Σ log(M_i)
            # Σ log Σ exp (log p(θ|data_n) - log π_n)
            for batched_data, batched_log_ref_priors, batched_masks in zip(
                data_group, log_ref_priors_group, masks_group
            ):
                safe_data = jnp.where(
                    jnp.expand_dims(batched_masks, axis=-1),
                    batched_data,
                    model_instance.support.feasible_like(batched_data),
                )
                safe_log_ref_prior = jnp.where(
                    batched_masks, batched_log_ref_priors, 0.0
                )

                n_events_per_bucket, n_samples, _ = batched_data.shape
                batched_model_log_prob = jax.vmap(
                    jax.vmap(model_instance.log_prob, axis_size=n_samples),
                    axis_size=n_events_per_bucket,
                )(safe_data)  # type: ignore
                safe_model_log_prob = jnp.where(
                    batched_masks, batched_model_log_prob, -jnp.inf
                )
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
                safe_log_prob_sum = jnp.where(
                    jnp.isneginf(log_prob_sum), -jnp.inf, log_prob_sum
                )
                total_log_likelihood += jnp.sum(safe_log_prob_sum, axis=-1)

            if where_fns is not None and len(where_fns) > 0:
                mask = where_fns[0](**constants, **mapped_params)
                for where_fn in where_fns[1:]:
                    mask = mask & where_fn(**constants, **mapped_params)
                total_log_likelihood = jnp.where(
                    mask,
                    total_log_likelihood,
                    -jnp.inf,  # type: ignore
                )
            return total_log_likelihood

        # - μ + Σ log Σ exp (log p(θ|data_n) - log π_n) - Σ log(M_i)
        numpyro.factor(
            "log_likelihood",
            _total_log_likelihood_fn(
                *data_group,
                *log_ref_priors_group,
                *masks_group,
            )
            - expected_rates,
        )

    return log_likelihood_fn  # type: ignore
