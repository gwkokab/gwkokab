# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

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
    constant_params: Dict[str, Any],
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

    sorted_variables = sorted(variables.items(), key=lambda x: x[0])

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
                for parameter_name, prior_dist in sorted_variables
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
                for parameter_name, prior_dist in sorted_variables
            ]

        @jax.jit
        def _log_likelihood_fn(
            data_group: List[Array],
            log_ref_priors_group: List[Array],
            masks_group: List[Array],
            variables_samples: List[Array],
        ):
            mapped_params = {
                name: variables_samples[i] for name, i in variables_index.items()
            }
            model_instance: DistributionT = dist_fn(
                **constant_params,
                **mapped_params,
                validate_args=True,
            )
            total_log_likelihood = log_constants  # - Σ log(M_i)
            # Σ log Σ exp (log p(θ|data_n) - log π_n)
            for batched_data, batched_log_ref_priors, batched_masks in zip(
                data_group, log_ref_priors_group, masks_group
            ):
                safe_data = jnp.where(
                    batched_masks[..., jnp.newaxis],
                    batched_data,
                    model_instance.support.feasible_like(batched_data),
                )

                n_events_per_bucket, _, _ = batched_data.shape
                batched_model_log_prob = jax.jit(
                    jax.vmap(model_instance.log_prob, axis_size=n_events_per_bucket)
                )(safe_data)
                batched_log_prob = jnp.where(
                    batched_masks,
                    batched_model_log_prob - batched_log_ref_priors,
                    -jnp.inf,
                )
                log_prob_sum = jax.nn.logsumexp(
                    batched_log_prob,
                    axis=-1,
                    where=~jnp.isneginf(batched_log_prob),
                )
                total_log_likelihood += log_prob_sum.sum(axis=-1, initial=0.0)

            # μ = E_{θ|Λ}[VT(θ)]
            expected_rates = poisson_mean_estimator(model_instance)

            # - μ + Σ log Σ exp (log p(θ|data_n) - log π_n) - Σ log(M_i)
            log_likelihood = total_log_likelihood - expected_rates

            return jnp.nan_to_num(
                log_likelihood,
                nan=-jnp.inf,
                posinf=-jnp.inf,
                neginf=-jnp.inf,
            )

        log_likelihood = _log_likelihood_fn(
            data_group,
            log_ref_priors_group,
            masks_group,
            variables_samples,
        )

        if where_fns is not None and len(where_fns) > 0:
            mapped_params = {
                name: variables_samples[i] for name, i in variables_index.items()
            }
            mask = where_fns[0](**constants, **mapped_params)
            for where_fn in where_fns[1:]:
                mask = mask & where_fn(**constants, **mapped_params)
            log_likelihood = jnp.where(
                mask,
                log_likelihood,
                -jnp.inf,  # type: ignore
            )

        numpyro.factor("log_likelihood", log_likelihood)

    return log_likelihood_fn  # type: ignore
