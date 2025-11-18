# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Dict, List, Tuple

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
) -> Callable[[List[Array], List[Array], List[Array]], Array]:
    is_lazy_prior = isinstance(priors, LazyJointDistribution)
    sorted_variables = sorted(variables.items(), key=lambda x: x[0])
    if is_lazy_prior:
        dependencies = priors.dependencies
        partial_order = priors.partial_order

        def variables_samples_fn():
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
            return partial_variables_samples
    else:

        def variables_samples_fn():
            return [
                numpyro.sample(parameter_name, prior_dist)
                for parameter_name, prior_dist in sorted_variables
            ]

    del priors

    # Equivalent to `jnp.nan_to_num(-jnp.inf)`.
    MIN_FLOAT = jnp.finfo(jnp.result_type(float)).min

    @jax.jit
    def compute_batched_log_likelihood(
        model_instance: DistributionT,
        batched_data: Array,
        batched_log_ref_priors: Array,
        batched_masks: Array,
    ) -> Array:
        """Compute log likelihood for a batch of data."""
        # Make data safe for log_prob computation
        safe_data = jnp.where(
            batched_masks[..., jnp.newaxis],
            batched_data,
            model_instance.support.feasible_like(batched_data),
        )

        n_events_per_bucket = batched_data.shape[0]

        # Vectorized log_prob computation
        batched_model_log_prob = jax.vmap(
            model_instance.log_prob,
            in_axes=0,
            out_axes=0,
            axis_size=n_events_per_bucket,
        )(safe_data)

        # Apply mask and compute difference in one step
        # Use where to avoid nan_to_num when possible
        batched_log_prob = jnp.where(
            batched_masks,
            batched_model_log_prob - batched_log_ref_priors,
            MIN_FLOAT,  # Use MIN_FLOAT directly instead of -inf
        )

        # Logsumexp is more numerically stable than log(sum(exp()))
        # But we need to handle the sum over axis carefully
        log_sum_exp_probs = jax.scipy.special.logsumexp(
            batched_log_prob, axis=-1, b=None
        )

        # Sum over buckets
        return jnp.sum(log_sum_exp_probs, axis=-1)

    def log_likelihood_fn(
        data_group: Tuple[Array, ...],
        log_ref_priors_group: Tuple[Array, ...],
        masks_group: Tuple[Array, ...],
    ):
        variables_samples = variables_samples_fn()

        mapped_params = {
            name: variables_samples[i] for name, i in variables_index.items()
        }
        model_instance: DistributionT = dist_fn(**mapped_params, validate_args=True)
        total_log_likelihood = log_constants  # - Σ log(M_i)
        # Σ log Σ exp (log p(θ|data_n) - log π_n)

        for batched_data, batched_log_ref_priors, batched_masks in zip(
            data_group, log_ref_priors_group, masks_group
        ):
            batch_contribution = compute_batched_log_likelihood(
                model_instance,
                batched_data,
                batched_log_ref_priors,
                batched_masks,
            )
            total_log_likelihood += batch_contribution

        # μ = E_{θ|Λ}[VT(θ)]
        expected_rates = poisson_mean_estimator(model_instance)

        # - μ + Σ log Σ exp (log p(θ|data_n) - log π_n) - Σ log(M_i)
        log_likelihood = total_log_likelihood - expected_rates

        numpyro.factor("log_likelihood", log_likelihood)

    return log_likelihood_fn  # type: ignore
