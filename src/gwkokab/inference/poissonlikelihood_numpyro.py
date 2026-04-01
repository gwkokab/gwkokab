# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

import jax
import numpyro
from jax import Array, numpy as jnp
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import discrete_poisson_likelihood_fn

from ..models.utils import JointDistribution, LazyJointDistribution, ScaledMixture


__all__ = ["numpyro_poisson_likelihood"]


def numpyro_poisson_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables: Dict[str, Distribution],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    where_fns: Optional[List[Callable[..., Array]]],
    constants: Dict[str, Array],
    variance_cut_threshold: float,
) -> Callable[[Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...]], Array]:
    if is_lazy_prior := isinstance(priors, LazyJointDistribution):
        dependencies = priors.dependencies
        partial_order = priors.partial_order
    del priors

    sorted_variables = sorted(variables.items(), key=lambda x: x[0])

    def log_likelihood_fn(
        data_group: Tuple[Array, ...],
        log_ref_priors_group: Tuple[Array, ...],
        masks_group: Tuple[Array, ...],
        pmean_kwargs: Dict[str, Any],
        N_pes: Tuple[Array, ...],
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
                    )  # type: ignore[arg-type]
                else:
                    prior_dist = prior_dist_fn  # type: ignore[assignment]
                partial_variables_samples[i] = numpyro.sample(
                    parameter_name, prior_dist
                )

            variables_samples = partial_variables_samples  # type: ignore[assignment]
        else:
            variables_samples = [
                numpyro.sample(parameter_name, prior_dist)
                for parameter_name, prior_dist in sorted_variables
            ]

        mapped_params = {
            name: variables_samples[i] for name, i in variables_index.items()
        }
        model_instance: Distribution = dist_fn(
            **constants, **mapped_params, validate_args=True
        )

        log_likelihood, variance = discrete_poisson_likelihood_fn(
            model_instance,
            poisson_mean_estimator,
            data_group,
            log_ref_priors_group,
            masks_group,
            pmean_kwargs,
            N_pes,
        )

        log_likelihood = jnp.where(
            variance < variance_cut_threshold,
            log_likelihood,
            -jnp.inf,
        )

        if where_fns is not None and len(where_fns) > 0:
            mapped_params = {
                name: variables_samples[i] for name, i in variables_index.items()
            }
            mask = where_fns[0](**constants, **mapped_params)
            for where_fn in where_fns[1:]:
                mask = jnp.logical_and(mask, where_fn(**constants, **mapped_params))

            log_likelihood = jnp.where(
                mask,
                log_likelihood,
                -jnp.inf,  # type: ignore[arg-type]
            )
            log_likelihood = jnp.nan_to_num(
                log_likelihood, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
            )

        numpyro.factor("log_likelihood", log_likelihood)

    return log_likelihood_fn  # type: ignore[return-value]
