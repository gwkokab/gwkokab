# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Any, Dict

import jax
import numpyro
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import (
    analytical_poisson_likelihood_fn,
)
from gwkokab.models.utils import JointDistribution, LazyJointDistribution, ScaledMixture


__all__ = ["numpyro_analytical_poisson_likelihood"]


def numpyro_analytical_poisson_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], tuple[Array, Array]],
    variance_cut_threshold: float,
) -> Callable[[Array, Array, Dict[str, Any]], Array]:

    if is_lazy_prior := isinstance(priors, LazyJointDistribution):
        dependencies = priors.dependencies
        partial_order = priors.partial_order
    del priors

    sorted_variables = sorted(variables_index.items(), key=lambda x: x[0])

    def log_likelihood_fn(
        samples_stack: Array,
        ln_offsets: Array,
        pmean_kwargs: Dict[str, Any],
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

        model_instance = dist_fn(**constant_params, **mapped_params)

        log_likelihood, variance = analytical_poisson_likelihood_fn(
            model_instance,
            poisson_mean_estimator,
            samples_stack,
            ln_offsets,
            pmean_kwargs,
        )

        log_likelihood = jnp.where(
            variance < variance_cut_threshold,
            log_likelihood,
            -jnp.inf,
        )

        log_likelihood = jnp.nan_to_num(
            log_likelihood,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        numpyro.factor("log_likelihood", log_likelihood)

    return log_likelihood_fn  # type: ignore[return-value]
