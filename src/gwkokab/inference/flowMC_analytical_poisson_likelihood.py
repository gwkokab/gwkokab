# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Dict

import jax
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import (
    analytical_poisson_likelihood_fn,
)
from gwkokab.models.utils import JointDistribution, ScaledMixture


__all__ = ["flowMC_analytical_poisson_likelihood"]


def flowMC_analytical_poisson_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables: Dict[str, Distribution],
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], tuple[Array, Array]],
    variance_cut_threshold: float,
) -> Callable[[Array, Dict[str, Any]], Array]:
    del variables

    def _map_params(x: Array) -> Dict[str, Array]:
        return {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

    def log_posterior_fn(x: Array, data: Dict[str, Any]) -> Array:
        ln_offsets = data["ln_offsets"]
        pmean_kwargs = data["pmean_kwargs"]
        samples_stack = data["samples_stack"]

        mapped_params = _map_params(x)
        model_instance = dist_fn(**constant_params, **mapped_params)

        log_likelihood, variance = analytical_poisson_likelihood_fn(
            model_instance,
            poisson_mean_estimator,
            samples_stack,
            ln_offsets,
            pmean_kwargs,
        )

        log_posterior = priors.log_prob(x) + log_likelihood

        log_likelihood = jnp.where(
            variance < variance_cut_threshold,
            log_likelihood,
            -jnp.inf,
        )

        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        return log_posterior

    return log_posterior_fn
