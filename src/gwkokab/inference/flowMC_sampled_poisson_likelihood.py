# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import Any, Dict

import jax
from jax import numpy as jnp
from jaxtyping import Array

from gwkokab.inference.poissonlikelihood_utils import (
    sampled_poisson_likelihood_fn,
)
from gwkokab.models.utils import JointDistribution


__all__ = ["flowMC_sampled_poisson_likelihood"]


def flowMC_sampled_poisson_likelihood(
    sampler_fn: Callable[..., tuple[Array, Array]],
    priors: JointDistribution,
    variables: Dict[str, Any],
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    event_log_likelihood_fns: Sequence[Callable[[Array], Array]],
    pdet_fn: Callable[[Array], Array] | None,
    variance_cut_threshold: float | None,
) -> Callable[[Array, Dict[str, Any]], Array]:
    r"""Sample-mode (likelihood-mode) flowMC population log-posterior.

    The dual of :func:`flowMC_analytical_poisson_likelihood`: the population
    model is supplied as a ``sampler_fn(**params) -> (samples, log_weights)``
    and the per-event terms are evaluated by a per-event **likelihood** rather
    than a model **density**.  ``data["T_obs"]`` is read at call time, mirroring
    how the analytical wrapper reads ``data["samples_stack"]`` etc.; the event
    evaluators and ``pdet_fn`` are static and bound at construction.
    """
    del variables

    def _map_params(x: Array) -> Dict[str, Array]:
        return {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

    def log_posterior_fn(x: Array, data: Dict[str, Any]) -> Array:
        T_obs = data["T_obs"]

        mapped_params = _map_params(x)
        model_samples, model_log_weights = sampler_fn(
            **constant_params, **mapped_params
        )

        log_likelihood, _diagnostics = sampled_poisson_likelihood_fn(
            event_log_likelihood_fns,
            model_samples,
            model_log_weights,
            pdet_fn,
            T_obs,
            variance_cut_threshold,
        )

        log_posterior = priors.log_prob(x) + log_likelihood

        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        return log_posterior

    return log_posterior_fn
