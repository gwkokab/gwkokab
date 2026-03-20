# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Dict

import jax
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array
from numpyro.distributions import Distribution

from gwkokab.models.utils import JointDistribution, ScaledMixture


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
) -> Callable[[Array, Dict[str, Any]], Array]:

    names = list(variables_index.keys())
    indices = jnp.array([variables_index[name] for name in names])

    def likelihood_fn(x: Array, data: Dict[str, Any]) -> Array:
        ln_offsets = data["ln_offsets"]
        pmean_kwargs = data["pmean_kwargs"]
        T_obs = pmean_kwargs["T_obs"]
        samples_stack = data["samples_stack"]

        _, n_events, _ = samples_stack.shape

        params = {name: x[idx] for name, idx in zip(names, indices)}
        model_instance = dist_fn(**constant_params, **params)

        mask = model_instance.support.check(samples_stack)

        def compute_event_log_prob(samples):
            return model_instance.log_prob(samples)

        log_prob_model = jax.vmap(compute_event_log_prob, in_axes=1, out_axes=1)(
            samples_stack
        )

        total_ln_l = jnp.sum(
            jnn.logsumexp(log_prob_model + ln_offsets, axis=0, where=mask)
        )

        expected_rates = poisson_mean_estimator(model_instance, **pmean_kwargs)

        ln_post = (
            priors.log_prob(x)
            + total_ln_l
            + (n_events * jnp.log(T_obs))
            - expected_rates
        )

        return jnp.where(jnp.isfinite(ln_post), ln_post, -jnp.inf)

    return likelihood_fn
