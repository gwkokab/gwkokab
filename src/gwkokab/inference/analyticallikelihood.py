# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Dict

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
    r"""Compute the analytical likelihood function for a model given its parameters.

    .. math::

        \mathcal{L}_{\mathrm{analytical}}(\Lambda,\kappa)\propto
        \exp\left(-\mu(\Lambda,\kappa)\right)
        \prod_{i=1}^{N}
        \iint\mathcal{G}(\theta,z|\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)
        \rho(\theta,z\mid\Lambda,\kappa)d\theta dz

        \mathcal{L}_{\mathrm{analytical}}(\Lambda,\kappa)\propto
        \exp{\left(-\mu(\Lambda,\kappa)\right)}\prod_{i=1}^{N}\bigg<
        \rho(\theta_{i,j},z_{i,j}\mid\Lambda,\kappa)
        \bigg>_{\theta_{i,j},z_{i,j}\sim\mathcal{G}(\theta,z|\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)}
    """

    def likelihood_fn(x: Array, data: Dict[str, Any]) -> Array:
        ln_offsets = data["ln_offsets"]
        pmean_kwargs = data["pmean_kwargs"]
        T_obs = pmean_kwargs["T_obs"]
        samples_stack = data["samples_stack"]

        _, n_events, _ = samples_stack.shape

        params = {name: x[i] for name, i in variables_index.items()}
        model_instance = dist_fn(**constant_params, **params)

        mask = model_instance.support.check(samples_stack)

        safe_samples = jnp.where(
            mask[..., jnp.newaxis],
            samples_stack,
            model_instance.support.feasible_like(samples_stack),
        )
        log_prob_model = model_instance.log_prob(safe_samples)

        log_weights = jnp.where(mask, ln_offsets + log_prob_model, -jnp.inf)

        log_est = jnn.logsumexp(log_weights, axis=0, where=mask)

        total_ln_l = jnp.sum(log_est)

        expected_rates = poisson_mean_estimator(model_instance, **pmean_kwargs)

        ln_post = (
            priors.log_prob(x) + total_ln_l + n_events * jnp.log(T_obs) - expected_rates
        )

        return jnp.nan_to_num(ln_post, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf)

    return likelihood_fn
