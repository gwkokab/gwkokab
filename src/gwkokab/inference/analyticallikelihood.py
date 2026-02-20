# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from typing import Any, Callable, Dict

import jax
import numpy as np
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import Distribution
from numpyro.distributions.continuous import _batch_mahalanobis, tri_logabsdet

from gwkokab.models.utils import JointDistribution, ScaledMixture


LOG_2PI = np.log(2.0 * np.pi)


def mvn_samples(
    loc: Array, scale_tril: Array, n_samples: int, key: PRNGKeyArray
) -> Array:
    """Generate samples from a multivariate normal distribution using method from
    `numpyro.distributions.MultivariateNormal`.

    Parameters
    ----------
    loc : Array
        Mean vector of the multivariate normal distribution.
    scale_tril : Array
        Lower-triangular Cholesky factor of the covariance matrix of the multivariate normal distribution.
    n_samples : int
        Number of samples to generate.
    key : PRNGKeyArray
        JAX random key for sampling.

    Returns
    -------
    Array
        Samples drawn from the multivariate normal distribution.
    """
    num_events, dim = loc.shape
    eps = jrd.normal(key, shape=(n_samples, num_events, dim))
    return loc + jnp.einsum("eij,sej->sei", scale_tril, eps)


@ft.partial(jax.vmap, in_axes=(0, 0, 1), out_axes=1)
def mvn_log_prob(loc: Array, scale_tril: Array, value: Array) -> Array:
    """Compute log probability of a multivariate normal distribution using method from
    `numpyro.distributions.MultivariateNormal`.

    Parameters
    ----------
    loc : Array
        Mean vector of the multivariate normal distribution.
    scale_tril : Array
        Lower-triangular Cholesky factor of the covariance matrix of the multivariate normal distribution.
    value : Array
        Value at which to evaluate the log probability.

    Returns
    -------
    Array
        Log probability of the multivariate normal distribution at the given value.
    """
    M = _batch_mahalanobis(scale_tril, value - loc)
    half_log_det = tri_logabsdet(scale_tril)
    normalize_term = half_log_det + 0.5 * scale_tril.shape[-1] * LOG_2PI
    return -0.5 * M - normalize_term


def cholesky_decomposition(covariance_matrix: Array) -> Array:
    """Compute the Cholesky decomposition of a covariance matrix.

    Parameters
    ----------
    covariance_matrix : Array
        Covariance matrix to decompose.

    Returns
    -------
    Array
        Lower-triangular Cholesky factor of the covariance matrix.
    """
    return jnp.linalg.cholesky(covariance_matrix)


def _importance_sample_log_prob(
    model_instance: Distribution,
    samples: Array,
    ms: Array,
    sts: Array,
    ss: Array,
    log_q: Array,
    coord_fn: Callable[[Array], Array],
) -> Array:
    """Helper to compute model log-probs and importance weights for a given sample
    set.
    """
    transformed = coord_fn(samples)
    mask = model_instance.support.check(transformed)

    # Safe evaluation of model log-prob
    safe_transformed = jnp.where(
        mask[..., jnp.newaxis],
        transformed,
        model_instance.support.feasible_like(transformed),
    )
    log_p_model = jnp.where(mask, model_instance.log_prob(safe_transformed), -jnp.inf)

    # Event-specific log-prob (MVN)
    log_p_event = mvn_log_prob(ms, sts, ss * samples)

    # Importance weights: log(p_model * p_event / q)
    return log_p_model + log_p_event - log_q


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    analytical_to_model_coord_fn: Callable[[Array], Array],
    n_samples: int = 500,
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
        mean_stack = data["mean_stack"]
        scale_tril_stack = data["scale_tril_stack"]
        lb = data["lower_bounds"]
        ub = data["upper_bounds"]
        scale_stack = data["scale_stack"]
        ln_offsets = data["ln_offsets"]
        pmean_kwargs = data["pmean_kwargs"]
        T_obs = pmean_kwargs["T_obs"]
        master_key = data["key"]

        n_events, n_dim = mean_stack.shape

        params = {name: x[i] for name, i in variables_index.items()}
        model_instance = dist_fn(**constant_params, **params)

        u1_samples = jrd.uniform(
            master_key, (n_samples, n_events, n_dim), minval=lb, maxval=ub
        )
        log_q1 = -jnp.log(ub - lb).sum(axis=-1)

        log_w1 = _importance_sample_log_prob(
            model_instance,
            u1_samples,
            mean_stack,
            scale_tril_stack,
            scale_stack,
            log_q1,
            analytical_to_model_coord_fn,
        )
        log_est_u1 = jnn.logsumexp(log_w1, axis=0, where=jnp.isfinite(log_w1))

        total_ln_l = jnp.sum(log_est_u1 + ln_offsets)

        log_norm = n_events * (jnp.log(n_samples) - jnp.log(T_obs))
        expected_rates = poisson_mean_estimator(model_instance, **pmean_kwargs)

        ln_post = priors.log_prob(x) + total_ln_l - log_norm - expected_rates

        return jnp.nan_to_num(ln_post, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf)

    return likelihood_fn
