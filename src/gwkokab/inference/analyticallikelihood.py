# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from typing import Any, Callable, Dict

import equinox as eqx
import jax
from jax import nn as jnn, numpy as jnp, random as jrd
from jax.scipy.linalg import cho_solve
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import Distribution
from numpyro.distributions.continuous import _batch_mahalanobis, tri_logabsdet
from numpyro.distributions.util import cholesky_of_inverse

from gwkokab.models.utils import JointDistribution, ScaledMixture


@eqx.filter_jit
def mvn_samples(
    loc: Array, scale_tril: Array, n_samples: int, key: PRNGKeyArray
) -> Array:
    """Generate samples from a multivariate normal distribution using method from
    `numpyro.distributions.MultivariateNormal`.

    Parameters
    ----------
    loc : Array
        Mean vector of the multivariate normal distribution.
    cov : Array
        Covariance matrix of the multivariate normal distribution.
    n_samples : int
        Number of samples to generate.
    key : PRNGKeyArray
        JAX random key for sampling.

    Returns
    -------
    Array
        Samples drawn from the multivariate normal distribution.
    """
    eps = jrd.normal(key, shape=(n_samples, *loc.shape))
    samples = loc + jnp.squeeze(jnp.matmul(scale_tril, eps[..., jnp.newaxis]), axis=-1)
    return samples


@jax.jit
@jax.vmap
@ft.partial(jax.vmap, in_axes=(None, None, 0))
def mvn_log_prob_scaled(loc: Array, scale_tril: Array, value: Array) -> Array:
    # removing the constant term -0.5 * D * log(2pi), where D is the dimension,
    # because it is being cancelled out in the importance sampling weights
    M = _batch_mahalanobis(scale_tril, value - loc)
    return -0.5 * M - tri_logabsdet(scale_tril)


@jax.jit
def covariance_matrix_from_cholesky_decomposition(scale_tril: Array) -> Array:
    return jnp.matmul(scale_tril, jnp.swapaxes(scale_tril, -1, -2))


@jax.jit
def precision_matrix_from_cholesky_decomposition(scale_tril: Array) -> Array:
    identity = jnp.broadcast_to(jnp.eye(scale_tril.shape[-1]), scale_tril.shape)
    return cho_solve((scale_tril, True), identity)


@jax.jit
def is_positive_semi_definite(matrix: Array) -> Array:
    """Check if a matrix is positive semi-definite, by verifying that all its
    eigenvalues are non-negative.

    Parameters
    ----------
    matrix : Array
        The matrix of shape (..., N, N) to be checked.

    Returns
    -------
    Array
        Boolean array of shape (...) indicating whether each matrix is positive
        semi-definite.
    """
    return jnp.all(jnp.linalg.eigvals(matrix) >= 0, axis=-1)


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    key: PRNGKeyArray,
    n_events: int,
    n_samples: int = 500,
) -> Callable[[Array, Dict[str, Array]], Array]:
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


    First we are optimizing the mean vector and covariance matrix of the multivariate
    normal distribution that approximates the product of event distributions (also a
    multivariate normal distribution) and phenomenological model distribution by
    Importance Sampling and Moment Matching. After that, we perform Variational
    Inference to optimize the mean vector of the multivariate normal distribution to
    minimize the Reverse KL divergence between the product of event distributions and
    the phenomenological model distribution.

    Parameters
    ----------
    dist_fn : Callable[..., Distribution]
        function that returns a Distribution instance
    priors : JointDistribution
        priors for the model parameters
    variables_index : Dict[str, int]
        mapping of variable names to their indices in the input array
    poisson_mean_estimator : Callable[[ScaledMixture], Array]
        function to compute the expected event rates
    n_events : int
        number of events
    key : PRNGKeyArray
        JAX random key for sampling
    n_samples : int, optional
        Number of samples to draw from the multivariate normal distribution for each
        event to compute the likelihood, by default 500

    Returns
    -------
    Callable[[Array, Array], Array]
        A function that computes the log posterior probability of the model parameters
        given the data. The function takes two arguments: an array of model parameters
        and a second array (not used in this implementation).
    """

    def log_likelihood_fn(x: Array, data: Dict[str, Array]) -> Array:
        mean_stack = data["mean_stack"]
        scale_tril_stack = data["scale_tril_stack"]
        T_obs = data["T_obs"]

        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: Distribution = dist_fn(**constant_params, **mapped_params)

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = poisson_mean_estimator(model_instance)

        rng_key = key

        hessian_log_prob = jax.jit(
            jax.vmap(jax.hessian(model_instance.log_prob), axis_size=n_events)
        )
        grad_log_prob = jax.jit(
            jax.vmap(jax.grad(model_instance.log_prob), axis_size=n_events)
        )

        fit_precision_matrix = precision_matrix_from_cholesky_decomposition(
            scale_tril_stack
        ) - hessian_log_prob(mean_stack)
        fit_covariance_matrix = jnp.linalg.solve(
            fit_precision_matrix, jnp.eye(n_events)
        )

        is_psd = is_positive_semi_definite(fit_covariance_matrix)

        safe_fit_covariance_matrix = jnp.where(
            is_psd[..., jnp.newaxis, jnp.newaxis],
            fit_covariance_matrix,
            jnp.broadcast_to(
                jnp.eye(scale_tril_stack.shape[-1]),
                scale_tril_stack.shape,
            ),
        )

        fit_mean = mean_stack + jnp.where(
            is_psd[..., jnp.newaxis],
            jnp.squeeze(
                safe_fit_covariance_matrix
                @ jnp.expand_dims(grad_log_prob(mean_stack), axis=-1),
                axis=-1,
            ),
            jnp.zeros_like(mean_stack),
        )
        fit_scale_tril: Array = jnp.where(
            is_psd[..., jnp.newaxis, jnp.newaxis],
            cholesky_of_inverse(fit_precision_matrix),
            scale_tril_stack,
        )

        keys = jrd.split(rng_key, n_events)

        # data ~ G(θ, z | μ_f, Σ_f)
        data: Array = jax.vmap(lambda m, s, k: mvn_samples(m, s, n_samples, k))(
            fit_mean, fit_scale_tril, keys
        )

        # log ρ(data | Λ, κ)
        model_instance_log_prob = model_instance.log_prob(data)

        # log G(θ, z | μ_i, Σ_i)
        event_mvn_log_prob = mvn_log_prob_scaled(mean_stack, scale_tril_stack, data)

        # log G(θ, z | μ_f, Σ_i)
        fit_mvn_log_prob = mvn_log_prob_scaled(fit_mean, fit_scale_tril, data)

        log_prob = model_instance_log_prob + event_mvn_log_prob - fit_mvn_log_prob
        log_estimates = jnn.logsumexp(log_prob, where=~jnp.isneginf(log_prob), axis=-1)

        total_log_likelihood = jnp.sum(log_estimates) - n_events * (
            jnp.log(n_samples) - jnp.log(T_obs)
        )

        # log L(Λ,κ) = -μ + Σ log Σ exp (log ρ(data_n|Λ,κ)) - Σ log(M_i)
        log_likelihood = total_log_likelihood - expected_rates
        # log p(Λ,κ|data) = log π(Λ,κ) + log L(Λ,κ)
        log_posterior = priors.log_prob(x) + log_likelihood

        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        return log_posterior

    return eqx.filter_jit(log_likelihood_fn)
