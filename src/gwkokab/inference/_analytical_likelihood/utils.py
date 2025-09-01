# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from typing import Callable, Optional, Tuple

import jax
import optax
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro._typing import DistributionT
from numpyro.distributions import MultivariateNormal


@jax.jit
def monte_carlo_log_estimate_and_error(
    log_probs: Array, N: Array
) -> Tuple[Array, Array]:
    """Computes the Monte Carlo estimate and error for the given log probabilities.

    Parameters
    ----------
    log_probs : Array
        Log probabilities of the samples.
    N : int
        Number of samples used for the estimate.

    Returns
    -------
    Tuple[Array, Array]
        Monte Carlo logarithm of estimate, and Monte Carlo error.
    """
    mask = ~jnp.isneginf(log_probs)
    log_moment_1 = jnn.logsumexp(log_probs, where=mask, axis=-1) - jnp.log(N)
    moment_2 = jnp.exp(jnn.logsumexp(2.0 * log_probs, where=mask, axis=-1)) / N
    error = jnp.sqrt((moment_2 - jnp.exp(2.0 * log_moment_1)) / (N - 1.0))
    return log_moment_1, error


@jax.jit
def combine_monte_carlo_log_estimates(
    log_estimates_1: Array, log_estimates_2: Array, N_1: int, N_2: int
) -> Array:
    r"""Combine two Monte Carlo estimates into a single estimate using the formula:

    .. math::

        \hat{\mu} = \frac{N_1 \hat{\mu}_1 + N_2 \hat{\mu}_2}{N_1 + N_2}

    Parameters
    ----------
    estimates_1 : Array
        First Monte Carlo estimate :math:`\hat{\mu}_1`.
    estimates_2 : Array
        Second Monte Carlo estimate :math:`\hat{\mu}_2`.
    N_1 : int
        Number of samples used for the first estimate :math:`N_1`.
    N_2 : int
        Number of samples used for the second estimate :math:`N_2`.

    Returns
    -------
    Array
        Combined Monte Carlo estimate :math:`\hat{\mu}`.
    """
    combined_log_estimate = jnp.logaddexp(
        jnp.log(N_1) + log_estimates_1, jnp.log(N_2) + log_estimates_2
    ) - jnp.log(N_1 + N_2)
    return combined_log_estimate


@jax.jit
def combine_monte_carlo_errors(
    error_1: Array,
    error_2: Array,
    log_estimate_1: Array,
    log_estimate_2: Array,
    log_estimate_3: Array,
    N_1: int,
    N_2: int,
) -> Array:
    r"""Combine two Monte Carlo errors into a single error estimate using the formula:

    .. math::

        \hat{\epsilon}=\sqrt{\frac{1}{N_3(N_3-1)}\sum_{k=1}^{2}\left\{N_k(N_k-1)\hat{\epsilon}_k^2+N_k\hat{\mu}^2_k\right\}-\frac{1}{N_3-1}\hat{\mu}^2}

    where, :math:`N_3 = N_1 + N_2` is the total number of samples.

    _extended_summary_

    Parameters
    ----------
    error_1 : Array
        Error of the first Monte Carlo estimate :math:`\hat{\epsilon}_1`.
    error_2 : Array
        Error of the second Monte Carlo estimate :math:`\hat{\epsilon}_2`.
    log_estimate_1 : Array
        Estimate of the first Monte Carlo estimate :math:`\hat{\mu}_1`.
    log_estimate_2 : Array
        Estimate of the second Monte Carlo estimate :math:`\hat{\mu}_2`.
    log_estimate_3 : Array
        Estimate of the combined Monte Carlo estimate :math:`\hat{\mu}`.
    N_1 : int
        Number of samples used for the first estimate :math:`N_1`.
    N_2 : int
        Number of samples used for the second estimate :math:`N_2`.

    Returns
    -------
    Array
        Combined Monte Carlo error estimate :math:`\hat{\epsilon}`.
    """
    N_3 = N_1 + N_2

    sum_prob_sq_1 = N_1 * (
        (N_1 - 1.0) * jnp.square(error_1) + jnp.exp(2.0 * log_estimate_1)
    )
    sum_prob_sq_2 = N_2 * (
        (N_2 - 1.0) * jnp.square(error_2) + jnp.exp(2.0 * log_estimate_2)
    )

    combined_error_sq = -jnp.exp(2.0 * log_estimate_3) / (N_3 - 1.0)
    combined_error_sq += (sum_prob_sq_1 + sum_prob_sq_2) / N_3 / (N_3 - 1.0)

    combined_error = jnp.sqrt(combined_error_sq)

    return combined_error


@ft.partial(jax.jit, static_argnames=("n_samples",))
def mvn_samples(loc: Array, cov: Array, n_samples: int, key: PRNGKeyArray) -> Array:
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
    scale_tril = jnp.linalg.cholesky(cov)
    eps = jrd.normal(key, shape=(n_samples, *loc.shape))
    samples = loc + jnp.squeeze(jnp.matmul(scale_tril, eps[..., jnp.newaxis]), axis=-1)
    return samples


def moment_match_mean(
    rng_key: PRNGKeyArray,
    mean: Array,
    cov: Array,
    normalized_weights_fn: Callable[[Array], Array],
    max_iter: int,
    n_samples: int,
) -> Array:
    @jax.jit
    def scan_moment_matching_mean_fn(
        moment_matching_mean_i: Array, key: PRNGKeyArray
    ) -> Tuple[Array, None]:
        samples = mvn_samples(
            loc=moment_matching_mean_i,
            cov=cov,
            n_samples=n_samples,
            key=key,
        )

        # Normalized weights = softmax(log ρ(samples | Λ, κ) + log G(θ, z | μ_i, Σ_i)))
        weights = normalized_weights_fn(samples)

        # μ_f = sum(weights * samples)
        new_fit_mean = jnp.sum(samples * weights, axis=0)

        return new_fit_mean, None

    rng_key, subkey = jrd.split(rng_key)

    moment_matching_mean, _ = jax.lax.scan(
        scan_moment_matching_mean_fn,  # type: ignore[arg-type]
        mean,
        jrd.split(subkey, max_iter),
        length=max_iter,
    )
    return moment_matching_mean


def moment_match_cov(
    rng_key: PRNGKeyArray,
    mean: Array,
    cov: Array,
    normalized_weights_fn: Callable[[Array], Array],
    max_iter: int,
    n_samples: int,
) -> Array:
    @jax.jit
    def scan_moment_matching_cov_fn(
        moment_matching_cov_i: Array, key: PRNGKeyArray
    ) -> Tuple[Array, None]:
        samples = mvn_samples(
            loc=mean,
            cov=moment_matching_cov_i,
            n_samples=n_samples,
            key=key,
        )

        # Normalized weights = softmax(log ρ(samples | Λ, κ) + log G(θ, z | μ_i, Σ_i)))
        weights = normalized_weights_fn(samples)

        # Σ_f = sum(weights * (samples - μ_f) * (samples - μ_f).T)
        centered = samples - mean
        new_fit_cov = jnp.einsum("sei,sej->eij", weights * centered, centered)

        return new_fit_cov, None

    rng_key, subkey = jrd.split(rng_key)
    moment_matching_cov, _ = jax.lax.scan(
        scan_moment_matching_cov_fn,  # type: ignore[arg-type]
        cov,
        jrd.split(subkey, max_iter),
        length=max_iter,
    )
    return moment_matching_cov


def match_mean_by_variational_inference(
    rng_key: PRNGKeyArray,
    mean: Array,
    cov: Array,
    model: DistributionT,
    learning_rate: float,
    steps: int,
    n_samples: int,
) -> Array:
    n_events = mean.shape[0]

    @ft.partial(jax.vmap, in_axes=(0, 0, 0))
    @jax.value_and_grad
    @jax.jit
    def loss_fn(mu: Array, cov: Array, key: PRNGKeyArray) -> Array:
        """Compute Reverse KL divergence between the model and the fitted multivariate
        normal distribution.

        Parameters
        ----------
        mu : Array
            mean vector of the multivariate normal distribution
        cov : Array
            covariance matrix of the multivariate normal distribution
        key : PRNGKeyArray
            random key for sampling

        Returns
        -------
        Array
            loss value
        """
        fit_dist = MultivariateNormal(loc=mu, covariance_matrix=cov)
        model_samples = model.sample(key=key, sample_shape=(n_samples,))

        log_p = model.log_prob(model_samples)

        return fit_dist.entropy() - jnp.mean(
            jnp.exp(fit_dist.log_prob(model_samples) - log_p) * log_p
        )

    @jax.jit
    def variational_inference_fn(
        carry: Tuple[Array, Array, optax.OptState, Array], _: Optional[Array]
    ) -> Tuple[Tuple[Array, Array, optax.OptState, Array], None]:
        """Perform a single step of the gradient descent optimization to update the mean
        vector and covariance matrix.

        Parameters
        ----------
        carry : Tuple[Array, Array, optax.OptState, Array]
            carry tuple containing the current mean vector, covariance matrix,
            optimizer state, and random key
        _ : Optional[Array]
            unused placeholder for the scan function

        Returns
        -------
        Tuple[Tuple[Array, Array, optax.OptState, Array], None]
            updated mean vector, unchanged covariance matrix, optimizer state, and
            random key
        """
        mu, cov, opt_state, key = carry
        key, subkey = jrd.split(key)
        keys = jrd.split(subkey, n_events)
        # Perform a single optimization step to update the mean vector to minimize the
        # loss function.
        _, grads = loss_fn(mu, cov, keys)
        updates, opt_state = opt.update(grads, opt_state)
        mu = optax.apply_updates(mu, updates)
        return (mu, cov, opt_state, key), None

    vi_mean = mean
    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(vi_mean)

    rng_key, subkey = jrd.split(rng_key)

    (vi_mean, _, _, _), _ = jax.lax.scan(
        variational_inference_fn,
        (vi_mean, cov, opt_state, subkey),
        length=steps,
    )

    return vi_mean
