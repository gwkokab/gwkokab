# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from typing import Any, Callable, Dict, Tuple

import jax
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import Distribution
from numpyro.distributions.continuous import _batch_mahalanobis, tri_logabsdet

from gwkokab.models.utils import JointDistribution, ScaledMixture


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
    eps = jrd.normal(key, shape=(n_samples, *loc.shape))
    samples = loc + jnp.squeeze(jnp.matmul(scale_tril, eps[..., jnp.newaxis]), axis=-1)
    return samples


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
    normalize_term = half_log_det + 0.5 * scale_tril.shape[-1] * jnp.log(2 * jnp.pi)
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


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    key: PRNGKeyArray,
    n_events: int,
    n_samples: int = 500,
    n_mom_samples: int = 500,
    max_iter_mean: int = 10,
    max_iter_cov: int = 3,
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

    @jax.jit
    def likelihood_fn(x: Array, data: Dict[str, Any]) -> Array:
        mean_stack: Array = data["mean_stack"]
        scale_tril_stack: Array = data["scale_tril_stack"]
        lower_bounds: Array = data["lower_bounds"]
        upper_bounds: Array = data["upper_bounds"]
        ln_offsets: Array = data["ln_offsets"]
        pmean_kwargs: Dict[str, Any] = data["pmean_kwargs"]
        T_obs: Array = pmean_kwargs["T_obs"]

        mapped_params = {name: x[i] for name, i in variables_index.items()}

        model_instance: Distribution = dist_fn(**constant_params, **mapped_params)

        rng_key = key

        def normalized_weights_fn(
            mean: Array,
            scale_tril: Array,
            samples: Array,
            lb: Array,
            ub: Array,
            ln_offsets: Array,
        ) -> Array:
            log_p_model = model_instance.log_prob(samples)
            log_p_event = mvn_log_prob(mean, scale_tril, samples) + ln_offsets
            is_inside = jnp.all((samples >= lb) & (samples <= ub), axis=-1)
            log_p_model = jnp.where(is_inside, log_p_event, -jnp.inf)
            log_weights = log_p_model + log_p_event
            log_weights = jnp.expand_dims(log_weights, axis=-1)
            return jax.nn.softmax(log_weights, axis=0)

        @jax.jit
        def scan_moment_matching_mean_fn(
            carry_in: tuple[Any, Array, Any, Any, Any], key: PRNGKeyArray
        ) -> Tuple[tuple[Any, Array, Any, Any, Any], None]:
            (
                moment_matching_mean_i,
                scale_tril_stack,
                lower_bounds,
                upper_bounds,
                ln_offsets,
            ) = carry_in
            samples = mvn_samples(
                loc=moment_matching_mean_i,
                scale_tril=scale_tril_stack,
                n_samples=n_mom_samples,
                key=key,
            )

            # Normalized weights = softmax(log ρ(samples | Λ, κ) + log G(θ, z | μ_i, Σ_i)))
            weights = normalized_weights_fn(
                moment_matching_mean_i,
                scale_tril_stack,
                samples,
                lower_bounds,
                upper_bounds,
                ln_offsets,
            )

            # μ_f = sum(weights * samples)
            new_fit_mean = jnp.sum(samples * weights, axis=0)

            carry_out = (
                new_fit_mean,
                scale_tril_stack,
                lower_bounds,
                upper_bounds,
                ln_offsets,
            )

            return carry_out, None

        rng_key, subkey = jrd.split(rng_key)

        (fit_mean_stack, _, _, _, _), _ = jax.lax.scan(
            scan_moment_matching_mean_fn,  # type: ignore[arg-type]
            (
                mean_stack,
                scale_tril_stack,
                lower_bounds,
                upper_bounds,
                ln_offsets,
            ),
            jrd.split(subkey, max_iter_mean),
            length=max_iter_mean,
        )

        @jax.jit
        def scan_moment_matching_scale_tril_fn(
            carry_in: tuple[Any, Array, Any, Any, Any], key: PRNGKeyArray
        ) -> Tuple[tuple[Any, Array, Any, Any, Any], None]:
            (
                mean_stack,
                moment_matching_scale_tril_i,
                lower_bounds,
                upper_bounds,
                ln_offsets,
            ) = carry_in
            samples = mvn_samples(
                loc=mean_stack,
                scale_tril=moment_matching_scale_tril_i,
                n_samples=n_mom_samples,
                key=key,
            )

            # Normalized weights = softmax(log ρ(samples | Λ, κ) + log G(θ, z | μ_i, Σ_i)))
            weights = normalized_weights_fn(
                mean_stack,
                moment_matching_scale_tril_i,
                samples,
                lower_bounds,
                upper_bounds,
                ln_offsets,
            )

            # Σ_f = sum(weights * (samples - μ_f) * (samples - μ_f).T)
            centered = samples - mean_stack

            new_fit_cov = jnp.einsum("sei,sej->eij", weights * centered, centered)

            carry_out = (
                mean_stack,
                cholesky_decomposition(new_fit_cov),
                lower_bounds,
                upper_bounds,
                ln_offsets,
            )

            return carry_out, None

        rng_key, subkey = jrd.split(rng_key)

        (_, fit_scale_tril_stack, _, _, _), _ = jax.lax.scan(
            scan_moment_matching_scale_tril_fn,  # type: ignore[arg-type]
            (
                fit_mean_stack,
                scale_tril_stack,
                lower_bounds,
                upper_bounds,
                ln_offsets,
            ),
            jrd.split(subkey, max_iter_cov),
            length=max_iter_cov,
        )

        samples: Array = mvn_samples(
            fit_mean_stack, fit_scale_tril_stack, n_samples, key
        )

        is_inside = jnp.all(
            (samples >= lower_bounds) & (samples <= upper_bounds), axis=-1
        )

        # log ρ(data | Λ, κ)
        log_p_model = model_instance.log_prob(samples)

        # log G(θ, z | μ_i, Σ_i)
        log_p_event = mvn_log_prob(mean_stack, scale_tril_stack, samples)

        # log G(θ, z | μ_f, Σ_i)
        log_q_fit = mvn_log_prob(fit_mean_stack, fit_scale_tril_stack, samples)

        log_weights = log_p_model + log_p_event - log_q_fit

        # Mask points outside the hypercube by setting log_prob to -inf
        log_weights = jnp.where(is_inside, log_weights, -jnp.inf)

        # Perform logsumexp over the samples (axis -1 of the generated samples)
        log_estimates = jnn.logsumexp(
            log_weights + ln_offsets, where=~jnp.isneginf(log_weights), axis=0
        )

        total_log_likelihood = jnp.sum(log_estimates) - n_events * (
            jnp.log(n_samples) - jnp.log(T_obs)
        )

        expected_rates = poisson_mean_estimator(model_instance, **pmean_kwargs)

        log_likelihood = total_log_likelihood - expected_rates

        log_posterior = priors.log_prob(x) + log_likelihood

        return jnp.nan_to_num(
            log_posterior, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
        )

    return likelihood_fn
