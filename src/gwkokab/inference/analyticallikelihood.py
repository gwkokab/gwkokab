# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from typing import Callable, Dict, List, Optional, Tuple

import jax
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger
from numpyro.distributions import Distribution, MultivariateNormal

from gwkokab.models.utils import JointDistribution


@ft.partial(jax.jit, static_argnames=("n_samples",))
def _mvn_samples(
    loc: Array, scale_tril: Array, n_samples: int, key: PRNGKeyArray
) -> Array:
    """Generate samples from a multivariate normal distribution using method from
    `numpyro.distributions.MultivariateNormal`.

    Parameters
    ----------
    loc : Array
        Mean vector of the multivariate normal distribution.
    scale_tril : Array
        Lower triangular matrix of the covariance matrix (Cholesky decomposition).
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


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables_index: Dict[str, int],
    ERate_fn: Callable[[Distribution, Optional[int]], Array],
    redshift_index: Optional[int],
    means: List[Array],
    covariances: List[Array],
    key: PRNGKeyArray,
    n_samples: int = 10_000,
    max_iter: int = 5,
) -> Callable[[Array, Array], Array]:
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

    Parameters
    ----------
    dist_fn : Callable[..., Distribution]
        function that returns a Distribution instance
    priors : JointDistribution
        priors for the model parameters
    variables_index : Dict[str, int]
        mapping of variable names to their indices in the input array
    ERate_fn : Callable[[Distribution, Optional[int]], Array]
        function to compute the expected event rates
    redshift_index : Optional[int]
        index of the redshift variable in the input array, if applicable
    means : List[Array]
        List of mean vectors for each event. Each vector should have the same length and
        should correspond to the parameters of the distribution along with covariances.
    covariances : List[Array]
        List of covariance matrices for each event. Each matrix should correspond to the
        mean vectors in `means`.
    key : PRNGKeyArray
        JAX random key for sampling
    n_samples : int, optional
        Number of samples to draw from the multivariate normal distribution for each
        event to compute the likelihood, by default 10_000
    max_iter : int, optional
        Maximum number of iterations for the fitting process, by default 5

    Returns
    -------
    Callable[[Array, Array], Array]
        A function that computes the log posterior probability of the model parameters
        given the data. The function takes two arguments: an array of model parameters
        and a second array (not used in this implementation).
    """
    n_events = len(means)
    mean_stack: Array = jax.block_until_ready(
        jax.device_put(jnp.stack(means, axis=0), may_alias=True)
    )
    cov_stack: Array = jax.block_until_ready(
        jax.device_put(jnp.stack(covariances, axis=0), may_alias=True)
    )
    logger.debug("mean_stack.shape: {shape}", shape=mean_stack.shape)
    logger.debug("cov_stack.shape: {shape}", shape=cov_stack.shape)

    def likelihood_fn(x: Array, _: Array) -> Array:
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: Distribution = dist_fn(**mapped_params)

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = ERate_fn(model_instance, redshift_index)

        event_mvn = MultivariateNormal(loc=mean_stack, covariance_matrix=cov_stack)

        rng_key = key
        fit_mean = mean_stack
        # Σ_f = Σ_i
        fit_scale_tril = event_mvn.scale_tril  # not changing the scale_tril here

        @jax.jit
        def scan_fit_fn(fit_mean: Array, key: PRNGKeyArray) -> Tuple[Array, None]:
            fit_samples = _mvn_samples(
                loc=fit_mean,
                scale_tril=fit_scale_tril,
                n_samples=1_000,
                key=key,
            )

            # weights = ρ(fit_samples | Λ, κ) * G(θ, z | μ_i, Σ_i))
            weights = jnp.expand_dims(
                jnp.exp(
                    jax.vmap(model_instance.log_prob, in_axes=(1,), out_axes=-1)(
                        fit_samples
                    )
                    + event_mvn.log_prob(fit_samples)
                ),
                axis=-1,
            )
            # μ_f = sum(weights * fit_samples) / sum(weights)
            new_fit_mean = jnp.sum(fit_samples * weights, axis=0) / jnp.sum(
                weights, axis=0
            )
            return new_fit_mean, None

        fit_mean, _ = jax.lax.scan(
            scan_fit_fn,  # type: ignore[arg-type]
            fit_mean,
            jrd.split(rng_key, max_iter),
            length=max_iter,
        )

        @jax.jit
        def scan_fn(
            carry: Array, loop_data: Tuple[Array, Array, Array, PRNGKeyArray]
        ) -> Tuple[Array, None]:
            mean, cov, fit_mean, rng_key = loop_data

            # event_mvn = G(θ, z | μ_i, Σ_i)
            event_mvn = MultivariateNormal(loc=mean, covariance_matrix=cov)

            # fit_mvn = G(θ, z | μ_f, Σ_i)
            fit_mvn = MultivariateNormal(loc=fit_mean, covariance_matrix=cov)
            # data ~ G(θ, z | μ_f, Σ_f)
            data = fit_mvn.sample(rng_key, (n_samples,))

            # log_prob = log ρ(data | Λ, κ) + log G(θ, z | μ_i, Σ_i) - log G(θ, z | μ_f, Σ_i)
            log_prob = (
                event_mvn.log_prob(data)
                + model_instance.log_prob(data)
                - fit_mvn.log_prob(data)
            )
            carry += jax.nn.logsumexp(
                log_prob, axis=-1, where=(~jnp.isneginf(log_prob))
            )
            return carry, None

        keys = jrd.split(rng_key, (n_events,))

        total_log_likelihood, _ = jax.lax.scan(
            scan_fn,  # type: ignore[arg-type]
            -n_events * jnp.log(n_samples),  # - Σ log(M_i)
            (mean_stack, cov_stack, fit_mean, keys),
            length=n_events,
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

    return likelihood_fn
