# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import optax
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import Distribution, MultivariateNormal

from gwkokab.models.utils import JointDistribution


@ft.partial(jax.jit, static_argnames=("n_samples",))
def _mvn_samples(loc: Array, cov: Array, n_samples: int, key: PRNGKeyArray) -> Array:
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


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables_index: Dict[str, int],
    ERate_fn: Callable[[Distribution], Array],
    key: PRNGKeyArray,
    n_events: int,
    n_samples: int = 10_000,
    max_iter_mean: int = 10,
    max_iter_cov: int = 3,
    n_vi_steps: int = 5,
    learning_rate: float = 1e-2,
    batch_size: int = 1_000,
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
    ERate_fn : Callable[[Distribution], Array]
        function to compute the expected event rates
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
    max_iter_mean : int, optional
        Maximum number of iterations for the fitting process of the mean, by default 10
    max_iter_cov : int, optional
        Maximum number of iterations for the fitting process of the covariance, by default 3
    n_vi_steps: int, optional
        Number of steps for the variational inference optimization, by default 5
    learning_rate : float, optional
        Learning rate for the Adam optimizer used in the variational inference
        optimization, by default 1e-2
    batch_size : int, optional
        Batch size for the sampling process, by default 1_000

    Returns
    -------
    Callable[[Array, Array], Array]
        A function that computes the log posterior probability of the model parameters
        given the data. The function takes two arguments: an array of model parameters
        and a second array (not used in this implementation).
    """

    def likelihood_fn(x: Array, data: Dict[str, Any]) -> Array:
        mean_stack: Array = data["mean_stack"]
        cov_stack: Array = data["cov_stack"]
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: Distribution = dist_fn(**mapped_params)

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = ERate_fn(model_instance)

        event_mvn = MultivariateNormal(loc=mean_stack, covariance_matrix=cov_stack)

        rng_key = key
        moment_matching_mean = mean_stack
        moment_matching_cov = event_mvn.covariance_matrix

        model_log_prob_vmap_fn = jax.vmap(
            model_instance.log_prob, in_axes=(1,), out_axes=-1
        )

        normalized_weights_fn = lambda samples: jax.nn.softmax(
            jnp.expand_dims(
                model_log_prob_vmap_fn(samples) + event_mvn.log_prob(samples),
                axis=-1,
            ),
            axis=0,
        )

        @jax.jit
        def scan_moment_matching_mean_fn(
            moment_matching_mean_i: Array, key: PRNGKeyArray
        ) -> Tuple[Array, None]:
            samples = _mvn_samples(
                loc=moment_matching_mean_i,
                cov=moment_matching_cov,
                n_samples=1_000,
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
            moment_matching_mean,
            jrd.split(subkey, max_iter_mean),
            length=max_iter_mean,
        )

        @jax.jit
        def scan_moment_matching_cov_fn(
            moment_matching_cov_i: Array, key: PRNGKeyArray
        ) -> Tuple[Array, None]:
            samples = _mvn_samples(
                loc=moment_matching_mean,
                cov=moment_matching_cov_i,
                n_samples=1_000,
                key=key,
            )

            # Normalized weights = softmax(log ρ(samples | Λ, κ) + log G(θ, z | μ_i, Σ_i)))
            weights = normalized_weights_fn(samples)

            # Σ_f = sum(weights * (samples - μ_f) * (samples - μ_f).T)
            centered = samples - moment_matching_mean
            new_fit_cov = jnp.einsum("sei,sej->eij", weights * centered, centered)

            return new_fit_cov, None

        rng_key, subkey = jrd.split(rng_key)
        moment_matching_cov, _ = jax.lax.scan(
            scan_moment_matching_cov_fn,  # type: ignore[arg-type]
            moment_matching_cov,
            jrd.split(subkey, max_iter_cov),
            length=max_iter_cov,
        )

        @jax.jit
        def loss_fn(mu: Array, cov: Array, key: PRNGKeyArray) -> Array:
            """Compute Reverse KL divergence between the model and the fitted
            multivariate normal distribution.

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
            model_samples = model_instance.sample(key=key, sample_shape=(1_000,))

            log_p = model_instance.log_prob(model_samples)

            return fit_dist.entropy() - jnp.mean(
                jnp.exp(fit_dist.log_prob(model_samples) - log_p) * log_p
            )

        @jax.jit
        def step(
            mu: Array,
            cov: Array,
            opt_state: optax.OptState,
            key: PRNGKeyArray,
        ) -> Tuple[Array, optax.OptState]:
            """Perform a single optimization step to update the mean vector to minimize
            the loss function.

            Parameters
            ----------
            mu : Array
                mean vector of the multivariate normal distribution
            cov : Array
                covariance matrix of the multivariate normal distribution
            opt_state : optax.OptState
                optimizer state
            key : PRNGKeyArray
                random key for sampling

            Returns
            -------
            Tuple[Array, optax.OptState]
                updated mean vector and optimizer state
            """
            keys = jrd.split(key, n_events)
            _, grads = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(0, 0, 0))(
                mu, cov, keys
            )
            updates, opt_state = opt.update(grads, opt_state)
            mu = optax.apply_updates(mu, updates)
            return mu, opt_state

        @jax.jit
        def variational_inference_fn(
            carry: Tuple[Array, Array, optax.OptState, Array], _: Optional[Array]
        ) -> Tuple[Tuple[Array, Array, optax.OptState, Array], None]:
            """Perform a single step of the gradient descent optimization to update the
            mean vector and covariance matrix.

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
            mu, opt_state = step(mu, cov, opt_state, subkey)
            return (mu, cov, opt_state, key), None

        vi_mean = moment_matching_mean
        opt = optax.adam(learning_rate=learning_rate)
        opt_state = opt.init(vi_mean)

        rng_key, subkey = jrd.split(rng_key)

        (vi_mean, _, _, _), _ = jax.lax.scan(
            variational_inference_fn,
            (vi_mean, moment_matching_cov, opt_state, subkey),
            length=n_vi_steps,
        )

        fit_mean = jnp.where(jnp.isnan(vi_mean), moment_matching_mean, vi_mean)
        fit_cov = moment_matching_cov

        @jax.jit
        def scan_fn(
            carry: Array, loop_data: Tuple[Array, Array, Array, Array, PRNGKeyArray]
        ) -> Tuple[Array, None]:
            mean, cov, fit_mean_i, fit_cov_i, rng_key_i = loop_data

            # event_mvn = G(θ, z | μ_i, Σ_i)
            event_mvn = MultivariateNormal(loc=mean, covariance_matrix=cov)

            # fit_mvn = G(θ, z | μ_f, Σ_f)
            fit_mvn = MultivariateNormal(loc=fit_mean_i, covariance_matrix=fit_cov_i)

            # data ~ G(θ, z | μ_f, Σ_f)
            data = fit_mvn.sample(rng_key_i, (n_samples,))

            # log ρ(data | Λ, κ)
            model_instance_log_prob = jax.lax.map(
                model_instance.log_prob, data, batch_size=batch_size
            )
            # log G(θ, z | μ_i, Σ_i)
            event_mvn_log_prob = jax.lax.map(
                event_mvn.log_prob, data, batch_size=batch_size
            )
            # log G(θ, z | μ_f, Σ_i)
            fit_mvn_log_prob = jax.lax.map(
                fit_mvn.log_prob, data, batch_size=batch_size
            )

            # log_prob = log ρ(data | Λ, κ) + log G(θ, z | μ_i, Σ_i) - log G(θ, z | μ_f, Σ_i)
            log_prob = model_instance_log_prob + event_mvn_log_prob - fit_mvn_log_prob
            log_likelihood = jax.nn.logsumexp(
                log_prob, axis=-1, where=(~jnp.isneginf(log_prob))
            )
            return carry + log_likelihood, None

        keys = jrd.split(rng_key, (n_events,))

        total_log_likelihood, _ = jax.lax.scan(
            scan_fn,  # type: ignore[arg-type]
            -n_events * jnp.log(n_samples),  # - Σ log(M_i)
            (mean_stack, cov_stack, fit_mean, fit_cov, keys),
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
