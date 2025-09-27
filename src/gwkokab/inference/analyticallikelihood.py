# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from typing import Callable, Dict, Optional, Tuple, TypeAlias

import equinox as eqx
import jax
import optax
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro._typing import DistributionT
from numpyro.distributions import Distribution
from numpyro.distributions.continuous import _batch_mahalanobis, tri_logabsdet

from gwkokab.models.utils import JointDistribution, ScaledMixture


StateT: TypeAlias = Tuple[
    Array,  # old monte-carlo-estimate
    Array,  # old error
    Array,  # old size
    PRNGKeyArray,  # old key
]
"""State of the Monte Carlo estimation process."""


@jax.jit
def monte_carlo_log_estimate_and_error(
    log_probs: Array, N: Array
) -> Tuple[Array, Array]:
    """Computes the Monte Carlo estimate and error for the given log probabilities.

    Parameters
    ----------
    log_probs : Array
        Log probabilities of the samples.
    N : Array
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
    log_estimates_1: Array, log_estimates_2: Array, N_1: Array, N_2: Array
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
    N_1 : Array
        Number of samples used for the first estimate :math:`N_1`.
    N_2 : Array
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
    N_1: Array,
    N_2: Array,
) -> Array:
    r"""Combine two Monte Carlo errors into a single error estimate using the formula:

    .. math::

        \hat{\epsilon}=\sqrt{\frac{1}{N_3(N_3-1)}\sum_{k=1}^{2}\left\{N_k(N_k-1)\hat{\epsilon}_k^2+N_k\hat{\mu}^2_k\right\}-\frac{1}{N_3-1}\hat{\mu}^2}

    where, :math:`N_3 = N_1 + N_2` is the total number of samples.

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
    N_1 : Array
        Number of samples used for the first estimate :math:`N_1`.
    N_2 : Array
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


@eqx.filter_jit
def mvn_log_prob(loc: Array, scale_tril: Array, value: Array) -> Array:
    M = _batch_mahalanobis(scale_tril, value - loc)
    half_log_det = tri_logabsdet(scale_tril)
    normalize_term = half_log_det + 0.5 * scale_tril.shape[-1] * jnp.log(2 * jnp.pi)
    return -0.5 * M - normalize_term


@ft.partial(jax.jit, static_argnames=("max_iter", "n_samples", "normalized_weights_fn"))
def moment_match_mean(
    rng_key: PRNGKeyArray,
    mean: Array,
    scale_tril: Array,
    normalized_weights_fn: Callable[[Array], Array],
    max_iter: int,
    n_samples: int,
) -> Array:
    def scan_moment_matching_mean_fn(
        moment_matching_mean_i: Array, key: PRNGKeyArray
    ) -> Tuple[Array, None]:
        samples = mvn_samples(
            loc=moment_matching_mean_i,
            scale_tril=scale_tril,
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


@ft.partial(jax.jit, static_argnames=("max_iter", "n_samples", "normalized_weights_fn"))
def moment_match_cov(
    rng_key: PRNGKeyArray,
    mean: Array,
    cov: Array,
    normalized_weights_fn: Callable[[Array], Array],
    max_iter: int,
    n_samples: int,
) -> Array:
    def scan_moment_matching_cov_fn(
        moment_matching_cov_i: Array, key: PRNGKeyArray
    ) -> Tuple[Array, None]:
        samples = mvn_samples(
            loc=mean,
            scale_tril=jnp.linalg.cholesky(moment_matching_cov_i),
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


def kl_divergence_of_two_mvn_with_same_cov(
    mean_1: Array, mean_2: Array, cov: Array
) -> Array:
    r"""Compute the KL divergence between two multivariate normal distributions.

    .. math::

        D_{KL}(\mathcal{N}_1||\mathcal{N}_2)=\frac{1}{2}(\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1)

    Parameters
    ----------
    mean_1 : Array
        Mean vector of the first multivariate normal distribution.
    mean_2 : Array
        Mean vector of the second multivariate normal distribution.
    cov : Array
        Covariance matrix of any multivariate normal distribution.

    Returns
    -------
    Array
        KL divergence between the two multivariate normal distributions.
    """
    inv_cov = jnp.linalg.inv(cov)
    diff = mean_2 - mean_1
    mahalanobis_term = jnp.einsum("...i,...ij,...j->...", diff, inv_cov, diff)
    kl_div = 0.5 * mahalanobis_term
    return kl_div


def match_mean_by_variational_inference(
    rng_key: PRNGKeyArray,
    event_mean: Array,
    mean: Array,
    scale_tril: Array,
    model: DistributionT,
    learning_rate: float,
    steps: int,
    n_samples: int,
) -> Array:
    n_events = mean.shape[0]

    @ft.partial(jax.vmap, in_axes=(0, 0, 0, 0))
    @jax.value_and_grad
    def loss_fn(
        mu: Array, mean_i: Array, scale_tril: Array, key: PRNGKeyArray
    ) -> Array:
        """Compute Reverse KL divergence between the model and the fitted multivariate
        normal distribution.

        Parameters
        ----------
        mu : Array
            mean vector of the multivariate normal distribution
        scale_tril : Array
            Cholesky factor of the covariance matrix of the multivariate normal distribution
        key : PRNGKeyArray
            random key for sampling

        Returns
        -------
        Array
            loss value
        """
        model_samples = model.sample(key, sample=(n_samples,))
        log_p = model.log_prob(model_samples)

        fit_dist_log_prob = mvn_log_prob(mu, scale_tril, model_samples)

        return kl_divergence_of_two_mvn_with_same_cov(
            mu, mean_i, scale_tril @ scale_tril.T
        ) - jnp.mean(jnp.exp(fit_dist_log_prob - log_p) * log_p)

    def variational_inference_fn(
        carry: Tuple[Array, Array, Array, optax.OptState, Array], _: Optional[Array]
    ) -> Tuple[Tuple[Array, Array, Array, optax.OptState, Array], None]:
        """Perform a single step of the gradient descent optimization to update the mean
        vector and covariance matrix.

        Parameters
        ----------
        carry : Tuple[Array, Array, Array, optax.OptState, Array]
            carry tuple containing the current mean vector, event mean vector, covariance matrix,
            optimizer state, and random key
        _ : Optional[Array]
            unused placeholder for the scan function

        Returns
        -------
        Tuple[Tuple[Array, Array, optax.OptState, Array], None]
            updated mean vector, unchanged covariance matrix, optimizer state, and
            random key
        """
        mu, mean_i, scale_tril, opt_state, key = carry
        key, subkey = jrd.split(key)
        keys = jrd.split(subkey, n_events)
        # Perform a single optimization step to update the mean vector to minimize the
        # loss function.
        _, grads = loss_fn(mu, mean_i, scale_tril, keys)
        updates, opt_state = opt.update(grads, opt_state)
        mu = optax.apply_updates(mu, updates)
        return (mu, mean_i, scale_tril, opt_state, key), None

    vi_mean = mean
    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(vi_mean)

    rng_key, subkey = jrd.split(rng_key)

    (vi_mean, _, _, _, _), _ = jax.lax.scan(
        variational_inference_fn,
        (vi_mean, event_mean, scale_tril, opt_state, subkey),
        length=steps,
    )

    return vi_mean


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    key: PRNGKeyArray,
    n_events: int,
    minimum_mc_error: float = 1e-2,
    n_samples: int = 100,
    max_iter_mean: int = 10,
    max_iter_cov: int = 3,
    n_vi_steps: int = 5,
    learning_rate: float = 1e-2,
    batch_size: int = 1_00,
    n_checkpoints: int = 10,
    n_max_steps: int = 20,
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
    minimum_mc_error : float, optional
        Minimum threshold for Monte Carlo error, by default 1e-2
    n_samples : int, optional
        Number of samples to draw from the multivariate normal distribution for each
        event to compute the likelihood, by default 100
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
        Batch size for the sampling process, by default 100
    n_checkpoints : int, optional
        Number of checkpoints to save during the optimization process, by default 1
    n_max_steps : int, optional
        Maximum number of steps for the optimization process, by default 20

    Returns
    -------
    Callable[[Array, Array], Array]
        A function that computes the log posterior probability of the model parameters
        given the data. The function takes two arguments: an array of model parameters
        and a second array (not used in this implementation).
    """

    def log_likelihood_fn(x: Array, data: Dict[str, Array]) -> Array:
        mean_stack = data["mean_stack"]
        cov_stack = data["cov_stack"]
        scale_tril_stack = data["scale_tril_stack"]
        T_obs = data["T_obs"]

        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: DistributionT = dist_fn(**mapped_params)

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = poisson_mean_estimator(model_instance)

        rng_key = key
        moment_matching_mean = mean_stack
        moment_matching_cov = cov_stack
        moment_matching_scale_tril = scale_tril_stack

        def normalized_weights_fn(samples: Array) -> Array:
            model_log_prob_vmap_fn = jax.vmap(
                model_instance.log_prob,
                in_axes=(1,),
                out_axes=-1,
            )
            model_log_prob = model_log_prob_vmap_fn(samples)
            safe_model_log_prob = jnp.where(
                jnp.isnan(model_log_prob) | jnp.isneginf(model_log_prob),
                -jnp.inf,
                model_log_prob,
            )
            log_prob_expanded = jnp.expand_dims(
                safe_model_log_prob
                + mvn_log_prob(mean_stack, scale_tril_stack, samples),
                axis=-1,
            )
            return jax.nn.softmax(
                log_prob_expanded,
                where=~jnp.isneginf(log_prob_expanded),
                axis=0,
            )

        if max_iter_mean > 0:
            rng_key, subkey = jrd.split(rng_key)
            moment_matching_mean = moment_match_mean(
                subkey,
                moment_matching_mean,
                moment_matching_scale_tril,
                normalized_weights_fn,
                max_iter_mean,
                100,
            )

        if max_iter_cov > 0:
            rng_key, subkey = jrd.split(rng_key)
            moment_matching_cov = moment_match_cov(
                subkey,
                moment_matching_mean,
                moment_matching_cov,
                normalized_weights_fn,
                max_iter_cov,
                100,
            )
            moment_matching_scale_tril = jnp.linalg.cholesky(moment_matching_cov)

        fit_mean = moment_matching_mean
        if n_vi_steps > 0:
            rng_key, subkey = jrd.split(rng_key)
            vi_mean = match_mean_by_variational_inference(
                subkey,
                mean_stack,
                moment_matching_mean,
                moment_matching_scale_tril,
                model_instance,
                learning_rate,
                n_vi_steps,
                100,
            )
            fit_mean = jnp.where(jnp.isnan(vi_mean), moment_matching_mean, vi_mean)

        fit_scale_tril = moment_matching_scale_tril

        mvn_log_prob_while_body = jax.vmap(mvn_log_prob, in_axes=(None, None, 0))

        def scan_fn(
            carry: Array, loop_data: Tuple[Array, Array, Array, Array, PRNGKeyArray]
        ) -> Tuple[Array, None]:
            mean, scale_tril, fit_mean_i, fit_scale_tril_i, rng_key_i = loop_data

            def while_body_fn(state: StateT) -> StateT:
                log_estimate_1, error_1, N_1, rng_key = state
                N_2 = n_samples
                rng_key, subkey = jrd.split(rng_key)

                # data ~ G(θ, z | μ_f, Σ_f)
                data = mvn_samples(
                    loc=fit_mean_i,
                    scale_tril=fit_scale_tril_i,
                    n_samples=N_2,
                    key=subkey,
                )

                # log ρ(data | Λ, κ)
                model_instance_log_prob = jax.vmap(model_instance.log_prob)(data)

                # log G(θ, z | μ_i, Σ_i)
                event_mvn_log_prob = mvn_log_prob_while_body(mean, scale_tril, data)

                # log G(θ, z | μ_f, Σ_i)
                fit_mvn_log_prob = mvn_log_prob_while_body(
                    fit_mean_i, fit_scale_tril_i, data
                )

                log_prob = (
                    model_instance_log_prob + event_mvn_log_prob - fit_mvn_log_prob
                )
                log_estimate_2, error_2 = monte_carlo_log_estimate_and_error(
                    log_prob,
                    N_2,  # type: ignore[arg-type]
                )
                log_estimate_3 = combine_monte_carlo_log_estimates(
                    log_estimate_1,
                    log_estimate_2,
                    N_1,
                    N_2,  # type: ignore[arg-type]
                )
                error_3 = combine_monte_carlo_errors(
                    error_1,
                    error_2,
                    log_estimate_1,
                    log_estimate_2,
                    log_estimate_3,
                    N_1,
                    N_2,  # type: ignore[arg-type]
                )
                return log_estimate_3, error_3, N_1 + N_2, rng_key

            state_0 = (
                -jnp.inf,  # starting log estimate is -inf
                jnp.zeros(()),  # starting error is zero
                n_samples,
                rng_key_i,
            )

            log_likelihood_i, _, _, _ = eqx.internal.while_loop(
                lambda state: jnp.less_equal(state[1], minimum_mc_error),
                while_body_fn,
                while_body_fn(state_0),  # this makes it a do-while loop
                kind="checkpointed",
                checkpoints=n_checkpoints,
                max_steps=n_max_steps - 1,  # already did one step
            )

            return carry + log_likelihood_i, None

        keys = jrd.split(rng_key, (n_events,))

        total_log_likelihood, _ = jax.lax.scan(
            scan_fn,  # type: ignore[arg-type]
            n_events * jnp.log(T_obs),
            (mean_stack, scale_tril_stack, fit_mean, fit_scale_tril, keys),
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

    return eqx.filter_jit(log_likelihood_fn)
