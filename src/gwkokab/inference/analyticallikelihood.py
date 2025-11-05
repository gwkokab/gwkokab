# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, Tuple, TypeAlias

import equinox as eqx
import jax
from jax import nn as jnn, numpy as jnp, random as jrd
from jax.scipy.linalg import cho_solve
from jaxtyping import Array, PRNGKeyArray
from numpyro._typing import DistributionT
from numpyro.distributions import Distribution
from numpyro.distributions.continuous import _batch_mahalanobis, tri_logabsdet
from numpyro.distributions.util import cholesky_of_inverse

from gwkokab.models.utils import JointDistribution, ScaledMixture


StateT: TypeAlias = Tuple[
    Array,  # old monte-carlo-estimate
    Array,  # old error square
    Array,  # old size
    PRNGKeyArray,  # old key
]
"""State of the Monte Carlo estimation process:
   (log_estimate, error_sq, sample_count, rng_key)."""


@jax.jit
def monte_carlo_log_estimate_and_error_sq(
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
        Monte Carlo logarithm of estimate, and square of the Monte Carlo error.
    """
    mask = ~jnp.isneginf(log_probs)
    log_moment_1 = jnn.logsumexp(log_probs, where=mask, axis=-1) - jnp.log(N)
    moment_2 = jnp.exp(jnn.logsumexp(2.0 * log_probs, where=mask, axis=-1)) / N
    error_sq = (moment_2 - jnp.exp(2.0 * log_moment_1)) / (N - 1.0)
    return log_moment_1, error_sq


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
        First Monte Carlo estimate :math:`\ln\hat{\mu}_1`.
    estimates_2 : Array
        Second Monte Carlo estimate :math:`\ln\hat{\mu}_2`.
    N_1 : Array
        Number of samples used for the first estimate :math:`N_1`.
    N_2 : Array
        Number of samples used for the second estimate :math:`N_2`.

    Returns
    -------
    Array
        Combined Monte Carlo estimate :math:`\ln\hat{\mu}`.
    """
    combined_log_estimate = jnp.logaddexp(
        jnp.log(N_1) + log_estimates_1, jnp.log(N_2) + log_estimates_2
    ) - jnp.log(N_1 + N_2)
    return combined_log_estimate


@jax.jit
def combine_monte_carlo_errors_sq(
    error_1_sq: Array,
    error_2_sq: Array,
    log_estimate_1: Array,
    log_estimate_2: Array,
    log_estimate_3: Array,
    N_1: Array,
    N_2: Array,
) -> Array:
    r"""Combine two Monte Carlo errors into a single error estimate using the formula:

    .. math::

        \hat{\epsilon}^2=\frac{1}{N_3(N_3-1)}\sum_{k=1}^{2}\left\{N_k(N_k-1)\hat{\epsilon}_k^2+N_k\hat{\mu}^2_k\right\}-\frac{1}{N_3-1}\hat{\mu}^2

    where, :math:`N_3 = N_1 + N_2` is the total number of samples.

    Parameters
    ----------
    error_1_sq : Array
        Square of error of the first Monte Carlo estimate :math:`\hat{\epsilon}_1^2`.
    error_2_sq : Array
        Square of error of the second Monte Carlo estimate :math:`\hat{\epsilon}_2^2`.
    log_estimate_1 : Array
        Estimate of the first Monte Carlo estimate :math:`\ln\hat{\mu}_1`.
    log_estimate_2 : Array
        Estimate of the second Monte Carlo estimate :math:`\ln\hat{\mu}_2`.
    log_estimate_3 : Array
        Estimate of the combined Monte Carlo estimate :math:`\ln\hat{\mu}`.
    N_1 : Array
        Number of samples used for the first estimate :math:`N_1`.
    N_2 : Array
        Number of samples used for the second estimate :math:`N_2`.

    Returns
    -------
    Array
        Combined Monte Carlo squared error estimate :math:`\hat{\epsilon}^2`.
    """
    N_3 = N_1 + N_2

    sum_prob_sq_1 = N_1 * ((N_1 - 1.0) * error_1_sq + jnp.exp(2.0 * log_estimate_1))
    sum_prob_sq_2 = N_2 * ((N_2 - 1.0) * error_2_sq + jnp.exp(2.0 * log_estimate_2))

    combined_error_sq = -jnp.exp(2.0 * log_estimate_3) / (N_3 - 1.0)
    combined_error_sq += (sum_prob_sq_1 + sum_prob_sq_2) / N_3 / (N_3 - 1.0)

    return combined_error_sq


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
def mvn_log_prob_scaled(loc: Array, scale_tril: Array, value: Array) -> Array:
    # removing the constant term -0.5 * D * log(2pi), where D is the dimension,
    # because it is being cancelled out in the importance sampling weights
    M = _batch_mahalanobis(scale_tril, value - loc)
    return -0.5 * M - tri_logabsdet(scale_tril)


@jax.jit
def covariance_matrix(scale_tril: Array) -> Array:
    return jnp.matmul(scale_tril, jnp.swapaxes(scale_tril, -1, -2))


@jax.jit
def precision_matrix(scale_tril: Array) -> Array:
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
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    key: PRNGKeyArray,
    n_events: int,
    minimum_mc_error: float = 1e-2,
    n_samples: int = 100,
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
    n_vi_steps: int, optional
        Number of steps for the variational inference optimization, by default 5
    learning_rate : float, optional
        Learning rate for the Adam optimizer used in the variational inference
        optimization, by default 1e-2
    n_checkpoints : int, optional
        Checkpoint interval for the checkpointed while-loop, by default 10. For more
        details, see
        [`equinox.internal.while_loop`](https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/loop.py)
        and its [`"checkpointed"` variant](https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/checkpointed.py).
    n_max_steps : int, optional
        Maximum number of steps for the optimization process, by default 20

    Returns
    -------
    Callable[[Array, Array], Array]
        A function that computes the log posterior probability of the model parameters
        given the data. The function takes two arguments: an array of model parameters
        and a second array (not used in this implementation).
    """
    minimum_mc_error_sq = minimum_mc_error**2

    def log_likelihood_fn(x: Array, data: Dict[str, Array]) -> Array:
        mean_stack = data["mean_stack"]
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

        hessian_log_prob = jax.jit(
            jax.vmap(jax.hessian(model_instance.log_prob), axis_size=n_events)
        )
        grad_log_prob = jax.jit(
            jax.vmap(jax.grad(model_instance.log_prob), axis_size=n_events)
        )

        fit_precision_matrix = precision_matrix(scale_tril_stack) - hessian_log_prob(
            mean_stack
        )
        fit_covariance_matrix = jnp.linalg.inv(fit_precision_matrix)

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
        fit_scale_tril = jnp.where(
            is_psd[..., jnp.newaxis, jnp.newaxis],
            cholesky_of_inverse(fit_precision_matrix),
            scale_tril_stack,
        )

        mvn_log_prob_while_body = jax.vmap(
            mvn_log_prob_scaled, in_axes=(None, None, 0), axis_size=n_samples
        )

        def scan_fn(
            carry: Array, loop_data: Tuple[Array, Array, Array, Array, PRNGKeyArray]
        ) -> Tuple[Array, None]:
            mean, scale_tril, fit_mean_i, fit_scale_tril_i, rng_key_i = loop_data

            def while_body_fn(state: StateT) -> StateT:
                log_estimate_1, error_1_sq, N_1, rng_key = state
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
                model_instance_log_prob = jax.vmap(
                    model_instance.log_prob, axis_size=N_2
                )(data)

                # log G(θ, z | μ_i, Σ_i)
                event_mvn_log_prob = mvn_log_prob_while_body(mean, scale_tril, data)

                # log G(θ, z | μ_f, Σ_i)
                fit_mvn_log_prob = mvn_log_prob_while_body(
                    fit_mean_i, fit_scale_tril_i, data
                )

                log_prob = (
                    model_instance_log_prob + event_mvn_log_prob - fit_mvn_log_prob
                )
                log_estimate_2, error_2_sq = monte_carlo_log_estimate_and_error_sq(
                    log_prob,
                    N_2,  # type: ignore[arg-type]
                )
                log_estimate_3 = combine_monte_carlo_log_estimates(
                    log_estimate_1,
                    log_estimate_2,
                    N_1,
                    N_2,  # type: ignore[arg-type]
                )
                error_3_sq = combine_monte_carlo_errors_sq(
                    error_1_sq,
                    error_2_sq,
                    log_estimate_1,
                    log_estimate_2,
                    log_estimate_3,
                    N_1,
                    N_2,  # type: ignore[arg-type]
                )
                return log_estimate_3, error_3_sq, N_1 + N_2, rng_key

            state_0 = (
                -jnp.inf,  # starting log estimate is -inf
                jnp.zeros(()),  # starting error is zero
                0.0,  # starting size is zero
                rng_key_i,
            )

            log_likelihood_i, _, _, _ = eqx.internal.while_loop(
                lambda state: jnp.less_equal(state[1], minimum_mc_error_sq),
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
