# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, Tuple, TypeAlias

import equinox as eqx
import jax
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro._typing import DistributionT
from numpyro.distributions import Distribution, MultivariateNormal

from gwkokab.models.utils import JointDistribution

from .utils import (
    combine_monte_carlo_errors,
    combine_monte_carlo_log_estimates,
    match_mean_by_variational_inference,
    moment_match_cov,
    moment_match_mean,
    monte_carlo_log_estimate_and_error,
)


StateT: TypeAlias = Tuple[
    Array,  # old monte-carlo-estimate
    Array,  # old error
    Array,  # old size
    PRNGKeyArray,  # old key
]
"""State of the Monte Carlo estimation process."""


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables_index: Dict[str, int],
    ERate_fn: Callable[[Distribution], Array],
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
    ERate_fn : Callable[[Distribution], Array]
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

    def likelihood_fn(x: Array, data: Dict[str, Array]) -> Array:
        mean_stack = data["mean_stack"]
        cov_stack = data["cov_stack"]

        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: DistributionT = dist_fn(**mapped_params)

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = ERate_fn(model_instance)

        event_mvn = MultivariateNormal(loc=mean_stack, covariance_matrix=cov_stack)

        rng_key = key
        moment_matching_mean = mean_stack
        moment_matching_cov = cov_stack

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
        if max_iter_mean > 0:
            rng_key, subkey = jrd.split(rng_key)
            moment_matching_mean = moment_match_mean(
                subkey,
                moment_matching_mean,
                moment_matching_cov,
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

        if n_vi_steps > 0:
            rng_key, subkey = jrd.split(rng_key)
            vi_mean = match_mean_by_variational_inference(
                subkey,
                moment_matching_mean,
                moment_matching_cov,
                model_instance,
                learning_rate,
                n_vi_steps,
                100,
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

            @jax.jit
            def while_body_fn(state: StateT) -> StateT:
                log_estimate_1, error_1, N_1, rng_key = state
                N_2 = n_samples
                rng_key, subkey = jrd.split(rng_key)

                # data ~ G(θ, z | μ_f, Σ_f)
                data = fit_mvn.sample(subkey, (N_2,))

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

                log_prob = (
                    model_instance_log_prob + event_mvn_log_prob - fit_mvn_log_prob
                )
                log_estimate_2, error_2 = monte_carlo_log_estimate_and_error(
                    log_prob, N_2
                )
                log_estimate_3 = combine_monte_carlo_log_estimates(
                    log_estimate_1, log_estimate_2, N_1, N_2
                )
                error_3 = combine_monte_carlo_errors(
                    error_1,
                    error_2,
                    log_estimate_1,
                    log_estimate_2,
                    log_estimate_3,
                    N_1,
                    N_2,
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
                max_steps=n_max_steps,
            )

            return carry + log_likelihood_i, None

        keys = jrd.split(rng_key, (n_events,))

        total_log_likelihood, _ = jax.lax.scan(
            scan_fn,  # type: ignore[arg-type]
            jnp.zeros(()),
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
