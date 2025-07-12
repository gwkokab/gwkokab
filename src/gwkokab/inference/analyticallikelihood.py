# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, List, Optional

import jax
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import Distribution

from gwkokab.models.utils import JointDistribution


def analytical_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables_index: Dict[str, int],
    ERate_fn: Callable[[Distribution, Optional[int]], Array],
    redshift_index: Optional[int],
    means: List[Array],
    covariances: List[Array],
    key: PRNGKeyArray,
    N_samples: int = 10_000,
) -> Callable[[Array, Array], Array]:
    scale_trils = [
        jnp.linalg.cholesky(covariance_matrix) for covariance_matrix in covariances
    ]
    n_dim = means[0].shape[0]

    def likelihood_fn(x: Array, _: Array) -> Array:
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: Distribution = dist_fn(**mapped_params)

        rng_key = key

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = jax.block_until_ready(ERate_fn(model_instance, redshift_index))

        # - Σ log(M_i)
        total_log_likelihood = -len(means) * jnp.log(N_samples)
        for mean, scale_tril in zip(means, scale_trils):
            # See implementation of `numpyro.distributions.MultivariateNormal.sample`
            # for method to generate samples from a multivariate normal distribution.
            # ε ~ N(0, I)
            eps = jrd.normal(rng_key, shape=(N_samples, n_dim))
            # L_i = cholesky(Σ_i)
            # data_n = μ_i + L_i * ε
            data = mean + jnp.squeeze(
                jnp.matmul(scale_tril, eps[..., jnp.newaxis]), axis=-1
            )
            # log ρ(data_n|Λ,κ)
            log_prob = model_instance.log_prob(data)
            # log Σ exp log p(data_n|Λ,κ)
            total_log_likelihood += jax.nn.logsumexp(
                log_prob, axis=-1, where=(~jnp.isneginf(log_prob))
            )
            (rng_key,) = jrd.split(rng_key, num=1)

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
