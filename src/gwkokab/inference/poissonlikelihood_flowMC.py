# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

import equinox as eqx
import jax
from jax import Array, numpy as jnp
from jaxtyping import ArrayLike
from numpyro._typing import DistributionLike

from ..models.utils import JointDistribution, ScaledMixture


__all__ = ["flowMC_poisson_likelihood"]


def flowMC_poisson_likelihood(
    dist_fn: Callable[..., DistributionLike],
    priors: JointDistribution,
    variables: Dict[str, DistributionLike],
    variables_index: Dict[str, int],
    log_constants: ArrayLike,
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    where_fns: Optional[List[Callable[..., Array]]],
    constants: Dict[str, Array],
) -> Callable[[Array, Dict[str, Any]], Array]:
    r"""This class is used to provide a likelihood function for the inhomogeneous Poisson
    process. The likelihood is given by,

    .. math::
        \log\mathcal{L}(\Lambda) \propto -\mu(\Lambda)
        +\log\sum_{n=1}^N \int \ell_n(\lambda) \rho(\lambda\mid\Lambda)
        \mathrm{d}\lambda


    where, :math:`\displaystyle\rho(\lambda\mid\Lambda) =
    \frac{\mathrm{d}N}{\mathrm{d}V\mathrm{d}t \mathrm{d}\lambda}` is the merger
    rate density for a population parameterized by :math:`\Lambda`, :math:`\mu(\Lambda)` is
    the expected number of detected mergers for that population, and
    :math:`\ell_n(\lambda)` is the likelihood for the :math:`n`-th observed event's
    parameters. Using Bayes' theorem, we can obtain the posterior
    :math:`p(\Lambda\mid\text{data})` by multiplying the likelihood by a prior
    :math:`\pi(\Lambda)`.

    .. math::
        p(\Lambda\mid\text{data}) \propto \pi(\Lambda) \mathcal{L}(\Lambda)

    The integral inside the main likelihood expression is then evaluated via
    Monte Carlo as

    .. math::
        \int \ell_n(\lambda) \rho(\lambda\mid\Lambda) \mathrm{d}\lambda \propto
        \int \frac{p(\lambda | \mathrm{data}_n)}{\pi_n(\lambda)}
        \rho(\lambda\mid\Lambda) \mathrm{d}\lambda \approx
        \frac{1}{N_{\mathrm{samples}}}
        \sum_{i=1}^{N_{\mathrm{samples}}}
        \frac{\rho(\lambda_{n,i}\mid\Lambda)}{\pi_{n,i}}
    """
    del variables

    def likelihood_fn(x: Array, data: Dict[str, Tuple[Array, ...]]) -> Array:
        data_group: Tuple[Array, ...] = data["data_group"]
        log_ref_priors_group: Tuple[Array, ...] = data["log_ref_priors_group"]
        masks_group: Tuple[Array, ...] = data["masks_group"]
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance = dist_fn(**mapped_params)

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = poisson_mean_estimator(model_instance)

        log_prob_fn = eqx.filter_jit(eqx.filter_vmap(model_instance.log_prob))

        def single_event_fn(
            carry: Array, input: Tuple[Array, Array, Array]
        ) -> Tuple[Array, None]:
            data, log_ref_prior, mask = input

            safe_data = jnp.where(
                jnp.expand_dims(mask, axis=-1),
                data,
                model_instance.support.feasible_like(data),
            )
            safe_log_ref_prior = jnp.where(mask, log_ref_prior, 0.0)

            # log p(ω|data_n)
            model_log_prob = log_prob_fn(safe_data)
            safe_model_log_prob = jnp.where(mask, model_log_prob, -jnp.inf)

            # log p(ω|data_n) - log π_n
            log_prob: Array = safe_model_log_prob - safe_log_ref_prior
            log_prob = jnp.where(mask & (~jnp.isnan(log_prob)), log_prob, -jnp.inf)

            # log Σ exp (log p(ω|data_n) - log π_n)
            log_prob_sum = jax.nn.logsumexp(
                log_prob,
                axis=-1,
                where=(~jnp.isneginf(log_prob)) & mask,
            )
            return carry + log_prob_sum, None

        total_log_likelihood = log_constants  # - Σ log(M_i)
        # Σ log Σ exp (log p(ω|data_n) - log π_n)
        for batched_data, batched_log_ref_priors, batched_mask in zip(
            data_group, log_ref_priors_group, masks_group
        ):
            total_log_likelihood, _ = jax.lax.scan(
                single_event_fn,  # type: ignore
                total_log_likelihood,
                (batched_data, batched_log_ref_priors, batched_mask),
            )

        log_prior = priors.log_prob(x)
        # log L(ω) = -μ + Σ log Σ exp (log p(ω|data_n) - log π_n) - Σ log(M_i)
        log_likelihood = total_log_likelihood - expected_rates
        # log p(ω|data) = log π(ω) + log L(ω)
        log_posterior = log_prior + log_likelihood

        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        return log_posterior

    if where_fns is None:
        return eqx.filter_jit(likelihood_fn)

    def likelihood_fn_with_checks(
        x: Array, data: Dict[str, Tuple[Array, ...]]
    ) -> Array:
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }
        predicate = priors.support.check(x)
        for where_fn in where_fns:  # type: ignore
            predicate = jnp.logical_and(
                predicate, where_fn(**constants, **mapped_params)
            )
        predicate = jnp.logical_and(jnp.all(jnp.isfinite(x)), predicate)
        return jnp.where(predicate, likelihood_fn(x, data), -jnp.inf)

    return eqx.filter_jit(likelihood_fn_with_checks)
