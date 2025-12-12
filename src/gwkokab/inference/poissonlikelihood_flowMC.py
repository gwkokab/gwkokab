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

    log_constants = jnp.asarray(log_constants)

    def _map_params(x: Array) -> Dict[str, Array]:
        return {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

    def likelihood_fn(x: Array, data: Dict[str, Tuple[Array, ...]]) -> Array:
        data_group: Tuple[Array, ...] = data["data_group"]
        log_ref_priors_group: Tuple[Array, ...] = data["log_ref_priors_group"]
        masks_group: Tuple[Array, ...] = data["masks_group"]
        mapped_params = _map_params(x)

        model_instance = dist_fn(**constants, **mapped_params)

        total_log_likelihood = log_constants  # - Σ log(M_i)

        # Σ log Σ exp (log p(ω|data_n) - log π_n)
        for batched_data, batched_log_ref_priors, batched_mask in zip(
            data_group, log_ref_priors_group, masks_group
        ):
            feasible_point = model_instance.support.feasible_like(batched_data[0])

            safe_data = jnp.where(
                batched_mask[..., jnp.newaxis],
                batched_data,
                feasible_point,
            )

            # log p(ω|data_n)
            batch_model_log_prob: Array = model_instance.log_prob(safe_data)

            # log p(ω|data_n) - log π_n
            log_prob = batch_model_log_prob - batched_log_ref_priors
            log_prob = jnp.where(batched_mask, log_prob, -jnp.inf)

            # log Σ exp (log p(ω|data_n) - log π_n)
            total_log_likelihood += jax.nn.logsumexp(log_prob, axis=-1).sum(
                axis=0, initial=0.0
            )

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = poisson_mean_estimator(model_instance)
        # log L(ω) = -μ + Σ log Σ exp (log p(ω|data_n) - log π_n) - Σ log(M_i)
        log_likelihood = total_log_likelihood - expected_rates
        # log p(ω|data) = log π(ω) + log L(ω)
        log_prior = priors.log_prob(x)
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
        mapped_params = _map_params(x)
        predicate = priors.support.check(x)
        for where_fn in where_fns:  # type: ignore
            predicate = jnp.logical_and(
                predicate, where_fn(**constants, **mapped_params)
            )
        predicate = jnp.logical_and(jnp.all(jnp.isfinite(x)), predicate)
        return jnp.where(predicate, likelihood_fn(x, data), -jnp.inf)

    return eqx.filter_jit(likelihood_fn_with_checks)
