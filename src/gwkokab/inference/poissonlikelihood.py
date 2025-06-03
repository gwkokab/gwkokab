# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import List, Optional, Tuple

import jax
import numpy as np
from jax import Array, numpy as jnp
from loguru import logger
from numpyro.distributions import Distribution

from ..models.utils import JointDistribution, ScaledMixture
from ..utils.tools import warn_if
from .bake import Bake


__all__ = ["poisson_likelihood"]


def poisson_likelihood(
    dist_builder: Bake,
    data: List[np.ndarray],
    log_ref_priors: List[np.ndarray],
    ERate_fn: Callable[[Distribution], Array],
    where_fns: Optional[List[Callable[..., Array]]] = None,
) -> Tuple[dict[str, int], JointDistribution, Callable[[Array, Array], Array]]:
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
    dummy_model = dist_builder.get_dummy()
    warn_if(
        not isinstance(dummy_model, ScaledMixture),
        msg="The model provided is not a ScaledMixture. "
        "Rate estimation will therefore be skipped.",
    )

    # maximum size of the data
    max_size = max([d.shape[0] for d in data])
    sum_log_size = sum([jnp.log(d.shape[0]) for d in data])
    log_constants = -sum_log_size  # -Σ log(M_i)

    # pad the data and log_ref_priors to the maximum size and create a mask for the data
    # to indicate which elements are valid and which are padded.
    data_padded = [
        jnp.pad(d, ((0, max_size - d.shape[0]),) + ((0, 0),) * (d.ndim - 1))
        for d in data
    ]
    mask = [
        jnp.pad(
            jnp.ones(d.shape[0], dtype=bool),
            (0, max_size - d.shape[0]),
            constant_values=jnp.zeros((), dtype=bool),
        )
        for d in data
    ]
    log_ref_priors_padded = [
        jnp.pad(
            l,
            ((0, max_size - l.shape[0]),) + ((0, 0),) * (l.ndim - 1),
            constant_values=0.0,
        )
        for l in log_ref_priors
    ]

    batched_data: Array = jax.block_until_ready(
        jax.device_put(jnp.stack(data_padded, axis=0), may_alias=True)
    )
    batched_log_ref_priors: Array = jax.block_until_ready(
        jax.device_put(jnp.stack(log_ref_priors_padded, axis=0), may_alias=True)
    )
    batched_mask: Array = jax.block_until_ready(
        jax.device_put(jnp.stack(mask, axis=0), may_alias=True)
    )

    logger.debug(
        "batched_data.shape: {batched_data_shape}",
        batched_data_shape=batched_data.shape,
    )
    logger.debug(
        "batched_log_ref_priors.shape: {batched_log_ref_priors_shape}",
        batched_log_ref_priors_shape=batched_log_ref_priors.shape,
    )
    logger.debug(
        "batched_mask.shape: {batched_mask_shape}",
        batched_mask_shape=batched_mask.shape,
    )

    constants, variables, duplicates, dist_builder = dist_builder.get_dist()  # type: ignore
    variables_index = {key: i for i, key in enumerate(variables.keys())}
    for key, value in duplicates.items():
        variables_index[key] = variables_index[value]

    logger.debug(
        "Recovering variables: {variables}", variables=list(variables_index.keys())
    )

    priors = JointDistribution(*variables.values(), validate_args=True)

    def likelihood_fn(x: Array, _: Array) -> Array:
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: Distribution = dist_builder(**mapped_params)

        def single_event_fn(
            carry: Array, input: Tuple[Array, Array, Array]
        ) -> Tuple[Array, None]:
            data, log_ref_prior, mask = input

            safe_data = jnp.where(mask[:, jnp.newaxis], data, 1.0)
            safe_log_ref_prior = jnp.where(mask, log_ref_prior, 0.0)

            # log p(ω|data_n) - log π_n
            log_prob: Array = model_instance.log_prob(safe_data) - safe_log_ref_prior
            log_prob = jnp.where(mask, log_prob, -jnp.inf)

            # log Σ exp (log p(ω|data_n) - log π_n)
            log_prob_sum = jax.nn.logsumexp(
                log_prob,
                axis=-1,
                where=(~jnp.isneginf(log_prob)) & mask,
            )
            return carry + log_prob_sum, None

        x_mask = jnp.ones((), dtype=bool)
        if where_fns is not None:
            for where_fn in where_fns:
                x_mask = jnp.logical_and(x_mask, where_fn(**constants, **mapped_params))

        # Σ log Σ exp (log p(ω|data_n) - log π_n)
        total_log_likelihood, _ = jax.lax.scan(
            single_event_fn,  # type: ignore
            jnp.zeros(()),
            (batched_data, batched_log_ref_priors, batched_mask),
        )
        total_log_likelihood = jnp.where(x_mask, total_log_likelihood, -jnp.inf)

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = jnp.where(x_mask, ERate_fn(model_instance), -jnp.inf)
        log_prior = jnp.where(x_mask, priors.log_prob(x), -jnp.inf)
        # log L(ω) = -μ + Σ log Σ exp (log p(ω|data_n) - log π_n) - Σ log(M_i)
        log_likelihood = total_log_likelihood - expected_rates + log_constants
        # log p(ω|data) = log π(ω) + log L(ω)
        log_posterior = log_prior + log_likelihood

        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        return log_posterior

    return variables_index, priors, likelihood_fn
