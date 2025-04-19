# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import warnings
from collections.abc import Callable
from typing import List, Tuple

import jax
from jax import Array, nn as jnn, numpy as jnp
from numpyro.distributions import Distribution

from ..models.utils import JointDistribution, ScaledMixture
from .bake import Bake


__all__ = ["poisson_likelihood"]


def poisson_likelihood(
    model: Bake,
    data: List[Array],
    log_ref_priors: List[Array],
    ERate_fn: Callable[[Distribution], Array],
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
    dummy_model = model.get_dummy()
    if not isinstance(dummy_model, ScaledMixture):
        warnings.warn(
            "The model provided is not a ScaledMixture. This means rate estimation "
            "will not be possible."
        )

    max_size = max([d.shape[0] for d in data])

    data_padded = [
        jnp.pad(d, ((0, max_size - d.shape[0]),) + ((0, 0),) * (d.ndim - 1))
        for d in data
    ]
    mask = [
        jnp.pad(jnp.ones(d.shape[0], dtype=jnp.bool), (0, max_size - d.shape[0]))
        for d in data
    ]
    log_ref_priors_padded = [
        jnp.pad(l, ((0, max_size - l.shape[0]),) + ((0, 0),) * (l.ndim - 1))
        for l in log_ref_priors
    ]

    stacked_data = jax.block_until_ready(
        jax.device_put(jnp.stack(data_padded, axis=0), may_alias=True)
    )
    stacked_log_ref_priors = jax.block_until_ready(
        jax.device_put(jnp.stack(log_ref_priors_padded, axis=0), may_alias=True)
    )
    stacked_mask = jax.block_until_ready(
        jax.device_put(jnp.stack(mask, axis=0), may_alias=True)
    )

    variables, duplicates, model = model.get_dist()  # type: ignore
    variables_index = {key: i for i, key in enumerate(variables.keys())}
    for key, value in duplicates.items():
        variables_index[key] = variables_index[value]

    priors = JointDistribution(*variables.values(), validate_args=True)

    def likelihood_fn(x: Array, _: Array) -> Array:
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: Distribution = model(**mapped_params)

        def single_event_fn(
            carry: Array, input: Tuple[Array, Array, Array]
        ) -> Tuple[Array, None]:
            data, log_ref_prior, mask = input

            safe_data = jnp.where(mask[:, None], data, jnp.ones_like(data))
            safe_log_ref_prior = jnp.where(mask, log_ref_prior, jnp.zeros_like(mask))

            log_prob = model_instance.log_prob(safe_data) - safe_log_ref_prior
            log_prob = jnp.where(mask, log_prob, jnp.full_like(mask, -jnp.inf))

            log_prob_sum = jnn.logsumexp(
                log_prob,
                axis=-1,
                where=~jnp.isneginf(log_prob),
            )
            return carry + log_prob_sum, None

        total_log_likelihood, _ = jax.lax.scan(
            single_event_fn,  # type: ignore
            jnp.zeros(()),
            (stacked_data, stacked_log_ref_priors, stacked_mask),
        )

        expected_rates = ERate_fn(model_instance)
        log_prior = priors.log_prob(x)
        log_likelihood = total_log_likelihood - expected_rates
        log_posterior = log_prior + log_likelihood

        log_posterior = jnp.nan_to_num(
            log_posterior, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
        )

        return log_posterior

    return variables_index, priors, likelihood_fn
