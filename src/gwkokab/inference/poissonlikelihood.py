# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import List, Optional, Tuple

import jax
import numpy as np
from jax import Array, numpy as jnp
from loguru import logger
from numpyro.distributions import Distribution

from ..models.utils import JointDistribution, ScaledMixture
from ..poisson_mean import PoissonMean
from ..utils.tools import warn_if
from .bake import Bake
from .jenks import pad_and_stack


__all__ = ["poisson_likelihood"]


def poisson_likelihood(
    dist_builder: Bake,
    data: List[np.ndarray],
    log_ref_priors: List[np.ndarray],
    ERate_obj: PoissonMean,
    where_fns: Optional[List[Callable[..., Array]]] = None,
    n_buckets: Optional[int] = None,
    threshold: float = 3.0,
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
    n_events = len(data)
    sum_log_size = sum([np.log(d.shape[0]) for d in data])
    log_constants = -sum_log_size  # -Σ log(M_i)
    log_constants += n_events * np.log(ERate_obj.time_scale)

    _data_group, _log_ref_priors_group, _masks_group = pad_and_stack(
        data, log_ref_priors, n_buckets=n_buckets, threshold=threshold
    )

    data_group: Sequence[Array] = jax.block_until_ready(
        jax.device_put(_data_group, may_alias=True)
    )
    log_ref_priors_group: Sequence[Array] = jax.block_until_ready(
        jax.device_put(_log_ref_priors_group, may_alias=True)
    )
    masks_group: Sequence[Array] = jax.block_until_ready(
        jax.device_put(_masks_group, may_alias=True)
    )

    logger.debug(
        "data_group.shape: {shape}",
        shape=", ".join([str(d.shape) for d in data_group]),
    )
    logger.debug(
        "log_ref_priors_group.shape: {shape}",
        shape=", ".join([str(d.shape) for d in log_ref_priors_group]),
    )
    logger.debug(
        "masks_group.shape: {shape}",
        shape=", ".join([str(d.shape) for d in masks_group]),
    )

    constants, variables, duplicates, dist_fn = dist_builder.get_dist()  # type: ignore
    variables_index: dict[str, int] = {
        key: i for i, key in enumerate(sorted(variables.keys()))
    }
    for key, value in duplicates.items():
        variables_index[key] = variables_index[value]

    group_variables: dict[int, list[str]] = {}
    for key, value in variables_index.items():  # type: ignore
        group_variables[value] = group_variables.get(value, []) + [key]  # type: ignore

    logger.debug(
        "Number of recovering variables: {num_vars}", num_vars=len(group_variables)
    )

    for key, value in constants.items():  # type: ignore
        logger.debug("Constant variable: {name} = {variable}", name=key, variable=value)

    for value in group_variables.values():  # type: ignore
        logger.debug("Recovering variable: {variable}", variable=", ".join(value))

    priors = JointDistribution(
        *[variables[key] for key in sorted(variables.keys())], validate_args=True
    )

    def likelihood_fn(x: Array, _: Array) -> Array:
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: Distribution = dist_fn(**mapped_params)

        # μ = E_{Ω|Λ}[VT(ω)]
        expected_rates = jax.block_until_ready(ERate_obj(model_instance))

        def single_event_fn(
            carry: Array, input: Tuple[Array, Array, Array]
        ) -> Tuple[Array, None]:
            safe_data, safe_log_ref_prior, mask = input

            # log p(ω|data_n)
            model_log_prob = jax.checkpoint(
                jax.vmap(model_instance.log_prob),
                prevent_cse=False,
            )(safe_data)
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
            with jax.ensure_compile_time_eval():
                safe_batched_data = jnp.where(
                    jnp.expand_dims(batched_mask, axis=-1),
                    batched_data,
                    model_instance.support.feasible_like(batched_data),
                )
                safe_log_ref_priors_group = jnp.where(
                    batched_mask, batched_log_ref_priors, 0.0
                )
            total_log_likelihood, _ = jax.lax.scan(
                single_event_fn,  # type: ignore
                total_log_likelihood,
                (safe_batched_data, safe_log_ref_priors_group, batched_mask),
            )

        total_log_likelihood = jax.block_until_ready(total_log_likelihood)

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

    def likelihood_fn_with_checks(x: Array, _: Array) -> Array:
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }
        predicate = priors.support.check(x)
        for where_fn in where_fns:  # type: ignore
            predicate = jnp.logical_and(
                predicate, where_fn(**constants, **mapped_params)
            )
        predicate = jnp.logical_and(jnp.all(x), predicate)
        return jnp.where(predicate, likelihood_fn(x, _), -jnp.inf)

    if where_fns is None:
        return variables_index, priors, likelihood_fn
    return variables_index, priors, likelihood_fn_with_checks
