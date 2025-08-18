# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Dict, List, Tuple

import jax
import numpyro
from jax import Array, numpy as jnp
from jaxtyping import ArrayLike
from loguru import logger
from numpyro.distributions import Distribution

from ..models.utils import ScaledMixture
from ..poisson_mean import PoissonMean
from ..utils.tools import warn_if
from .bake import Bake


__all__ = ["numpyro_poisson_likelihood"]


def numpyro_poisson_likelihood(
    dist_builder: Bake, log_constants: ArrayLike, ERate_obj: PoissonMean
) -> Tuple[Callable[..., Array], Dict[str, int]]:
    dummy_model = dist_builder.get_dummy()
    warn_if(
        not isinstance(dummy_model, ScaledMixture),
        msg="The model provided is not a ScaledMixture. "
        "Rate estimation will therefore be skipped.",
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

    def likelihood_fn(
        data_group: List[Array],
        log_ref_priors_group: List[Array],
        masks_group: List[Array],
    ):
        variables_samples = [
            numpyro.sample(parameter_name, prior_dist)
            for parameter_name, prior_dist in sorted(
                variables.items(), key=lambda x: x[0]
            )
        ]
        mapped_params = {
            name: variables_samples[i] for name, i in variables_index.items()
        }

        model_instance: Distribution = dist_fn(**mapped_params)

        # μ = E_{Ω|Λ}[VT(ω)]
        numpyro.factor("expected_rates", ERate_obj(model_instance))

        def single_event_fn(
            carry: Array, input: Tuple[Array, Array, Array]
        ) -> Tuple[Array, None]:
            safe_data, safe_log_ref_prior, mask = input

            # log p(ω|data_n)
            model_log_prob = jax.lax.map(
                model_instance.log_prob, safe_data, batch_size=1000
            )
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
            safe_batched_data = jnp.where(
                jnp.expand_dims(batched_mask, axis=-1),
                batched_data,
                model_instance.support.feasible_like(batched_data),
            )
            safe_batched_log_ref_priors = jnp.where(
                batched_mask, batched_log_ref_priors, 0.0
            )
            total_log_likelihood, _ = jax.lax.scan(
                single_event_fn,  # type: ignore
                total_log_likelihood,
                (safe_batched_data, safe_batched_log_ref_priors, batched_mask),
            )

        # Σ log Σ exp (log p(ω|data_n) - log π_n) - Σ log(M_i)
        numpyro.factor(
            "total_log_likelihood", jnp.nan_to_num(total_log_likelihood, nan=-jnp.inf)
        )

    return likelihood_fn, variables_index  # type: ignore
