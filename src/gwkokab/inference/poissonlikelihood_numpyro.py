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

    def likelihood_fn(data: List[Array], log_ref_priors: List[Array]):
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

        # μ = E_{θ|Λ}[VT(θ)]
        numpyro.factor("expected_rates", ERate_obj(model_instance))

        def single_event_fn(data: Array, log_ref_prior: Array) -> Array:
            # log p(θ|data_n)
            model_log_prob = jax.jit(jax.vmap(jax.jit(model_instance.log_prob)))(data)

            # log p(θ|data_n) - log π_n
            log_prob: Array = model_log_prob - log_ref_prior
            log_prob = jnp.where(~jnp.isnan(log_prob), log_prob, -jnp.inf)

            # log Σ exp (log p(θ|data_n) - log π_n)
            log_prob_sum = jax.nn.logsumexp(
                log_prob,
                axis=-1,
                where=(~jnp.isneginf(log_prob)),
            )
            return jnp.clip(
                log_prob_sum,
                min=jnp.finfo(jnp.result_type(float)).min,
                max=jnp.finfo(jnp.result_type(float)).max,
            )

        numpyro.factor("log_constants", log_constants)  # - Σ log(M_i)

        # Σ log Σ exp (log p(θ|data_n) - log π_n)
        for i, (d, lrp) in enumerate(zip(data, log_ref_priors)):
            numpyro.factor(f"log_likelihood_{i}", single_event_fn(d, lrp))

    return likelihood_fn, variables_index  # type: ignore
