# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import Any, Optional, Tuple, Union

import jax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from gwkokab.utils.exceptions import LoggedTypeError, LoggedValueError

from ..models import PowerlawRedshift
from ..models.utils import JointDistribution, ScaledMixture
from ..utils.train import load_model


def poisson_mean_from_neural_pdet(
    key: PRNGKeyArray,
    parameters: Sequence[str],
    filename: str,
    batch_size: Optional[int] = None,
    num_samples: int = 1_000,
    time_scale: Union[int, float, Array] = 1.0,
) -> Tuple[
    Optional[Callable[[Array], Array]],
    Callable[..., Array],
    dict[str, Any],
]:
    if not parameters:
        raise LoggedValueError("parameters sequence cannot be empty")
    if not isinstance(parameters, Sequence):
        raise LoggedTypeError(f"parameters must be a Sequence, got {type(parameters)}")
    if not all(isinstance(p, str) for p in parameters):
        raise LoggedTypeError("all parameters must be strings")
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise LoggedTypeError(
                f"batch_size must be an integer, got {type(batch_size)}",
            )
        if batch_size < 1:
            raise LoggedValueError(
                f"batch_size must be a positive integer, got {batch_size}",
            )

    names, neural_vt_model = load_model(filename)
    if any(name not in parameters for name in names):
        raise LoggedValueError(
            f"Model in {filename} expects parameters {names}, but received "
            f"{parameters}. Missing: {set(names) - set(parameters)}",
        )

    shuffle_indices = [parameters.index(name) for name in names]

    @jax.jit
    def log_pdet(x: Array) -> Array:
        x_new = x[..., shuffle_indices]
        y_new = jnp.squeeze(
            jax.lax.map(neural_vt_model, x_new, batch_size=batch_size), axis=-1
        )
        mask = jnp.less_equal(y_new, 0.0)
        safe_y_new = jnp.where(mask, 1.0, y_new)
        return jnp.where(mask, -jnp.inf, jnp.log(safe_y_new))

    def _poisson_mean(
        scaled_mixture: ScaledMixture, T_obs: Array
    ) -> Tuple[Array, Array]:
        component_sample = scaled_mixture.component_sample(key, (num_samples,))
        # vmapping over components
        log_pdet_values = jax.vmap(log_pdet, in_axes=1)(component_sample)

        redshift_log_norm = []
        for component_dist in scaled_mixture.component_distributions:
            if isinstance(component_dist, JointDistribution):
                for m_dist in component_dist.marginal_distributions:
                    if isinstance(m_dist, PowerlawRedshift):
                        redshift_log_norm.append(m_dist.log_norm())
                        break
                else:
                    redshift_log_norm.append(jnp.zeros(()))
            else:
                redshift_log_norm.append(jnp.zeros(()))

        mean_per_component = jnp.exp(
            scaled_mixture.log_scales
            + jnp.stack(redshift_log_norm, axis=-1)
            + jax.nn.logsumexp(log_pdet_values, axis=-1)
        )
        mean = (T_obs / num_samples) * jnp.sum(mean_per_component, axis=-1)

        term2 = jnp.exp(
            2.0 * jnp.log(T_obs)
            - 3.0 * jnp.log(num_samples)
            + 2.0 * scaled_mixture.log_scales
            + 2.0 * jnp.stack(redshift_log_norm, axis=-1)
            + 2.0 * jax.nn.logsumexp(log_pdet_values, axis=-1)
        )
        term1 = jnp.exp(
            2.0 * jnp.log(T_obs)
            - 2.0 * jnp.log(num_samples)
            + 2.0 * scaled_mixture.log_scales
            + 2.0 * jnp.stack(redshift_log_norm, axis=-1)
            + jax.nn.logsumexp(2.0 * log_pdet_values, axis=-1)
        )
        variance_per_component = term1 - term2
        variance = jnp.sum(variance_per_component, axis=-1)
        return mean, variance

    return (
        log_pdet,
        _poisson_mean,
        {"T_obs": time_scale},
    )
