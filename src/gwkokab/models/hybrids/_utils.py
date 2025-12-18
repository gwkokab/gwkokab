# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, Final, List, Literal, Optional, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample

from ...parameters import Parameters as P
from ...utils.kernel import log_planck_taper_window
from ..constraints import all_constraint
from ._ncombination import (
    combine_distributions,
    create_beta_distributions,
    create_eccentric_mixture_models,
    create_minimum_tilt_model,
    create_powerlaw_redshift,
    create_spin_magnitude_mixture_models,
)


_M1_GRID_SIZE: Final[int] = 1000
_Q_GRID_SIZE: Final[int] = 500


def build_non_mass_distributions(
    N: int,
    component_type: Literal["pl", "bpl", "g"],
    mass_distributions: List[Distribution],
    use_beta_spin_magnitude: bool,
    use_spin_magnitude_mixture: bool,
    use_tilt: bool,
    use_eccentricity_mixture: bool,
    use_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    build_distributions = mass_distributions
    # fmt: off
    _info_collection: List[Tuple[bool, str, Callable[..., List[Distribution]]]] = [
        (use_beta_spin_magnitude, P.PRIMARY_SPIN_MAGNITUDE.value, create_beta_distributions),
        (use_beta_spin_magnitude, P.SECONDARY_SPIN_MAGNITUDE.value, create_beta_distributions),
        (use_spin_magnitude_mixture, P.PRIMARY_SPIN_MAGNITUDE.value + "_" + P.SECONDARY_SPIN_MAGNITUDE.value, create_spin_magnitude_mixture_models),
        # combined tilt distribution
        (use_tilt, P.COS_TILT_1.value + "_" + P.COS_TILT_2.value, create_minimum_tilt_model),
        (use_eccentricity_mixture, P.ECCENTRICITY.value, create_eccentric_mixture_models),
        (use_redshift, P.REDSHIFT.value, create_powerlaw_redshift),

    ]
    # fmt: on

    # Iterate over the list of tuples and build distributions
    for use, param_name, build_func in _info_collection:
        if use:
            distributions = build_func(
                N=N,
                parameter_name=param_name,
                component_type=component_type,
                params=params,
                validate_args=validate_args,
            )
            build_distributions = combine_distributions(
                build_distributions, distributions
            )

    return build_distributions


class _SmoothedPowerlawMassRatioAndRest(Distribution):
    arg_constraints = {
        "beta": constraints.real,
        "delta_m1": constraints.positive,
        "delta_m2": constraints.positive,
        "log_rate": constraints.real,
        "logZ": constraints.real,
        "m1max": constraints.positive,
        "m1min": constraints.positive,
        "m2min": constraints.positive,
    }
    pytree_data_fields = (
        "_m1s",
        "_support",
        "_Z_q_given_m1",
        "beta",
        "delta_m1",
        "delta_m2",
        "log_rate",
        "logZ",
        "m1max",
        "m1min",
        "m2min",
        "rest_dist",
    )

    def __init__(
        self,
        rest_dist: Distribution,
        beta: ArrayLike,
        delta_m1: ArrayLike,
        delta_m2: ArrayLike,
        log_rate: ArrayLike,
        logZ: ArrayLike,
        m1max: ArrayLike,
        m1min: ArrayLike,
        m2min: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ):
        self.rest_dist = rest_dist
        (
            self.beta,
            self.delta_m1,
            self.delta_m2,
            self.log_rate,
            self.logZ,
            self.m1max,
            self.m1min,
            self.m2min,
        ) = promote_shapes(
            beta,
            delta_m1,
            delta_m2,
            log_rate,
            logZ,
            m1max,
            m1min,
            m2min,
        )
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(beta),
            jnp.shape(delta_m1),
            jnp.shape(delta_m2),
            jnp.shape(log_rate),
            jnp.shape(logZ),
            jnp.shape(m1max),
            jnp.shape(m1min),
            jnp.shape(m2min),
            rest_dist.batch_shape,
        )
        n_dim_rest_dist = rest_dist.event_shape[0]
        self._support = all_constraint(
            [rest_dist._support, constraints.interval(m2min, m1max)],
            [(0, n_dim_rest_dist), n_dim_rest_dist],
        )
        super(_SmoothedPowerlawMassRatioAndRest, self).__init__(
            batch_shape=batch_shape,
            event_shape=(n_dim_rest_dist + 1,),
            validate_args=validate_args,
        )

        self._m1s = jnp.linspace(m1min, m1max, _M1_GRID_SIZE)
        qs = jnp.linspace(0.005, 1.0, _Q_GRID_SIZE)
        m1s_grid, qs_grid = jnp.meshgrid(self._m1s, qs, indexing="ij")

        prob_q_unnorm = jnp.exp(self._log_prob_q_unnorm(m1s_grid, qs_grid))
        self._Z_q_given_m1 = jnp.trapezoid(prob_q_unnorm, qs, axis=1)

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    def _log_prob_q_unnorm(self, m1: Array, q: Array) -> Array:
        m2 = m1 * q
        safe_delta = jnp.where(self.delta_m2 <= 0.0, 1.0, self.delta_m2)
        log_smoothing_m2 = log_planck_taper_window((m2 - self.m2min) / safe_delta)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_m2
        return jnp.where(
            (self.delta_m2 <= 0.0)
            | (m2 < self.m2min)
            | (m2 > m1)
            | (m1 > self.m1max)
            | jnp.isneginf(log_smoothing_m2),
            -jnp.inf,
            log_prob_q,
        )

    @validate_sample
    def log_prob(self, value: Array) -> ArrayLike:
        m1 = jax.lax.dynamic_index_in_dim(value, 0, axis=-1, keepdims=False)
        m2 = jax.lax.dynamic_index_in_dim(value, -1, axis=-1, keepdims=False)
        rest = jax.lax.dynamic_slice_in_dim(value, 0, value.shape[-1] - 1, axis=-1)

        safe_delta = jnp.where(self.delta_m1 <= 0.0, 1.0, self.delta_m1)
        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / safe_delta)
        rest_log_prob = self.rest_dist.log_prob(rest) + log_smoothing_m1 - self.logZ

        log_prob_q_unnorm = self._log_prob_q_unnorm(m1, m2 / m1)
        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1, left=0.0, right=0.0)
        safe_Z_q = jnp.where(_Z_q <= 0, 1.0, _Z_q)
        log_Z_q = jnp.where(_Z_q <= 0, 0.0, jnp.log(safe_Z_q))

        log_prob_val = (
            self.log_rate
            - jnp.log(m1)  # Jacobian
            + rest_log_prob
            + log_prob_q_unnorm
            - log_Z_q
        )

        return jnp.where(
            (self.delta_m1 <= 0.0) | jnp.isneginf(log_prob_val), -jnp.inf, log_prob_val
        )
