# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, List, Literal, Optional, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import (
    CategoricalProbs,
    constraints,
    Distribution,
    MixtureGeneral,
)
from numpyro.distributions.util import promote_shapes, validate_sample

from ...parameters import Parameters as P
from ...utils.kernel import log_planck_taper_window
from ..constraints import all_constraint, any_constraint
from ..utils import JointDistribution, ScaledMixture
from ._ncombination import (
    combine_distributions,
    create_broken_powerlaws,
    create_independent_spin_orientation_gaussian_isotropic,
    create_powerlaw_redshift,
    create_truncated_normal_distributions,
)


def _build_non_mass_distributions(
    N: int,
    component_type: Literal["bpl", "g"],
    mass_distributions: List[Distribution],
    use_spin_magnitude: bool,
    use_tilt: bool,
    use_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    build_distributions = mass_distributions
    # fmt: off
    _info_collection: List[Tuple[bool, str, Callable[..., List[Distribution]]]] = [
        (use_spin_magnitude, P.PRIMARY_SPIN_MAGNITUDE.value, create_truncated_normal_distributions),
        (use_spin_magnitude, P.SECONDARY_SPIN_MAGNITUDE.value, create_truncated_normal_distributions),
        # combined tilt distribution
        (use_tilt, P.COS_TILT_1.value + "_" + P.COS_TILT_2.value, create_independent_spin_orientation_gaussian_isotropic),
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


def _build_bpl_component_distributions(
    N: int,
    use_spin_magnitude: bool,
    use_tilt: bool,
    use_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> Tuple[List[Distribution], List[JointDistribution]]:
    mass_distributions = create_broken_powerlaws(
        N=N, params=params, validate_args=validate_args
    )

    build_distributions = _build_non_mass_distributions(
        N=N,
        component_type="bpl",
        mass_distributions=[[d] for d in mass_distributions],
        params=params,
        use_spin_magnitude=use_spin_magnitude,
        use_tilt=use_tilt,
        use_redshift=use_redshift,
        validate_args=validate_args,
    )

    return mass_distributions, [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def _build_g_component_distributions(
    N: int,
    use_spin_magnitude: bool,
    use_tilt: bool,
    use_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> Tuple[List[Distribution], List[JointDistribution]]:
    mass_distributions = create_truncated_normal_distributions(
        N=N,
        parameter_name="m1",
        component_type="g",
        params=params,
        validate_args=validate_args,
    )

    build_distributions = _build_non_mass_distributions(
        N=N,
        component_type="g",
        mass_distributions=[[d] for d in mass_distributions],
        use_spin_magnitude=use_spin_magnitude,
        use_tilt=use_tilt,
        use_redshift=use_redshift,
        params=params,
        validate_args=validate_args,
    )

    return mass_distributions, [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


class _SmoothedPowerlawMassRatioAndRest(Distribution):
    arg_constraints = {
        "beta": constraints.real,
        "delta_m1": constraints.positive,
        "delta_m2": constraints.positive,
        "m1min": constraints.positive,
        "m2min": constraints.positive,
    }
    pytree_data_fields = (
        "_log_Z_q",
        "_m1s",
        "beta",
        "delta_m1",
        "delta_m2",
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
            self.m1min,
            self.m2min,
        ) = promote_shapes(
            beta,
            delta_m1,
            delta_m2,
            m1min,
            m2min,
        )
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(beta),
            jnp.shape(delta_m1),
            jnp.shape(delta_m2),
            jnp.shape(m1min),
            jnp.shape(m2min),
            rest_dist.batch_shape,
        )
        self._support = all_constraint(
            [rest_dist.support, constraints.interval(m2min, m1min)],
            [(0, len(rest_dist.event_shape)), len(rest_dist.event_shape)],
        )
        super(_SmoothedPowerlawMassRatioAndRest, self).__init__(
            batch_shape=batch_shape,
            event_shape=(rest_dist.event_shape[0] + 1,),
            validate_args=validate_args,
        )

        self._m1s = jnp.linspace(m1min, m1max, 1000)
        m2s = jnp.linspace(m2min, m1min, 500)
        m1s_grid, m2s_grid = jnp.meshgrid(self._m1s, m2s, indexing="ij")

        log_prob_q_unnorm = self._log_prob_q_unnorm(m1s_grid, m2s_grid)
        prob_q = jnp.exp(log_prob_q_unnorm)
        _Z_q_given_m1 = jnp.trapezoid(prob_q, m2s, axis=1)
        safe_Z_q_given_m1 = jnp.where(_Z_q_given_m1 <= 0, 1.0, _Z_q_given_m1)
        self._log_Z_q = jnp.where(
            _Z_q_given_m1 <= 0, jnp.nan_to_num(-jnp.inf), jnp.log(safe_Z_q_given_m1)
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        return self._support

    def _log_prob_q_unnorm(self, m1: Array, m2: Array) -> Array:
        safe_delta = jnp.where(self.delta_m2 <= 0.0, 1.0, self.delta_m2)
        log_smoothing_q = log_planck_taper_window((m2 - self.m2min) / safe_delta)
        log_prob_q = self.beta * jnp.log(m2 / m1) + log_smoothing_q
        return jnp.where(
            (self.delta_m2 <= 0.0) | (m2 < self.m2min) | (m2 > m1) | (m1 < self.m1min),
            -jnp.inf,
            log_prob_q,
        )

    @validate_sample
    def log_prob(self, value: Array) -> ArrayLike:
        m1 = jax.lax.dynamic_index_in_dim(value, 0, axis=-1, keepdims=False)
        m2 = jax.lax.dynamic_index_in_dim(value, -1, axis=-1, keepdims=False)
        rest = jax.lax.dynamic_slice_in_dim(value, 0, value.shape[-1] - 1, axis=-1)

        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / self.delta_m1)
        rest_log_prob = self.rest_dist.log_prob(rest) + log_smoothing_m1

        log_prob_q_unnorm = self._log_prob_q_unnorm(m1, m2)

        return (
            -jnp.log(m1)  # Jacobian
            + rest_log_prob
            + log_prob_q_unnorm
            - jnp.interp(m1, self._m1s, self._log_Z_q, left=0.0, right=0.0)
        )


def NBrokenPowerlawMGaussian(
    N_bpl: int,
    N_g: int,
    use_spin_magnitude: bool,
    use_tilt: bool,
    use_redshift: bool,
    *,
    validate_args=None,
    **params,
):
    beta = params.pop("beta")
    delta_m1 = params.pop("delta_m1")
    delta_m2 = params.pop("delta_m2")
    log_rate = params.pop("log_rate")
    m1max = params.pop("m1max")
    m1min = params.pop("m1min")
    m2min = params.pop("m2min")

    _lambdas = [params.pop(f"lambda_{i}") for i in range(N_bpl + N_g - 1)]
    _lambdas.append(jnp.asarray(1.0) - sum(_lambdas, start=jnp.asarray(0.0)))
    lambdas = jnp.stack(_lambdas, axis=-1)

    if N_bpl > 0:
        broken_powerlaws, pl_component_dist = _build_bpl_component_distributions(
            N=N_bpl,
            use_spin_magnitude=use_spin_magnitude,
            use_tilt=use_tilt,
            use_redshift=use_redshift,
            params=params,
            validate_args=validate_args,
        )

    if N_g > 0:
        mass_gaussians, g_component_dist = _build_g_component_distributions(
            N=N_g,
            use_spin_magnitude=use_spin_magnitude,
            use_tilt=use_tilt,
            use_redshift=use_redshift,
            params=params,
            validate_args=validate_args,
        )

    if N_bpl == 0 and N_g != 0:
        component_dists = g_component_dist
        mass_dist = mass_gaussians
    elif N_g == 0 and N_bpl != 0:
        component_dists = pl_component_dist
        mass_dist = broken_powerlaws
    else:
        component_dists = pl_component_dist + g_component_dist
        mass_dist = broken_powerlaws + mass_gaussians

    mass_dist_mixture = MixtureGeneral(
        CategoricalProbs(probs=lambdas, validate_args=validate_args),
        mass_dist,
        validate_args=validate_args,
    )

    mm = jnp.linspace(m1min, m1max, 1000)

    Z = jnp.trapezoid(
        jnp.exp(
            mass_dist_mixture.log_prob(mm)
            + log_planck_taper_window((mm - m1min) / delta_m1)
        ),
        mm,
        axis=0,
    )
    logZ = jnp.log(Z)

    dist_m1_and_rest = ScaledMixture(
        log_rate + jnp.log(lambdas) - logZ,
        component_dists,
        support=any_constraint(
            [component_dists.support for component_dists in component_dists]
        ),
        validate_args=validate_args,
    )

    return _SmoothedPowerlawMassRatioAndRest(
        rest_dist=dist_m1_and_rest,
        beta=beta,
        delta_m1=delta_m1,
        delta_m2=delta_m2,
        m1max=m1max,
        m1min=m1min,
        m2min=m2min,
        validate_args=validate_args,
    )
