# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Optional, Tuple

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import (
    CategoricalProbs,
    constraints,
    Distribution,
    MixtureGeneral,
)

from ...utils.kernel import log_planck_taper_window
from ..constraints import any_constraint
from ..utils import JointDistribution
from ._ncombination import (
    create_broken_powerlaws,
    create_truncated_normal_distributions,
)
from ._utils import (
    _M1_GRID_SIZE,
    _SmoothedPowerlawMassRatioAndRest,
    build_non_mass_distributions,
)


def _build_bpl_component_distributions(
    N: int,
    use_beta_spin_magnitude: bool,
    use_truncated_normal_spin_magnitude: bool,
    use_tilt: bool,
    use_truncated_normal_eccentricity: bool,
    use_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> Tuple[List[Distribution], List[JointDistribution]]:
    mass_distributions = create_broken_powerlaws(
        N=N, params=params, validate_args=validate_args
    )

    build_distributions = build_non_mass_distributions(
        N=N,
        component_type="bpl",
        mass_distributions=[[d] for d in mass_distributions],
        params=params,
        use_beta_spin_magnitude=use_beta_spin_magnitude,
        use_truncated_normal_spin_magnitude=use_truncated_normal_spin_magnitude,
        use_tilt=use_tilt,
        use_truncated_normal_eccentricity=use_truncated_normal_eccentricity,
        use_redshift=use_redshift,
        validate_args=validate_args,
    )

    return mass_distributions, [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def _build_g_component_distributions(
    N: int,
    use_beta_spin_magnitude: bool,
    use_truncated_normal_spin_magnitude: bool,
    use_tilt: bool,
    use_truncated_normal_eccentricity: bool,
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

    build_distributions = build_non_mass_distributions(
        N=N,
        component_type="g",
        mass_distributions=[[d] for d in mass_distributions],
        use_beta_spin_magnitude=use_beta_spin_magnitude,
        use_truncated_normal_spin_magnitude=use_truncated_normal_spin_magnitude,
        use_tilt=use_tilt,
        use_truncated_normal_eccentricity=use_truncated_normal_eccentricity,
        use_redshift=use_redshift,
        params=params,
        validate_args=validate_args,
    )

    return mass_distributions, [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def NBrokenPowerlawMGaussian(
    N_bpl: int,
    N_g: int,
    use_beta_spin_magnitude: bool,
    use_truncated_normal_spin_magnitude: bool,
    use_tilt: bool,
    use_truncated_normal_eccentricity: bool,
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
    _lambdas.append(1.0 - sum(_lambdas))
    lambdas = jnp.stack(_lambdas, axis=-1)

    bpl_component_dist: List[JointDistribution] = []
    bpl_dists: List[JointDistribution] = []
    if N_bpl > 0:
        bpl_dists, bpl_component_dist = _build_bpl_component_distributions(
            N=N_bpl,
            use_beta_spin_magnitude=use_beta_spin_magnitude,
            use_truncated_normal_spin_magnitude=use_truncated_normal_spin_magnitude,
            use_tilt=use_tilt,
            use_truncated_normal_eccentricity=use_truncated_normal_eccentricity,
            use_redshift=use_redshift,
            params=params,
            validate_args=validate_args,
        )

    g_component_dist: List[JointDistribution] = []
    gaussian_dists: List[JointDistribution] = []
    if N_g > 0:
        gaussian_dists, g_component_dist = _build_g_component_distributions(
            N=N_g,
            use_beta_spin_magnitude=use_beta_spin_magnitude,
            use_truncated_normal_spin_magnitude=use_truncated_normal_spin_magnitude,
            use_tilt=use_tilt,
            use_truncated_normal_eccentricity=use_truncated_normal_eccentricity,
            use_redshift=use_redshift,
            params=params,
            validate_args=validate_args,
        )

    component_dists = bpl_component_dist + g_component_dist
    mass_dist = bpl_dists + gaussian_dists

    mixing_distribution = CategoricalProbs(probs=lambdas, validate_args=validate_args)
    mass_dist_mixture = MixtureGeneral(
        mixing_distribution,
        mass_dist,
        support=constraints.interval(m1min, m1max),
        validate_args=validate_args,
    )

    mm = jnp.linspace(m1min, m1max, _M1_GRID_SIZE)
    safe_delta_m1 = jnp.where(delta_m1 <= 0.0, 1.0, delta_m1)
    _log_prob_m1 = mass_dist_mixture.log_prob(mm) + log_planck_taper_window(
        (mm - m1min) / safe_delta_m1
    )
    _prob_m1 = jnp.where(delta_m1 <= 0.0, 0.0, jnp.exp(_log_prob_m1))
    Z = jnp.trapezoid(_prob_m1, mm, axis=0)
    logZ = jnp.log(Z)

    dist_m1_and_rest = MixtureGeneral(
        mixing_distribution,
        component_dists,
        support=any_constraint(
            [component_dist._support for component_dist in component_dists]
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
        log_rate=log_rate,
        logZ=logZ,
        validate_args=validate_args,
    )
