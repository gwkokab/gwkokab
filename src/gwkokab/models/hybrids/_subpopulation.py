# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import (
    CategoricalProbs,
    constraints,
    Distribution,
    MixtureGeneral,
)

from ...parameters import Parameters as P
from ...utils.kernel import log_planck_taper_window
from ..constraints import any_constraint
from ..utils import JointDistribution, ScaledMixture
from ._ncombination import (
    combine_distributions,
    create_beta_distributions,
    create_broken_powerlaws,
    create_generic_powerlaws,
    create_gwtc4_effective_spin_skew_normal_models,
    create_independent_spin_orientation_gaussian_isotropic,
    create_powerlaw_redshift,
    create_powerlaws,
    create_spin_magnitude_mixture_models,
    create_truncated_normal_distributions,
    create_two_truncated_normal_mixture,
    create_uniform_distributions,
)
from ._utils import _GenericSubPopulationModel, _M1_GRID_SIZE


def _build_non_mass_distributions(
    N: int,
    component_type: Literal["spl", "bpl", "g"],
    mass_distributions: List[Distribution],
    use_beta_spin_magnitude: bool,
    use_spin_magnitude_mixture: bool,
    use_truncated_normal_spin_x: bool,
    use_truncated_normal_spin_y: bool,
    use_truncated_normal_spin_z: bool,
    use_chi_eff_mixture: bool,
    use_skew_normal_chi_eff: bool,
    use_truncated_normal_chi_p: bool,
    use_tilt: bool,
    use_eccentricity_mixture: bool,
    use_eccentricity_powerlaw: bool,
    use_mean_anomaly: bool,
    use_redshift: bool,
    use_cos_iota: bool,
    use_polarization_angle: bool,
    use_right_ascension: bool,
    use_sin_declination: bool,
    use_detection_time: bool,
    use_phi_1: bool,
    use_phi_2: bool,
    use_phi_12: bool,
    use_phi_orb: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    build_distributions = mass_distributions
    # fmt: off
    _info_collection: Sequence[Tuple[bool, str, Callable[..., Sequence[Distribution]]]] = [
        (use_beta_spin_magnitude, P.PRIMARY_SPIN_MAGNITUDE, create_beta_distributions),
        (use_beta_spin_magnitude, P.SECONDARY_SPIN_MAGNITUDE, create_beta_distributions),
        # combined spin magnitude distribution
        (use_spin_magnitude_mixture, P.PRIMARY_SPIN_MAGNITUDE + "_" + P.SECONDARY_SPIN_MAGNITUDE, create_spin_magnitude_mixture_models),
        (use_truncated_normal_spin_x, P.PRIMARY_SPIN_X, create_truncated_normal_distributions),
        (use_truncated_normal_spin_x, P.SECONDARY_SPIN_X, create_truncated_normal_distributions),
        (use_truncated_normal_spin_y, P.PRIMARY_SPIN_Y, create_truncated_normal_distributions),
        (use_truncated_normal_spin_y, P.SECONDARY_SPIN_Y, create_truncated_normal_distributions),
        (use_truncated_normal_spin_z, P.PRIMARY_SPIN_Z, create_truncated_normal_distributions),
        (use_truncated_normal_spin_z, P.SECONDARY_SPIN_Z, create_truncated_normal_distributions),
        (use_chi_eff_mixture, P.EFFECTIVE_SPIN, create_two_truncated_normal_mixture),
        (use_skew_normal_chi_eff, P.EFFECTIVE_SPIN, create_gwtc4_effective_spin_skew_normal_models),
        (use_truncated_normal_chi_p, P.PRECESSING_SPIN, create_truncated_normal_distributions),
        # combined tilt distribution
        (use_tilt, P.COS_TILT_1 + "_" + P.COS_TILT_2, create_independent_spin_orientation_gaussian_isotropic),
        (use_phi_1, P.PHI_1, create_uniform_distributions),
        (use_phi_2, P.PHI_2, create_uniform_distributions),
        (use_phi_12, P.PHI_12, create_uniform_distributions),
        (use_eccentricity_mixture, P.ECCENTRICITY, create_two_truncated_normal_mixture),
        (use_eccentricity_powerlaw, P.ECCENTRICITY, create_generic_powerlaws),
        (use_mean_anomaly, P.MEAN_ANOMALY, create_uniform_distributions),
        (use_redshift, P.REDSHIFT, create_powerlaw_redshift),
        (use_right_ascension, P.RIGHT_ASCENSION, create_uniform_distributions),
        (use_sin_declination, P.SIN_DECLINATION, create_uniform_distributions),
        (use_detection_time, P.DETECTION_TIME, create_uniform_distributions),
        (use_cos_iota, P.COS_IOTA, create_uniform_distributions),
        (use_polarization_angle, P.POLARIZATION_ANGLE, create_uniform_distributions),
        (use_phi_orb, P.PHI_ORB, create_uniform_distributions),
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


def _build_component_distributions(
    N: int,
    component_type: Literal["spl", "bpl", "g"],
    use_beta_spin_magnitude: bool,
    use_spin_magnitude_mixture: bool,
    use_truncated_normal_spin_x: bool,
    use_truncated_normal_spin_y: bool,
    use_truncated_normal_spin_z: bool,
    use_chi_eff_mixture: bool,
    use_skew_normal_chi_eff: bool,
    use_truncated_normal_chi_p: bool,
    use_tilt: bool,
    use_eccentricity_mixture: bool,
    use_eccentricity_powerlaw: bool,
    use_mean_anomaly: bool,
    use_redshift: bool,
    use_cos_iota: bool,
    use_polarization_angle: bool,
    use_right_ascension: bool,
    use_sin_declination: bool,
    use_detection_time: bool,
    use_phi_1: bool,
    use_phi_2: bool,
    use_phi_12: bool,
    use_phi_orb: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> Tuple[List[Distribution], List[JointDistribution]]:
    if N == 0:
        return [], []

    if component_type == "spl":
        mass_distributions = [
            [d]
            for d in create_powerlaws(
                N=N,
                component_type=component_type,
                params=params,
                validate_args=validate_args,
            )
        ]

    if component_type == "bpl":
        mass_distributions = [
            [d]
            for d in create_broken_powerlaws(
                N=N, params=params, validate_args=validate_args
            )
        ]

    if component_type == "g":
        mass_distributions = [
            [d]
            for d in create_truncated_normal_distributions(
                N=N,
                parameter_name="m1",
                component_type=component_type,
                params=params,
                validate_args=validate_args,
            )
        ]

    build_distributions = _build_non_mass_distributions(
        N=N,
        component_type=component_type,
        mass_distributions=mass_distributions,
        use_beta_spin_magnitude=use_beta_spin_magnitude,
        use_spin_magnitude_mixture=use_spin_magnitude_mixture,
        use_truncated_normal_spin_x=use_truncated_normal_spin_x,
        use_truncated_normal_spin_y=use_truncated_normal_spin_y,
        use_truncated_normal_spin_z=use_truncated_normal_spin_z,
        use_chi_eff_mixture=use_chi_eff_mixture,
        use_skew_normal_chi_eff=use_skew_normal_chi_eff,
        use_truncated_normal_chi_p=use_truncated_normal_chi_p,
        use_tilt=use_tilt,
        use_eccentricity_mixture=use_eccentricity_mixture,
        use_eccentricity_powerlaw=use_eccentricity_powerlaw,
        use_redshift=use_redshift,
        use_cos_iota=use_cos_iota,
        use_phi_12=use_phi_12,
        use_polarization_angle=use_polarization_angle,
        use_right_ascension=use_right_ascension,
        use_sin_declination=use_sin_declination,
        use_detection_time=use_detection_time,
        use_phi_1=use_phi_1,
        use_phi_2=use_phi_2,
        use_phi_orb=use_phi_orb,
        use_mean_anomaly=use_mean_anomaly,
        params=params,
        validate_args=validate_args,
    )

    return mass_distributions, [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def SubPopulationModel(
    N_spl: int,
    N_bpl: int,
    N_g: int,
    use_beta_spin_magnitude: bool = False,
    use_spin_magnitude_mixture: bool = False,
    use_truncated_normal_spin_x: bool = False,
    use_truncated_normal_spin_y: bool = False,
    use_truncated_normal_spin_z: bool = False,
    use_chi_eff_mixture: bool = False,
    use_skew_normal_chi_eff: bool = False,
    use_truncated_normal_chi_p: bool = False,
    use_tilt: bool = False,
    use_eccentricity_mixture: bool = False,
    use_eccentricity_powerlaw: bool = False,
    use_redshift: bool = False,
    use_cos_iota: bool = False,
    use_phi_12: bool = False,
    use_polarization_angle: bool = False,
    use_right_ascension: bool = False,
    use_sin_declination: bool = False,
    use_detection_time: bool = False,
    use_phi_1: bool = False,
    use_phi_2: bool = False,
    use_phi_orb: bool = False,
    use_mean_anomaly: bool = False,
    *,
    validate_args=None,
    **params,
) -> ScaledMixture:
    component_dists = []
    mass_dist = []
    for component_type, N in zip(("spl", "bpl", "g"), (N_spl, N_bpl, N_g)):
        m_dist, _component_dists = _build_component_distributions(
            N=N,
            component_type=component_type,
            use_beta_spin_magnitude=use_beta_spin_magnitude,
            use_spin_magnitude_mixture=use_spin_magnitude_mixture,
            use_truncated_normal_spin_x=use_truncated_normal_spin_x,
            use_truncated_normal_spin_y=use_truncated_normal_spin_y,
            use_truncated_normal_spin_z=use_truncated_normal_spin_z,
            use_chi_eff_mixture=use_chi_eff_mixture,
            use_skew_normal_chi_eff=use_skew_normal_chi_eff,
            use_truncated_normal_chi_p=use_truncated_normal_chi_p,
            use_tilt=use_tilt,
            use_eccentricity_mixture=use_eccentricity_mixture,
            use_eccentricity_powerlaw=use_eccentricity_powerlaw,
            use_redshift=use_redshift,
            use_cos_iota=use_cos_iota,
            use_phi_12=use_phi_12,
            use_polarization_angle=use_polarization_angle,
            use_right_ascension=use_right_ascension,
            use_sin_declination=use_sin_declination,
            use_detection_time=use_detection_time,
            use_phi_1=use_phi_1,
            use_phi_2=use_phi_2,
            use_phi_orb=use_phi_orb,
            use_mean_anomaly=use_mean_anomaly,
            params=params,
            validate_args=validate_args,
        )
        mass_dist.extend(m_dist)
        component_dists.extend(_component_dists)

    N = N_spl + N_bpl + N_g
    _lambdas = [params.pop(f"lambda_{i}") for i in range(N - 1)]
    _lambdas.append(1.0 - sum(_lambdas))
    lambdas = jnp.stack(_lambdas, axis=-1)

    beta = params.pop("beta")
    delta_m1 = params.pop("delta_m1")
    delta_m2 = params.pop("delta_m2")
    log_rate = params.pop("log_rate")
    log_rate = params.pop("log_rate")
    m1max = params.pop("m1max")
    m1min = params.pop("m1min")
    m2min = params.pop("m2min")

    mixing_distribution = CategoricalProbs(probs=lambdas, validate_args=validate_args)
    mass_dist_mixture = MixtureGeneral(
        mixing_distribution,
        mass_dist,
        support=constraints.interval(m1min, m1max),
        validate_args=validate_args,
    )

    mm = jnp.linspace(m1min, m1max, _M1_GRID_SIZE)
    safe_delta_m = jnp.where(delta_m1 <= 0.0, 1.0, delta_m1)
    _log_prob_m1 = mass_dist_mixture.log_prob(mm) + log_planck_taper_window(
        (mm - m1min) / safe_delta_m
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

    return _GenericSubPopulationModel(
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
