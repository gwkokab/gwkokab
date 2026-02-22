# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, List, Literal, Optional, Tuple

from jax import numpy as jnp, tree as jtr
from jaxtyping import Array
from numpyro.distributions import Distribution

from ...parameters import Parameters as P
from ..constraints import any_constraint
from ..utils import (
    ExtendedSupportTransformedDistribution,
    JointDistribution,
    ScaledMixture,
)
from ._ncombination import (
    combine_distributions,
    create_beta_distributions,
    create_gwtc4_effective_spin_skew_normal_models,
    create_independent_spin_orientation_gaussian_isotropic,
    create_powerlaw_redshift,
    create_smoothed_broken_powerlaws_mass_ratio_powerlaw,
    create_smoothed_gaussian_primary_mass_ratio,
    create_spin_magnitude_mixture_models,
    create_truncated_normal_distributions,
    create_two_truncated_normal_mixture,
    create_uniform_distributions,
)


def _build_non_mass_distributions(
    N: int,
    component_type: Literal["sbpl", "sg"],
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
    """Build distributions for non-mass parameters.

    Parameters
    ----------
    N : int
        Number of components
    component_type : Literal[&quot;pl&quot;, &quot;g&quot;]
        type of component, either "pl" or "g"
    mass_distributions : List[Distribution]
        list of mass distributions
    use_spin : bool
        whether to include spin
    use_tilt : bool
        whether to include tilt
    use_eccentricity_mixture : bool
        whether to include eccentricity
    use_mean_anomaly : bool
        whether to include mean_anomaly
    use_redshift : bool
        whether to include redshift
    use_cos_iota : bool
        whether to include cos_iota
    use_polarization_angle : bool
        whether to include polarization_angle
    use_right_ascension : bool
        whether to include right_ascension
    use_sin_declination : bool
        whether to include sin_declination
    use_detection_time : bool
        whether to include detection_time
    use_phi_1 : bool
        whether to include phi_1
    use_phi_2 : bool
        whether to include phi_2
    use_phi_12 : bool
        whether to include phi_12
    use_phi_orb : bool
        whether to include phi_orb
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, by default None

    Returns
    -------
    List[Distribution]
        list of distributions
    """
    build_distributions = mass_distributions
    # fmt: off
    _info_collection: List[Tuple[bool, str, Callable[..., List[Distribution]]]] = [
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
    component_type: Literal["sbpl", "sgpl", "gg"],
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
) -> List[JointDistribution]:
    """Build distributions for Gaussian components.

    Parameters
    ----------
    N : int
        Number of components
    use_spin : bool
        whether to include spin
    use_tilt : bool
        whether to include tilt
    use_eccentricity_mixture : bool
        whether to include eccentricity
    use_mean_anomaly : bool
        whether to include mean_anomaly
    use_redshift : bool
        whether to include redshift
    use_cos_iota : bool
        whether to include cos_iota
    use_polarization_angle : bool
        whether to include polarization_angle
    use_right_ascension : bool
        whether to include right_ascension
    use_sin_declination : bool
        whether to include sin_declination
    use_detection_time : bool
        whether to include detection_time
    use_phi_1 : bool
        whether to include phi_1
    use_phi_2 : bool
        whether to include phi_2
    use_phi_12 : bool
        whether to include phi_12
    use_phi_orb : bool
        whether to include phi_orb
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, by default None

    Returns
    -------
    List[JointDistribution]
        list of JointDistribution
    """
    if N == 0:
        return []
    if component_type == "sbpl":
        powerlaws = create_smoothed_broken_powerlaws_mass_ratio_powerlaw(
            N=N, params=params, validate_args=validate_args
        )
        mass_distributions = jtr.map(
            lambda powerlaw: [powerlaw],
            powerlaws,
            is_leaf=lambda x: isinstance(x, ExtendedSupportTransformedDistribution),
        )

    if component_type == "sgpl":
        powerlaws = create_smoothed_gaussian_primary_mass_ratio(
            N=N, params=params, validate_args=validate_args
        )
        mass_distributions = jtr.map(
            lambda powerlaw: [powerlaw],
            powerlaws,
            is_leaf=lambda x: isinstance(x, ExtendedSupportTransformedDistribution),
        )

    if component_type == "gg":
        m1_dists = create_truncated_normal_distributions(
            N=N,
            parameter_name="m1",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        m2_dists = create_truncated_normal_distributions(
            N=N,
            parameter_name="m2",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )

        mass_distributions = jtr.map(
            lambda m1, m2: [m1, m2],
            m1_dists,
            m2_dists,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

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

    return [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def NSmoothedBrokenPowerlawMSmoothedGaussian(
    N_sbpl: int,
    N_sgpl: int,
    N_gg: int,
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
    sbpl_component_dist = _build_component_distributions(
        N=N_sbpl,
        component_type="sbpl",
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

    sgpl_component_dist = _build_component_distributions(
        N=N_sgpl,
        component_type="sgpl",
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

    gg_component_dist = _build_component_distributions(
        N=N_gg,
        component_type="gg",
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

    component_dists = sbpl_component_dist + sgpl_component_dist + gg_component_dist

    N = N_sbpl + N_sgpl + N_gg
    log_rates = jnp.stack([params[f"log_rate_{i}"] for i in range(N)], axis=-1)

    return ScaledMixture(
        log_rates,
        component_dists,
        support=any_constraint(
            [component_dists.support for component_dists in component_dists]
        ),
        validate_args=validate_args,
    )
