# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Optional, TypeVar

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import (
    Beta,
    Distribution,
    MixtureGeneral,
    TruncatedNormal,
    Uniform,
)

from ...parameters import Parameters as P
from ..mass import (
    BrokenPowerlaw,
    GaussianPrimaryMassRatio,
    GenericSmoothedPowerlawMassRatio,
    PowerlawPrimaryMassRatio,
    SmoothedBrokenPowerlawMassRatioPowerlaw,
    SmoothedGaussianPrimaryMassRatio,
    SmoothedPowerlawPrimaryMassRatio,
)
from ..redshift import MadauDickinsonRedshiftModel, PowerlawRedshiftModel
from ..spin import BetaFromMeanVar, GenericTiltModel, GWTC4EffectiveSpinSkewNormalModel
from ..sundry import NDTwoTruncatedNormalMixture, TwoTruncatedNormalMixture
from ..transformations import PrimaryMassAndMassRatioToComponentMassesTransform
from ..utils import DoublyTruncatedPowerLaw, ExtendedSupportTransformedDistribution


__all__ = [
    "combine_distributions",
    "create_beta_distributions",
    "create_broken_powerlaws",
    "create_gaussian_primary_mass_ratio",
    "create_generic_powerlaws",
    "create_generic_smoothed_powerlaw_mass_ratio",
    "create_gwtc4_effective_spin_skew_normal_models",
    "create_madau_dickinson_redshift_model",
    "create_generic_tilt_model",
    "create_powerlaw_primary_mass_ratios",
    "create_powerlaw_redshift_model",
    "create_powerlaws",
    "create_smoothed_broken_powerlaws_mass_ratio_powerlaw",
    "create_smoothed_gaussian_primary_mass_ratio",
    "create_smoothed_powerlaw_primary_mass_ratio",
    "create_spin_magnitude_mixture_models",
    "create_truncated_normal_distributions",
    "create_two_truncated_normal_mixture",
    "create_uniform_distributions",
]


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _get_parameter(
    params: Dict[_KT, _VT],
    name: _KT,
    is_necessary: bool = True,
    default: Optional[_VT] = None,
) -> Optional[_VT]:
    if (value := params.get(name, None)) is not None:
        return value
    if default is not None:
        return default
    if is_necessary:
        raise ValueError(f"Missing parameter {name}")
    return None


def combine_distributions(
    base_dists: List[List[Distribution]], add_dists: List[Distribution]
):
    """Helper function to combine base distributions with additional distributions like
    spin, tilt, or eccentricity.
    """
    return [dists + [add_dist] for dists, add_dist in zip(base_dists, add_dists)]


def create_beta_distributions(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Beta]:
    mean_name = f"{parameter_name}_mean_{component_type}"
    variance_name = f"{parameter_name}_variance_{component_type}"

    return [
        BetaFromMeanVar(
            mean=_get_parameter(params, f"{mean_name}_{i}"),  # type: ignore
            variance=_get_parameter(params, f"{variance_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_truncated_normal_distributions(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    loc_name = f"{parameter_name}_loc_{component_type}"
    scale_name = f"{parameter_name}_scale_{component_type}"
    low_name = f"{parameter_name}_low_{component_type}"
    high_name = f"{parameter_name}_high_{component_type}"

    # fmt: off
    return [
        TruncatedNormal(
            loc=_get_parameter(params, f"{loc_name}_{i}"),  # type: ignore
            scale=_get_parameter(params, f"{scale_name}_{i}"),  # type: ignore
            low=_get_parameter(params, f"{low_name}_{i}", is_necessary=False),  # type: ignore
            high=_get_parameter(params, f"{high_name}_{i}", is_necessary=False),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]
    # fmt: on


def create_powerlaw_primary_mass_ratios(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[ExtendedSupportTransformedDistribution]:
    powerlaws_collection = []

    alpha_name = "alpha_" + component_type
    beta_name = "beta_" + component_type
    mmin_name = "mmin_" + component_type
    mmax_name = "mmax_" + component_type

    for i in range(N):
        powerlaw = PowerlawPrimaryMassRatio(
            alpha=_get_parameter(params, f"{alpha_name}_{i}"),  # type: ignore
            beta=_get_parameter(params, f"{beta_name}_{i}"),  # type: ignore
            mmin=_get_parameter(params, f"{mmin_name}_{i}"),  # type: ignore
            mmax=_get_parameter(params, f"{mmax_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        transformed_powerlaw = ExtendedSupportTransformedDistribution(
            base_distribution=powerlaw,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )
        powerlaws_collection.append(transformed_powerlaw)
    return powerlaws_collection


def create_powerlaw_redshift_model(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    kappa_name = parameter_name + "_kappa_" + component_type
    z_max_name = parameter_name + "_z_max_" + component_type

    return [
        PowerlawRedshiftModel(
            kappa=_get_parameter(params, f"{kappa_name}_{i}"),  # type: ignore
            z_max=_get_parameter(params, f"{z_max_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_madau_dickinson_redshift_model(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    kappa_name = parameter_name + "_kappa_" + component_type
    z_max_name = parameter_name + "_z_max_" + component_type
    gamma_name = parameter_name + "_gamma_" + component_type
    z_peak_name = parameter_name + "_z_peak_" + component_type
    return [
        MadauDickinsonRedshiftModel(
            kappa=_get_parameter(params, f"{kappa_name}_{i}"),  # type: ignore
            z_max=_get_parameter(params, f"{z_max_name}_{i}"),  # type: ignore
            gamma=_get_parameter(params, f"{gamma_name}_{i}"),  # type: ignore
            z_peak=_get_parameter(params, f"{z_peak_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_uniform_distributions(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    low_name = f"{parameter_name}_low_{component_type}"
    high_name = f"{parameter_name}_high_{component_type}"

    return [
        Uniform(
            low=_get_parameter(params, f"{low_name}_{i}"),  # type: ignore
            high=_get_parameter(params, f"{high_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_broken_powerlaws(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    alpha1_name = parameter_name + "_alpha1_" + component_type
    alpha2_name = parameter_name + "_alpha2_" + component_type
    mbreak_name = parameter_name + "_break_" + component_type
    mmax_name = parameter_name + "_high_" + component_type
    mmin_name = parameter_name + "_low_" + component_type
    return [
        BrokenPowerlaw(
            alpha1=_get_parameter(params, f"{alpha1_name}_{i}"),  # type: ignore
            alpha2=_get_parameter(params, f"{alpha2_name}_{i}"),  # type: ignore
            mbreak=_get_parameter(params, f"{mbreak_name}_{i}"),  # type: ignore
            mmin=_get_parameter(params, f"{mmin_name}_{i}"),  # type: ignore
            mmax=_get_parameter(params, f"{mmax_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_generic_tilt_model(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[MixtureGeneral]:
    zeta_name = "cos_tilt_zeta_" + component_type
    loc1_name = P.COS_TILT_1 + "_loc_" + component_type
    loc2_name = P.COS_TILT_2 + "_loc_" + component_type
    scale1_name = P.COS_TILT_1 + "_scale_" + component_type
    scale2_name = P.COS_TILT_2 + "_scale_" + component_type
    low1_name = P.COS_TILT_1 + "_low_" + component_type
    low2_name = P.COS_TILT_2 + "_low_" + component_type
    high1_name = P.COS_TILT_1 + "_high_" + component_type
    high2_name = P.COS_TILT_2 + "_high_" + component_type

    return [
        GenericTiltModel(
            zeta=_get_parameter(params, f"{zeta_name}_{i}"),  # type: ignore
            loc1=_get_parameter(params, f"{loc1_name}_{i}"),  # type: ignore
            loc2=_get_parameter(params, f"{loc2_name}_{i}"),  # type: ignore
            scale1=_get_parameter(params, f"{scale1_name}_{i}"),  # type: ignore
            scale2=_get_parameter(params, f"{scale2_name}_{i}"),  # type: ignore
            low1=_get_parameter(params, f"{low1_name}_{i}", default=-1.0),  # type: ignore
            low2=_get_parameter(params, f"{low2_name}_{i}", default=-1.0),  # type: ignore
            high1=_get_parameter(params, f"{high1_name}_{i}", default=1.0),  # type: ignore
            high2=_get_parameter(params, f"{high2_name}_{i}", default=1.0),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_powerlaws(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    alpha_name = parameter_name + "_alpha_" + component_type
    mmax_name = parameter_name + "_high_" + component_type
    mmin_name = parameter_name + "_low_" + component_type

    return [
        DoublyTruncatedPowerLaw(
            alpha=-_get_parameter(params, f"{alpha_name}_{i}"),  # type: ignore
            mmin=_get_parameter(params, f"{mmin_name}_{i}"),  # type: ignore
            mmax=_get_parameter(params, f"{mmax_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_generic_powerlaws(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    alpha_name = parameter_name + "_alpha_" + component_type
    high_name = parameter_name + "_high_" + component_type
    low_name = parameter_name + "_low_" + component_type
    return [
        DoublyTruncatedPowerLaw(
            alpha=_get_parameter(params, f"{alpha_name}_{i}"),  # type: ignore
            low=_get_parameter(params, f"{low_name}_{i}"),  # type: ignore
            high=_get_parameter(params, f"{high_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_two_truncated_normal_mixture(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[MixtureGeneral]:
    comp1_high_name = parameter_name + "_comp1_high_" + component_type
    comp2_high_name = parameter_name + "_comp2_high_" + component_type
    comp1_loc_name = parameter_name + "_comp1_loc_" + component_type
    comp2_loc_name = parameter_name + "_comp2_loc_" + component_type
    comp1_low_name = parameter_name + "_comp1_low_" + component_type
    comp2_low_name = parameter_name + "_comp2_low_" + component_type
    comp1_scale_name = parameter_name + "_comp1_scale_" + component_type
    comp2_scale_name = parameter_name + "_comp2_scale_" + component_type
    zeta_name = parameter_name + "_zeta_" + component_type

    # fmt: off
    return [
        TwoTruncatedNormalMixture(
            comp1_high=_get_parameter(params, f"{comp1_high_name}_{i}", is_necessary=False),  # type: ignore
            comp2_high=_get_parameter(params, f"{comp2_high_name}_{i}", is_necessary=False),  # type: ignore
            comp1_loc=_get_parameter(params, f"{comp1_loc_name}_{i}"),  # type: ignore
            comp2_loc=_get_parameter(params, f"{comp2_loc_name}_{i}"),  # type: ignore
            comp1_low=_get_parameter(params, f"{comp1_low_name}_{i}", is_necessary=False),  # type: ignore
            comp2_low=_get_parameter(params, f"{comp2_low_name}_{i}", is_necessary=False),  # type: ignore
            comp1_scale=_get_parameter(params, f"{comp1_scale_name}_{i}"),  # type: ignore
            comp2_scale=_get_parameter(params, f"{comp2_scale_name}_{i}"),  # type: ignore
            zeta=_get_parameter(params, f"{zeta_name}_{i}", zeta_name),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]
    # fmt: on


def create_spin_magnitude_mixture_models(
    N: int,
    parameter_name,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
):
    # fmt: off
    zeta_name = "a_zeta_" + component_type
    a_1_comp1_high_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_" + component_type
    a_1_comp1_loc_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_" + component_type
    a_1_comp1_low_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_" + component_type
    a_1_comp1_scale_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_" + component_type
    a_1_comp2_high_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_" + component_type
    a_1_comp2_loc_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_" + component_type
    a_1_comp2_low_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_" + component_type
    a_1_comp2_scale_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_" + component_type
    a_2_comp1_high_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_" + component_type
    a_2_comp1_loc_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_" + component_type
    a_2_comp1_low_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_" + component_type
    a_2_comp1_scale_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_" + component_type
    a_2_comp2_high_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_" + component_type
    a_2_comp2_loc_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_" + component_type
    a_2_comp2_low_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_" + component_type
    a_2_comp2_scale_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_" + component_type
    # fmt: on

    spin_collection = []

    for i in range(N):
        # fmt: off
        zeta = _get_parameter(params, f"{zeta_name}_{i}", zeta_name)
        a_1_comp1_high: ArrayLike = _get_parameter(params, f"{a_1_comp1_high_name}_{i}", default=1.0) # type: ignore
        a_1_comp1_loc: ArrayLike = _get_parameter(params, f"{a_1_comp1_loc_name}_{i}") # type: ignore
        a_1_comp1_low: ArrayLike = _get_parameter(params, f"{a_1_comp1_low_name}_{i}", default=0.0) # type: ignore
        a_1_comp1_scale: ArrayLike = _get_parameter(params, f"{a_1_comp1_scale_name}_{i}") # type: ignore
        a_1_comp2_high: ArrayLike = _get_parameter(params, f"{a_1_comp2_high_name}_{i}", default=1.0) # type: ignore
        a_1_comp2_loc: ArrayLike = _get_parameter(params, f"{a_1_comp2_loc_name}_{i}") # type: ignore
        a_1_comp2_low: ArrayLike = _get_parameter(params, f"{a_1_comp2_low_name}_{i}", default=0.0) # type: ignore
        a_1_comp2_scale: ArrayLike = _get_parameter(params, f"{a_1_comp2_scale_name}_{i}") # type: ignore
        a_2_comp1_high: ArrayLike = _get_parameter(params, f"{a_2_comp1_high_name}_{i}", default=1.0) # type: ignore
        a_2_comp1_loc: ArrayLike = _get_parameter(params, f"{a_2_comp1_loc_name}_{i}") # type: ignore
        a_2_comp1_low: ArrayLike = _get_parameter(params, f"{a_2_comp1_low_name}_{i}", default=0.0) # type: ignore
        a_2_comp1_scale: ArrayLike = _get_parameter(params, f"{a_2_comp1_scale_name}_{i}") # type: ignore
        a_2_comp2_high: ArrayLike = _get_parameter(params, f"{a_2_comp2_high_name}_{i}", default=1.0) # type: ignore
        a_2_comp2_loc: ArrayLike = _get_parameter(params, f"{a_2_comp2_loc_name}_{i}") # type: ignore
        a_2_comp2_low: ArrayLike = _get_parameter(params, f"{a_2_comp2_low_name}_{i}", default=0.0) # type: ignore
        a_2_comp2_scale: ArrayLike = _get_parameter(params, f"{a_2_comp2_scale_name}_{i}") # type: ignore
        # fmt: on

        comp1_high = jnp.stack((a_1_comp1_high, a_2_comp1_high), axis=-1)
        comp1_loc = jnp.stack((a_1_comp1_loc, a_2_comp1_loc), axis=-1)
        comp1_low = jnp.stack((a_1_comp1_low, a_2_comp1_low), axis=-1)
        comp1_scale = jnp.stack((a_1_comp1_scale, a_2_comp1_scale), axis=-1)
        comp2_high = jnp.stack((a_1_comp2_high, a_2_comp2_high), axis=-1)
        comp2_loc = jnp.stack((a_1_comp2_loc, a_2_comp2_loc), axis=-1)
        comp2_low = jnp.stack((a_1_comp2_low, a_2_comp2_low), axis=-1)
        comp2_scale = jnp.stack((a_1_comp2_scale, a_2_comp2_scale), axis=-1)

        spin_dist = NDTwoTruncatedNormalMixture(
            zeta=zeta,
            comp1_high=comp1_high,
            comp1_loc=comp1_loc,
            comp1_low=comp1_low,
            comp1_scale=comp1_scale,
            comp2_high=comp2_high,
            comp2_loc=comp2_loc,
            comp2_low=comp2_low,
            comp2_scale=comp2_scale,
            validate_args=validate_args,
        )

        spin_collection.append(spin_dist)

    return spin_collection


def create_gwtc4_effective_spin_skew_normal_models(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    loc_name = parameter_name + "_loc_" + component_type
    scale_name = parameter_name + "_scale_" + component_type
    epsilon_name = parameter_name + "_epsilon_" + component_type

    return [
        GWTC4EffectiveSpinSkewNormalModel(
            loc=_get_parameter(params, f"{loc_name}_{i}"),  # type: ignore
            scale=_get_parameter(params, f"{scale_name}_{i}"),  # type: ignore
            epsilon=_get_parameter(params, f"{epsilon_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_smoothed_broken_powerlaws_mass_ratio_powerlaw(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    collection = []

    alpha1_name = "m1_alpha1_" + component_type
    alpha2_name = "m1_alpha2_" + component_type
    beta_name = "beta_" + component_type
    delta_m1_name = "m1_delta_" + component_type
    delta_m2_name = "m2_delta_" + component_type
    m1min_name = "m1_low_" + component_type
    m2min_name = "m2_low_" + component_type
    mbreak_name = "m1_break_" + component_type
    mmax_name = "m1_high_" + component_type

    for i in range(N):
        suffix = f"_{i}"
        broken_powerlaw = SmoothedBrokenPowerlawMassRatioPowerlaw(
            alpha1=_get_parameter(params, alpha1_name + suffix),  # type: ignore
            alpha2=_get_parameter(params, alpha2_name + suffix),  # type: ignore
            beta=_get_parameter(params, beta_name + suffix),  # type: ignore
            delta_m1=_get_parameter(params, delta_m1_name + suffix),  # type: ignore
            delta_m2=_get_parameter(params, delta_m2_name + suffix),  # type: ignore
            m1min=_get_parameter(params, m1min_name + suffix),  # type: ignore
            m2min=_get_parameter(params, m2min_name + suffix),  # type: ignore
            mbreak=_get_parameter(params, mbreak_name + suffix),  # type: ignore
            mmax=_get_parameter(params, mmax_name + suffix),  # type: ignore
            validate_args=validate_args,
        )
        distribution = ExtendedSupportTransformedDistribution(
            base_distribution=broken_powerlaw,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )

        collection.append(distribution)
    return collection


def create_smoothed_gaussian_primary_mass_ratio(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    collection = []

    loc_name = "loc_" + component_type
    scale_name = "scale_" + component_type
    beta_name = "beta_" + component_type
    m1min_name = "m1min_" + component_type
    m2min_name = "m2min_" + component_type
    mmax_name = "mmax_" + component_type
    delta_m1_name = "delta_m1_" + component_type
    delta_m2_name = "delta_m2_" + component_type

    for i in range(N):
        suffix = f"_{i}"

        smoothed_gaussian = SmoothedGaussianPrimaryMassRatio(
            loc=_get_parameter(params, loc_name + suffix),  # type: ignore
            scale=_get_parameter(params, scale_name + suffix),  # type: ignore
            beta=_get_parameter(params, beta_name + suffix),  # type: ignore
            m1min=_get_parameter(params, m1min_name + suffix),  # type: ignore
            m2min=_get_parameter(params, m2min_name + suffix),  # type: ignore
            mmax=_get_parameter(params, mmax_name + suffix),  # type: ignore
            delta_m1=_get_parameter(params, delta_m1_name + suffix),  # type: ignore
            delta_m2=_get_parameter(params, delta_m2_name + suffix),  # type: ignore
            validate_args=validate_args,
        )

        distribution = ExtendedSupportTransformedDistribution(
            base_distribution=smoothed_gaussian,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )

        collection.append(distribution)
    return collection


def create_gaussian_primary_mass_ratio(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    collection = []

    loc_name = "m1_loc_" + component_type
    scale_name = "m1_scale_" + component_type
    beta_name = "beta_" + component_type
    mmin_name = "m1_low_" + component_type
    mmax_name = "m1_high_" + component_type

    for i in range(N):
        suffix = f"_{i}"
        gaussian = GaussianPrimaryMassRatio(
            loc=_get_parameter(params, loc_name + suffix),  # type: ignore
            scale=_get_parameter(params, scale_name + suffix),  # type: ignore
            beta=_get_parameter(params, beta_name + suffix),  # type: ignore
            mmin=_get_parameter(params, mmin_name + suffix),  # type: ignore
            mmax=_get_parameter(params, mmax_name + suffix),  # type: ignore
            validate_args=validate_args,
        )

        distribution = ExtendedSupportTransformedDistribution(
            base_distribution=gaussian,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )

        collection.append(distribution)
    return collection


def create_smoothed_powerlaw_primary_mass_ratio(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    collection = []

    alpha_name = "m1_alpha_" + component_type
    beta_name = "beta_" + component_type
    delta_m1_name = "m1_delta_" + component_type
    delta_m2_name = "m2_delta_" + component_type
    m1min_name = "m1_low_" + component_type
    m2min_name = "m2_low_" + component_type
    mmax_name = "m1_high_" + component_type

    for i in range(N):
        suffix = f"_{i}"
        broken_powerlaw = SmoothedPowerlawPrimaryMassRatio(
            alpha=_get_parameter(params, alpha_name + suffix),  # type: ignore
            beta=_get_parameter(params, beta_name + suffix),  # type: ignore
            delta_m1=_get_parameter(params, delta_m1_name + suffix),  # type: ignore
            delta_m2=_get_parameter(params, delta_m2_name + suffix),  # type: ignore
            m1min=_get_parameter(params, m1min_name + suffix),  # type: ignore
            m2min=_get_parameter(params, m2min_name + suffix),  # type: ignore
            mmax=_get_parameter(params, mmax_name + suffix),  # type: ignore
            validate_args=validate_args,
        )

        distribution = ExtendedSupportTransformedDistribution(
            base_distribution=broken_powerlaw,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )

        collection.append(distribution)
    return collection


def create_generic_smoothed_powerlaw_mass_ratio(
    N: int,
    primary_mass_distributions: List[Distribution],
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:

    beta_name = "beta_" + component_type
    delta_m1_name = "m1_delta"
    delta_m2_name = "m2_delta_" + component_type
    m2min_name = "m2_low_" + component_type

    delta_m1 = _get_parameter(params, delta_m1_name)

    return [
        GenericSmoothedPowerlawMassRatio(
            primary_mass_distribution=primary_mass_distributions[i],
            delta_m1=delta_m1,  # type: ignore
            beta=_get_parameter(params, f"{beta_name}_{i}"),  # type: ignore
            delta_m2=_get_parameter(params, f"{delta_m2_name}_{i}"),  # type: ignore
            m2min=_get_parameter(params, f"{m2min_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]
