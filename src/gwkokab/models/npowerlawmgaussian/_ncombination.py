# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Literal, Optional

from jaxtyping import Array
from numpyro.distributions import (
    Beta,
    Distribution,
    MixtureGeneral,
    Normal,
    TruncatedNormal,
    TwoSidedTruncatedDistribution,
    Uniform,
)

from ...utils.tools import fetch_first_matching_value
from .._models import PowerlawPrimaryMassRatio
from ..redshift import PowerlawRedshift
from ..spin import BetaFromMeanVar, IndependentSpinOrientationGaussianIsotropic
from ..transformations import PrimaryMassAndMassRatioToComponentMassesTransform
from ..utils import ExtendedSupportTransformedDistribution


__all__ = [
    "combine_distributions",
    "create_beta_distributions",
    "create_powerlaw_redshift",
    "create_powerlaws",
    "create_truncated_normal_distributions_for_cos_tilt",
    "create_truncated_normal_distributions",
]


def combine_distributions(
    base_dists: List[List[Distribution]], add_dists: List[Distribution]
):
    """Helper function to combine base distributions with additional distributions like
    spin, tilt, or eccentricity.
    """
    return [dists + [add_dist] for dists, add_dist in zip(base_dists, add_dists)]


def create_beta_distributions(
    N: int,
    parameter_name: Literal["chi1", "chi2"],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Beta]:
    """Create a list of Beta distributions.

    Parameters
    ----------
    N : int
        Number of components
    parameter_name : Literal[&quot;chi1&quot;, &quot;chi2&quot;]
        name of the parameter to create distributions for
    component_type : Literal[&quot;pl&quot;, &quot;g&quot;]
        type of component, either "pl" or "g"
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, defaults to None, by default None

    Returns
    -------
    List[Beta]
        list of Beta distributions

    Raises
    ------
    ValueError
        if mean or variance is missing
    """

    beta_collection = []
    mean_name = f"{parameter_name}_mean_{component_type}"
    variance_name = f"{parameter_name}_variance_{component_type}"
    for i in range(N):
        mean = fetch_first_matching_value(params, f"{mean_name}_{i}", mean_name)
        if mean is None:
            raise ValueError(f"Missing parameter {mean_name}_{i}")
        variance = fetch_first_matching_value(
            params, f"{variance_name}_{i}", variance_name
        )
        if variance is None:
            raise ValueError(f"Missing parameter {variance_name}_{i}")

        beta_collection.append(
            BetaFromMeanVar(
                mean=mean,
                variance=variance,
                validate_args=validate_args,
            )
        )
    return beta_collection


def create_truncated_normal_distributions(
    N: int,
    parameter_name: Literal[
        "m1",
        "m2",
        "chi1",
        "chi2",
        "cos_tilt1",
        "cos_tilt2",
        "ecc",
        "cos_iota",
        "phi_12",
        "polarization_angle",
        "right_ascension",
        "sin_declination",
    ],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Create a list of TruncatedNormal distributions.

    Parameters
    ----------
    N : int
        Number of components
    parameter_name : Literal[ &quot;m1&quot;, &quot;m2&quot;, &quot;chi1&quot;, &quot;chi2&quot;, &quot;cos_tilt1&quot;, &quot;cos_tilt2&quot;, &quot;ecc&quot;, &quot;cos_iota&quot;, &quot;phi_12&quot;, &quot;polarization_angle&quot;, &quot;right_ascension&quot;, &quot;sin_declination&quot;, ]
        name of the parameter to create distributions for
    component_type : Literal[&quot;pl&quot;, &quot;g&quot;]
        type of component, either "pl" or "g"
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, defaults to None, by default None

    Returns
    -------
    List[Distribution]
        list of TruncatedNormal distributions

    Raises
    ------
    ValueError
        if loc, scale, low, or high is missing
    """
    truncated_normal_collection = []
    loc_name = f"{parameter_name}_loc_{component_type}"
    scale_name = f"{parameter_name}_scale_{component_type}"
    low_name = f"{parameter_name}_low_{component_type}"
    high_name = f"{parameter_name}_high_{component_type}"
    for i in range(N):
        loc = fetch_first_matching_value(params, f"{loc_name}_{i}", loc_name)
        if loc is None:
            raise ValueError(f"Missing parameter {loc_name}_{i}")

        scale = fetch_first_matching_value(params, f"{scale_name}_{i}", scale_name)
        if scale is None:
            raise ValueError(f"Missing parameter {scale_name}_{i}")

        low = fetch_first_matching_value(params, f"{low_name}_{i}", low_name)
        if low is None:
            raise ValueError(f"Missing parameter {low_name}_{i}")

        high = fetch_first_matching_value(params, f"{high_name}_{i}", high_name)
        if high is None:
            raise ValueError(f"Missing parameter {high_name}_{i}")

        truncated_normal_collection.append(
            TruncatedNormal(
                loc=loc, scale=scale, low=low, high=high, validate_args=validate_args
            )
        )

    return truncated_normal_collection


def create_truncated_normal_distributions_for_cos_tilt(
    N: int,
    parameter_name: Literal["cos_tilt1", "cos_tilt2"],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[TwoSidedTruncatedDistribution]:
    """Create a list of TwoSidedTruncatedDistribution distributions for tilt.

    Parameters
    ----------
    N : int
        Number of components
    parameter_name : Literal[&quot;cos_tilt1&quot;, &quot;cos_tilt2&quot;]
        name of the parameter to create distributions for
    component_type : Literal[&quot;pl&quot;, &quot;g&quot;]
        type of component, either "pl" or "g"
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, defaults to None, by default None

    Returns
    -------
    List[TwoSidedTruncatedDistribution]
        list of TwoSidedTruncatedDistribution distributions

    Raises
    ------
    ValueError
        if scale is missing
    """
    truncated_normal_for_tilt_collection = []
    scale_name = f"{parameter_name}_scale_{component_type}"
    for i in range(N):
        scale = fetch_first_matching_value(params, f"{scale_name}_{i}", scale_name)
        if scale is None:
            raise ValueError(f"Missing parameter {scale_name}_{i}")

        truncated_normal_for_tilt_collection.append(
            TwoSidedTruncatedDistribution(
                Normal(loc=1.0, scale=scale, validate_args=validate_args),
                low=-1,
                high=1,
                validate_args=validate_args,
            )
        )

    return truncated_normal_for_tilt_collection


def create_independent_spin_orientation_gaussian_isotropic(
    N: int,
    parameter_name: Literal["cos_tilt1", "cos_tilt2"],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[MixtureGeneral]:
    """Create a list of :func:`IndependentSpinOrientationGaussianIsotropic`
    distributions for tilt.

    Parameters
    ----------
    N : int
        Number of components
    parameter_name : Literal[&quot;cos_tilt1&quot;, &quot;cos_tilt2&quot;]
        name of the parameter to create distributions for
    component_type : Literal[&quot;pl&quot;, &quot;g&quot;]
        type of component, either "pl" or "g"
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, defaults to None, by default None

    Returns
    -------
    List[MixtureGeneral]
        list of :func:`IndependentSpinOrientationGaussianIsotropic` distributions

    Raises
    ------
    ValueError
        if scale is missing
    """
    dist_collection = []
    zeta_name = f"cos_tilt_zeta_{component_type}"
    scale1_name = f"cos_tilt1_scale_{component_type}"
    scale2_name = f"cos_tilt2_scale_{component_type}"

    for i in range(N):
        zeta = fetch_first_matching_value(params, f"{zeta_name}_{i}", zeta_name)
        if zeta is None:
            raise ValueError(f"Missing parameter {zeta_name}_{i}")

        scale1 = fetch_first_matching_value(params, f"{scale1_name}_{i}", scale1_name)
        if scale1 is None:
            raise ValueError(f"Missing parameter {scale1_name}_{i}")

        scale2 = fetch_first_matching_value(params, f"{scale2_name}_{i}", scale2_name)
        if scale2 is None:
            raise ValueError(f"Missing parameter {scale2_name}_{i}")

        dist_collection.append(
            IndependentSpinOrientationGaussianIsotropic(
                zeta=zeta,
                scale1=scale1,
                scale2=scale2,
                validate_args=validate_args,
            )
        )

    return dist_collection


def create_powerlaws(
    N: int,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[ExtendedSupportTransformedDistribution]:
    """Create a list of ExtendedSupportTransformedDistribution for powerlaws.

    Parameters
    ----------
    N : int
        Number of components
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, defaults to None, by default None

    Returns
    -------
    List[ExtendedSupportTransformedDistribution]
        list of ExtendedSupportTransformedDistribution for powerlaws

    Raises
    ------
    ValueError
        if alpha, beta, mmin, or mmax is missing
    """
    powerlaws_collection = []
    alpha_name = "alpha_pl"
    beta_name = "beta_pl"
    mmin_name = "mmin_pl"
    mmax_name = "mmax_pl"
    for i in range(N):
        alpha = fetch_first_matching_value(params, f"{alpha_name}_{i}", alpha_name)
        if alpha is None:
            raise ValueError(f"Missing parameter {alpha_name}_{i}")

        beta = fetch_first_matching_value(params, f"{beta_name}_{i}", beta_name)
        if beta is None:
            raise ValueError(f"Missing parameter {beta_name}_{i}")

        mmin = fetch_first_matching_value(params, f"{mmin_name}_{i}", mmin_name)
        if mmin is None:
            raise ValueError(f"Missing parameter {mmin_name}_{i}")

        mmax = fetch_first_matching_value(params, f"{mmax_name}_{i}", mmax_name)
        if mmax is None:
            raise ValueError(f"Missing parameter {mmax_name}_{i}")

        powerlaw = PowerlawPrimaryMassRatio(
            alpha=alpha, beta=beta, mmin=mmin, mmax=mmax, validate_args=validate_args
        )
        transformed_powerlaw = ExtendedSupportTransformedDistribution(
            base_distribution=powerlaw,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )
        powerlaws_collection.append(transformed_powerlaw)
    return powerlaws_collection


def create_powerlaw_redshift(
    N: int,
    parameter_name: Literal["redshift"],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Create a list of PowerlawRedshift distributions.

    Parameters
    ----------
    N : int
        Number of components
    parameter_name : Literal[&quot;redshift&quot;]
        name of the parameter to create distributions for
    component_type : Literal[&quot;pl&quot;, &quot;g&quot;]
        type of component, either "pl" or "g"
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, defaults to None, by default None

    Returns
    -------
    List[Distribution]
        list of PowerlawRedshift distributions

    Raises
    ------
    ValueError
        if :code:`kappa` or :code:`z_max` parameters are missing
    """
    powerlaw_redshift_collection = []
    kappa_name = f"{parameter_name}_kappa_{component_type}"
    z_max_name = f"{parameter_name}_z_max_{component_type}"

    for i in range(N):
        kappa = fetch_first_matching_value(params, f"{kappa_name}_{i}", kappa_name)
        if kappa is None:
            raise ValueError(f"Missing parameter {kappa_name}_{i}")

        z_max = fetch_first_matching_value(params, f"{z_max_name}_{i}", z_max_name)
        if z_max is None:
            raise ValueError(f"Missing parameter {z_max_name}_{i}")

        powerlaw_redshift_collection.append(
            PowerlawRedshift(kappa=kappa, z_max=z_max, validate_args=validate_args)
        )

    return powerlaw_redshift_collection


def create_uniform_distributions(
    N: int,
    parameter_name: Literal[
        "chi1",
        "chi2",
        "cos_iota",
        "cos_tilt1",
        "cos_tilt2",
        "dec",
        "detection_time",
        "ecc",
        "m1",
        "m2",
        "mean_anomaly",
        "phi_1",
        "phi_12",
        "phi_2",
        "phi_orb",
        "psi",
        "ra",
        "redshift",
    ],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Create a list of Uniform distributions.

    Parameters
    ----------
    N : int
        Number of components
    parameter_name : Literal[ &quot;chi1&quot;, &quot;chi2&quot;, &quot;cos_iota&quot;, &quot;cos_tilt1&quot;, &quot;cos_tilt2&quot;, &quot;dec&quot;, &quot;detection_time&quot;, &quot;ecc&quot;, &quot;m1&quot;, &quot;m2&quot;, &quot;mean_anomaly&quot;, &quot;phi_1&quot;, &quot;phi_12&quot;, &quot;phi_2&quot;, &quot;phi_orb&quot;, &quot;psi&quot;, &quot;ra&quot;, &quot;redshift&quot;, ]
        name of the parameter to create distributions for
    component_type : Literal[&quot;pl&quot;, &quot;g&quot;]
        type of component, either &quot;pl&quot; or &quot;g&quot;
    params : Dict[str, Array]
        dictionary of parameters
    validate_args : Optional[bool], optional
        whether to validate arguments, by default None

    Returns
    -------
    List[Distribution]
        list of Uniform distributions
    """
    uniform_collection = []
    low_name = f"{parameter_name}_low_{component_type}"
    high_name = f"{parameter_name}_high_{component_type}"
    for i in range(N):
        low = fetch_first_matching_value(params, f"{low_name}_{i}", low_name)
        if low is None:
            raise ValueError(f"Missing parameter {low_name}_{i}")

        high = fetch_first_matching_value(params, f"{high_name}_{i}", high_name)
        if high is None:
            raise ValueError(f"Missing parameter {high_name}_{i}")

        uniform_collection.append(
            Uniform(low=low, high=high, validate_args=validate_args)
        )

    return uniform_collection
