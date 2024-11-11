# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from typing_extensions import Dict, List, Literal, Optional

from jax import numpy as jnp
from jaxtyping import Array, Bool, Int
from numpyro.distributions import (
    Beta,
    Distribution,
    Normal,
    TransformedDistribution,
    TruncatedNormal,
    TwoSidedTruncatedDistribution,
)

from ...cosmology import PLANCK_2015_Cosmology
from ...models._models import (
    PowerlawPrimaryMassRatio,
    SmoothedGaussianPrimaryMassRatio,
    SmoothedPowerlawPrimaryMassRatio,
)
from ...models.redshift import PowerlawRedshift
from ...models.transformations import PrimaryMassAndMassRatioToComponentMassesTransform
from ...utils.math import beta_dist_mean_variance_to_concentrations
from ...utils.tools import fetch_first_matching_value


__all__ = [
    "combine_distributions",
    "create_beta_distributions",
    "create_powerlaw_redshift",
    "create_powerlaws",
    "create_smoothed_gaussians",
    "create_smoothed_powerlaws",
    "create_truncated_normal_distributions_for_cos_tilt",
    "create_truncated_normal_distributions",
]


def combine_distributions(
    base_dists: List[List[Distribution]], add_dists: List[Distribution]
):
    """Helper function to combine base distributions with additional distributions
    like spin, tilt, or eccentricity."""
    return [dists + [add_dist] for dists, add_dist in zip(base_dists, add_dists)]


def create_beta_distributions(
    N: Int[int, ""],
    parameter_name: Literal[
        "m1", "m2", "chi1", "chi2", "cos_tilt1", "cos_tilt2", "ecc"
    ],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Bool[Optional[bool], "True", "False", "None"] = None,
) -> List[Beta]:
    r"""Create a list of Beta distributions.

    :param N: Number of components
    :param parameter_name: name of the parameter to create distributions for
    :param component_type: type of component, either "pl" or "g"
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :raises ValueError: if mean or variance is missing
    :return: list of Beta distributions
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
        concentrations = beta_dist_mean_variance_to_concentrations(
            mean=mean, variance=variance
        )
        beta_collection.append(Beta(*concentrations, validate_args=validate_args))
    return beta_collection


def create_truncated_normal_distributions(
    N: Int[int, ""],
    parameter_name: Literal[
        "m1", "m2", "chi1", "chi2", "cos_tilt1", "cos_tilt2", "ecc"
    ],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Bool[Optional[bool], "True", "False", "None"] = None,
) -> List[Distribution]:
    r"""Create a list of TruncatedNormal distributions.

    :param N: Number of components
    :param parameter_name: name of the parameter to create distributions for
    :param component_type: type of component, either "pl" or "g"
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :raises ValueError: if loc, scale, low, or high is missing
    :return: list of TruncatedNormal distributions
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
        high = fetch_first_matching_value(params, f"{high_name}_{i}", high_name)

        truncated_normal_collection.append(
            TruncatedNormal(
                loc=loc, scale=scale, low=low, high=high, validate_args=validate_args
            )
        )

    return truncated_normal_collection


def create_truncated_normal_distributions_for_cos_tilt(
    N: Int[int, ""],
    parameter_name: Literal["cos_tilt1", "cos_tilt2"],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Bool[Optional[bool], "True", "False", "None"] = None,
) -> List[TwoSidedTruncatedDistribution]:
    r"""Create a list of TwoSidedTruncatedDistribution distributions for tilt.

    :param N: Number of components
    :param parameter_name: name of the parameter to create distributions for
    :param component_type: type of component, either "pl" or "g"
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :raises ValueError: if scale is missing
    :return: list of TwoSidedTruncatedDistribution distributions
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


def create_powerlaws(
    N: Int[int, ""],
    params: Dict[str, Array],
    validate_args: Bool[Optional[bool], "True", "False", "None"] = None,
) -> List[TransformedDistribution]:
    r"""Create a list of TransformedDistribution for powerlaws.

    :param N: Number of components
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :raises ValueError: if alpha, beta, mmin, or mmax is missing
    :return: list of TransformedDistribution for powerlaws
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
        transformed_powerlaw = TransformedDistribution(
            base_distribution=powerlaw,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )
        powerlaws_collection.append(transformed_powerlaw)
    return powerlaws_collection


zgrid = jnp.linspace(0.001, 10, 1000)
dVcdz = 4.0 * jnp.pi * PLANCK_2015_Cosmology.dVcdz(zgrid)


def create_powerlaw_redshift(
    N: Int[int, ""],
    parameter_name: Literal["redshift"],
    component_type: Literal["pl", "g"],
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    r"""Create a list of PowerlawRedshift distributions.

    :param N: Number of components
    :param parameter_name: name of the parameter to create distributions for
    :param component_type: type of component, either "pl" or "g"
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :raises ValueError: if lamb or z_max parameters are missing
    :return: list of PowerlawRedshift distributions
    """
    powerlaw_redshift_collection = []
    lamb_name = f"{parameter_name}_lamb_{component_type}"
    z_max_name = f"{parameter_name}_z_max_{component_type}"

    for i in range(N):
        lamb = fetch_first_matching_value(params, f"{lamb_name}_{i}", lamb_name)
        if lamb is None:
            raise ValueError(f"Missing parameter {lamb_name}_{i}")

        z_max = fetch_first_matching_value(params, f"{z_max_name}_{i}", z_max_name)
        if z_max is None:
            raise ValueError(f"Missing parameter {z_max_name}_{i}")

        powerlaw_redshift_collection.append(
            PowerlawRedshift(
                lamb=lamb,
                z_max=z_max,
                zgrid=zgrid,
                dVcdz=dVcdz,
                validate_args=validate_args,
            )
        )

    return powerlaw_redshift_collection


def create_smoothed_powerlaws(
    N: Int[int, ""],
    params: Dict[str, Array],
    validate_args: Bool[Optional[bool], "True", "False", "None"] = None,
) -> List[SmoothedPowerlawPrimaryMassRatio]:
    r"""Create a list of SmoothedPowerlawPrimaryMassRatio for powerlaws.

    :param N: Number of components
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :raises ValueError: if alpha, beta, mmin, mmax or delta is missing
    :return: list of TransformedDistribution for powerlaws
    """
    smoothed_powerlaws_collection = []
    alpha_name = "alpha_pl"
    beta_name = "beta_pl"
    mmin_name = "mmin_pl"
    mmax_name = "mmax_pl"
    delta_name = "delta_pl"
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

        delta = fetch_first_matching_value(params, f"{delta_name}_{i}", delta_name)
        if delta is None:
            raise ValueError(f"Missing parameter {delta_name}_{i}")

        powerlaw = SmoothedPowerlawPrimaryMassRatio(
            alpha=alpha,
            beta=beta,
            mmin=mmin,
            mmax=mmax,
            delta=delta,
            validate_args=validate_args,
        )
        smoothed_powerlaws_collection.append(powerlaw)
    return smoothed_powerlaws_collection


def create_smoothed_gaussians(
    N: Int[int, ""],
    params: Dict[str, Array],
    validate_args: Bool[Optional[bool], "True", "False", "None"] = None,
) -> List[SmoothedPowerlawPrimaryMassRatio]:
    r"""Create a list of SmoothedPowerlawPrimaryMassRatio for powerlaws.

    :param N: Number of components
    :param params: dictionary of parameters
    :param validate_args: whether to validate arguments, defaults to None
    :raises ValueError: if loc, scale, beta, mmin, mmax or delta is missing
    :return: list of TransformedDistribution for powerlaws
    """
    smoothed_gaussians_collection = []
    loc_name = "loc_g"
    scale_name = "scale_g"
    beta_name = "beta_g"
    mmin_name = "mmin_g"
    delta_name = "delta_g"
    for i in range(N):
        loc = fetch_first_matching_value(params, f"{loc_name}_{i}", loc_name)
        if loc is None:
            raise ValueError(f"Missing parameter {loc_name}_{i}")

        scale = fetch_first_matching_value(params, f"{scale_name}_{i}", scale_name)
        if scale is None:
            raise ValueError(f"Missing parameter {scale_name}_{i}")

        beta = fetch_first_matching_value(params, f"{beta_name}_{i}", beta_name)
        if beta is None:
            raise ValueError(f"Missing parameter {beta_name}_{i}")

        mmin = fetch_first_matching_value(params, f"{mmin_name}_{i}", mmin_name)
        if mmin is None:
            raise ValueError(f"Missing parameter {mmin_name}_{i}")

        delta = fetch_first_matching_value(params, f"{delta_name}_{i}", delta_name)
        if delta is None:
            raise ValueError(f"Missing parameter {delta_name}_{i}")

        powerlaw = SmoothedGaussianPrimaryMassRatio(
            loc=loc,
            scale=scale,
            beta=beta,
            mmin=mmin,
            delta=delta,
            validate_args=validate_args,
        )
        smoothed_gaussians_collection.append(powerlaw)
    return smoothed_gaussians_collection
