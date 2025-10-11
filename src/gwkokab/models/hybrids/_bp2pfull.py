# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import HalfNormal, Independent, TruncatedNormal

from ..mass import BrokenPowerlawTwoPeak
from ..redshift import PowerlawRedshift
from ..spin import MinimumTiltModel
from ..transformations import PrimaryMassAndMassRatioToComponentMassesTransform
from ..utils import (
    ExtendedSupportTransformedDistribution,
    JointDistribution,
    ScaledMixture,
)


def BrokenPowerlawTwoPeakFull(
    use_spin: bool = False,
    use_redshift: bool = False,
    use_tilt: bool = False,
    use_eccentricity: bool = False,
    validate_args: Optional[bool] = None,
    **params: Array,
) -> ScaledMixture:
    # NOTE: If you change something here, please also change in
    # kokab/bp2pfull/ppd.py
    smoothing_model = ExtendedSupportTransformedDistribution(
        BrokenPowerlawTwoPeak(
            alpha1=params["alpha1"],
            alpha2=params["alpha2"],
            beta=params["beta"],
            loc1=params["loc1"],
            loc2=params["loc2"],
            scale1=params["scale1"],
            scale2=params["scale2"],
            delta_m1=params["delta_m1"],
            delta_m2=params["delta_m2"],
            lambda_0=params["lambda_0"],
            lambda_1=params["lambda_1"],
            m1min=params["m1min"],
            m2min=params["m2min"],
            mmax=params["mmax"],
            mbreak=params["mbreak"],
            validate_args=validate_args,
        ),
        transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
        validate_args=validate_args,
    )

    component_distributions = [smoothing_model]

    if use_spin:
        chi_dist = Independent(
            TruncatedNormal(
                loc=jnp.stack([params["chi_loc"], params["chi_loc"]], axis=-1),
                scale=jnp.stack([params["chi_scale"], params["chi_scale"]], axis=-1),
                low=0.0,
                high=1.0,
                validate_args=validate_args,
            ),
            reinterpreted_batch_ndims=1,
            validate_args=validate_args,
        )
        component_distributions.append(chi_dist)

    if use_tilt:
        tilt_dist = MinimumTiltModel(
            zeta=params["cos_tilt_zeta"],
            loc=params["cos_tilt_loc"],
            scale=params["cos_tilt_scale"],
            minimum=params.get("cos_tilt_minimum", -1.0),
            validate_args=validate_args,
        )
        component_distributions.append(tilt_dist)

    if use_eccentricity:
        ecc_dist = HalfNormal(
            scale=params["eccentricity_scale"], validate_args=validate_args
        )
        component_distributions.append(ecc_dist)

    if use_redshift:
        z_max = params["z_max"]
        kappa = params["kappa"]
        powerlaw_z = PowerlawRedshift(
            z_max=z_max, kappa=kappa, validate_args=validate_args
        )
        component_distributions.append(powerlaw_z)

    if len(component_distributions) > 1:
        component_distributions = [
            JointDistribution(*component_distributions, validate_args=validate_args)
        ]

    return ScaledMixture(
        log_scales=jnp.asarray([params["log_rate"]]),
        component_distributions=component_distributions,
        support=component_distributions[0]._support,  # type: ignore
        validate_args=validate_args,
    )
