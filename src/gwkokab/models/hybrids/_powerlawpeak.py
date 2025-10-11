# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import HalfNormal, Independent

from ..mass import SmoothedTwoComponentPrimaryMassRatio
from ..redshift import PowerlawRedshift
from ..spin import BetaFromMeanVar, IndependentSpinOrientationGaussianIsotropic
from ..transformations import PrimaryMassAndMassRatioToComponentMassesTransform
from ..utils import (
    ExtendedSupportTransformedDistribution,
    JointDistribution,
    ScaledMixture,
)


def PowerlawPeak(
    use_spin: bool = False,
    use_redshift: bool = False,
    use_tilt: bool = False,
    use_eccentricity: bool = False,
    validate_args: Optional[bool] = None,
    **params: Array,
) -> ScaledMixture:
    # NOTE: If you change something here, please also change in
    # kokab/one_powerlaw_one_peak/ppd.py
    smoothing_model = ExtendedSupportTransformedDistribution(
        SmoothedTwoComponentPrimaryMassRatio(
            alpha=params["alpha"],
            beta=params["beta"],
            delta=params["delta"],
            lambda_peak=params["lambda_peak"],
            loc=params["loc"],
            mmax=params["mmax"],
            mmin=params["mmin"],
            scale=params["scale"],
            validate_args=validate_args,
        ),
        transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
        validate_args=validate_args,
    )

    component_distributions = [smoothing_model]

    if use_spin:
        chi_dist = Independent(
            BetaFromMeanVar(
                mean=jnp.stack([params["chi_mean"], params["chi_mean"]], axis=-1),
                variance=jnp.stack(
                    [params["chi_variance"], params["chi_variance"]], axis=-1
                ),
                validate_args=validate_args,
            ),
            reinterpreted_batch_ndims=1,
            validate_args=validate_args,
        )
        component_distributions.append(chi_dist)

    if use_tilt:
        tilt_dist = IndependentSpinOrientationGaussianIsotropic(
            zeta=params["cos_tilt_zeta"],
            scale1=params["cos_tilt_scale"],
            scale2=params["cos_tilt_scale"],
            validate_args=validate_args,
        )

        component_distributions.append(tilt_dist)

    if use_eccentricity:
        ecc_dist = HalfNormal(
            scale=params["eccentricity_scale"],
            validate_args=validate_args,
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
