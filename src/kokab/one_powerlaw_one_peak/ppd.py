# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Optional

import pandas as pd
from jax import numpy as jnp
from jaxtyping import Array
from numpyro._typing import DistributionLike

from gwkokab.models import (
    SmoothedGaussianPrimaryMassRatio,
    SmoothedPowerlawAndPeak,
    SmoothedPowerlawPrimaryMassRatio,
)
from gwkokab.models.constraints import any_constraint
from gwkokab.models.redshift import PowerlawRedshift
from gwkokab.models.spin import (
    BetaFromMeanVar,
    IndependentSpinOrientationGaussianIsotropic,
)
from gwkokab.models.utils import (
    JointDistribution,
    ScaledMixture,
)
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    MASS_RATIO,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    REDSHIFT,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)
from gwkokab.utils.tools import error_if
from kokab.utils import ppd, ppd_parser
from kokab.utils.common import ppd_ranges, read_json


def SmoothedPowerlawAndPeak_raw(
    use_spin: bool = False,
    use_redshift: bool = False,
    use_tilt: bool = False,
    validate_args: Optional[bool] = None,
    **params: Array,
) -> ScaledMixture:
    smoothed_powerlaw = SmoothedPowerlawPrimaryMassRatio(
        alpha=params["alpha"],
        beta=params["beta"],
        mmin=params["mmin"],
        mmax=params["mmax"],
        delta=params["delta"],
        validate_args=validate_args,
    )
    smoothed_gaussian = SmoothedGaussianPrimaryMassRatio(
        loc=params["loc"],
        scale=params["scale"],
        beta=params["beta"],
        mmin=params["mmin"],
        mmax=params["mmax"],
        delta=params["delta"],
        validate_args=validate_args,
    )

    component_distribution_pl = [smoothed_powerlaw]
    component_distribution_g = [smoothed_gaussian]

    if use_spin:
        chi1_dist_pl = BetaFromMeanVar(
            mean=params["chi1_mean_pl"],
            variance=params["chi1_variance_pl"],
            validate_args=validate_args,
        )

        chi2_dist_pl = BetaFromMeanVar(
            mean=params["chi2_mean_pl"],
            variance=params["chi2_variance_pl"],
            validate_args=validate_args,
        )

        chi1_dist_g = BetaFromMeanVar(
            mean=params["chi1_mean_g"],
            variance=params["chi1_variance_g"],
            validate_args=validate_args,
        )

        chi2_dist_g = BetaFromMeanVar(
            mean=params["chi2_mean_g"],
            variance=params["chi2_variance_g"],
            validate_args=validate_args,
        )

        component_distribution_pl.extend([chi1_dist_pl, chi2_dist_pl])
        component_distribution_g.extend([chi1_dist_g, chi2_dist_g])

    if use_tilt:
        tilt1_dist_pl = IndependentSpinOrientationGaussianIsotropic(
            zeta=params["cos_tilt_zeta_pl"],
            scale1=params["cos_tilt1_scale_pl"],
            scale2=params["cos_tilt2_scale_pl"],
            validate_args=validate_args,
        )
        tilt1_dist_g = IndependentSpinOrientationGaussianIsotropic(
            zeta=params["cos_tilt_zeta_g"],
            scale1=params["cos_tilt1_scale_g"],
            scale2=params["cos_tilt2_scale_g"],
            validate_args=validate_args,
        )

        component_distribution_pl.append(tilt1_dist_pl)
        component_distribution_g.append(tilt1_dist_g)

    if use_redshift:
        z_max = params["z_max"]
        kappa = params["kappa"]
        powerlaw_z = PowerlawRedshift(
            z_max=z_max, kappa=kappa, validate_args=validate_args
        )

        component_distribution_pl.append(powerlaw_z)
        component_distribution_g.append(powerlaw_z)

    if len(component_distribution_pl) == 1:
        component_distribution_pl = component_distribution_pl[0]
    else:
        component_distribution_pl = JointDistribution(
            *component_distribution_pl, validate_args=validate_args
        )

    if len(component_distribution_g) == 1:
        component_distribution_g = component_distribution_g[0]
    else:
        component_distribution_g = JointDistribution(
            *component_distribution_g, validate_args=validate_args
        )
    return ScaledMixture(
        log_scales=jnp.stack(
            [
                params["log_rate"] + jnp.log1p(-params["lambda_peak"]),  # type: ignore[arg-type, operator]
                params["log_rate"] + jnp.log(params["lambda_peak"]),  # type: ignore[arg-type]
            ],
            axis=-1,
        ),
        component_distributions=[component_distribution_pl, component_distribution_g],
        support=any_constraint(
            [
                component_distribution_pl.support,  # type: ignore[attr-defined]
                component_distribution_g.support,  # type: ignore[attr-defined]
            ]
        ),
        validate_args=validate_args,
    )


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = ppd_parser.get_parser(parser)

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--raw",
        action="store_true",
        help="The raw parameters for this model are primary mass and mass ratio. To"
        "align with the rest of the codebase, we transform primary mass and mass ratio"
        "to primary and secondary mass. This flag will use the raw parameters i.e."
        "primary mass and mass ratio.",
    )
    return parser


def model(raw: bool, **params) -> DistributionLike:
    validate_args = params.pop("validate_args", True)
    _model = SmoothedPowerlawAndPeak_raw if raw else SmoothedPowerlawAndPeak
    return _model(**params, validate_args=validate_args)


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    error_if(
        not str(args.filename).endswith(".hdf5"),
        msg="Output file must be an HDF5 file.",
    )

    constants = read_json(args.constants)
    nf_samples_mapping = read_json(args.nf_samples_mapping)

    use_spin = constants.get("use_spin", False)
    use_tilt = constants.get("use_tilt", False)
    use_redshift = constants.get("use_redshift", False)

    parameters = [PRIMARY_MASS_SOURCE.name]
    if args.raw:
        parameters.append(MASS_RATIO.name)
    else:
        parameters.append(SECONDARY_MASS_SOURCE.name)

    if use_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE.name, SECONDARY_SPIN_MAGNITUDE.name])
    if use_tilt:
        parameters.extend([COS_TILT_1.name, COS_TILT_2.name])
    if use_redshift:
        parameters.append(REDSHIFT.name)

    ranges = ppd_ranges(parameters, args.range)

    nf_samples = pd.read_csv(
        args.sample_filename, delimiter=" ", comment="#", header=None
    ).to_numpy()

    ppd.compute_and_save_ppd(
        ft.partial(model, raw=args.raw),
        nf_samples,
        ranges,
        "rate_scaled_" + args.filename,
        parameters,
        constants,
        nf_samples_mapping,
        args.batch_size,
    )

    nf_samples, constants = ppd.wipe_log_rate(nf_samples, nf_samples_mapping, constants)

    ppd.compute_and_save_ppd(
        ft.partial(model, raw=args.raw),
        nf_samples,
        ranges,
        args.filename,
        parameters,
        constants,
        nf_samples_mapping,
        args.batch_size,
    )
