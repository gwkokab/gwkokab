# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

import gwkokab
from gwkokab.models import NSmoothedPowerlawMSmoothedGaussian
from gwkokab.models.utils import (
    create_smoothed_gaussians_raw,
    create_smoothed_powerlaws_raw,
    create_truncated_normal_distributions,
)
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    ECCENTRICITY,
    MASS_RATIO,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)
from gwkokab.utils.tools import error_if
from kokab.utils import ppd, ppd_parser
from kokab.utils.common import ppd_ranges, read_json


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = ppd_parser.get_parser(parser)

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--spin-truncated-normal",
        action="store_true",
        help="Use truncated normal distributions for spin parameters.",
    )
    model_group.add_argument(
        "--raw",
        action="store_true",
        help="The raw parameters for this model are primary mass and mass ratio. To"
        "align with the rest of the codebase, we transform primary mass and mass ratio"
        "to primary and secondary mass. This flag will use the raw parameters i.e."
        "primary mass and mass ratio.",
    )

    msm_group = parser.add_argument_group("Multi Source Model Options")
    msm_group.add_argument(
        "--per-component",
        action="store_true",
        help="Compute the PPD for each component of the multi source model.",
    )

    return parser


def compute_per_component_ppd(
    nf_samples,
    ranges,
    constants,
    component_prefix,
    parameters,
    args,
    nf_samples_mapping,
    N_pl,
    N_g,
):
    for n_pl in range(1, N_pl + 1):
        constants_copy = constants.copy()
        constants_copy["N_pl"] = n_pl
        constants_copy["N_g"] = 0
        ppd.compute_and_save_ppd(
            NSmoothedPowerlawMSmoothedGaussian,
            nf_samples,
            ranges,
            f"{component_prefix}powerlaw_{n_pl}_" + args.filename,
            parameters,
            constants_copy,
            nf_samples_mapping,
            args.batch_size,
        )

    for n_g in range(1, N_g + 1):
        constants_copy = constants.copy()
        constants_copy["N_pl"] = 0
        constants_copy["N_g"] = n_g
        ppd.compute_and_save_ppd(
            NSmoothedPowerlawMSmoothedGaussian,
            nf_samples,
            ranges,
            f"{component_prefix}gaussian_{n_g}_" + args.filename,
            parameters,
            constants_copy,
            nf_samples_mapping,
            args.batch_size,
        )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.spin_truncated_normal:
        gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_spin_distributions = create_truncated_normal_distributions

    error_if(
        not str(args.filename).endswith(".hdf5"),
        msg="Output file must be an HDF5 file.",
    )

    constants = read_json(args.constants)
    nf_samples_mapping = read_json(args.nf_samples_mapping)

    N_pl = constants["N_pl"]
    N_g = constants["N_g"]
    has_spin = constants.get("use_spin", False)
    has_tilt = constants.get("use_tilt", False)
    has_eccentricity = constants.get("use_eccentricity", False)

    parameters = [PRIMARY_MASS_SOURCE.name]
    if args.raw:
        gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_powerlaw_distributions = create_smoothed_powerlaws_raw
        gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_gaussian_distributions = create_smoothed_gaussians_raw
        parameters.append(MASS_RATIO.name)
    else:
        parameters.append(SECONDARY_MASS_SOURCE.name)
    if has_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE.name, SECONDARY_SPIN_MAGNITUDE.name])
    if has_tilt:
        parameters.extend([COS_TILT_1.name, COS_TILT_2.name])
    if has_eccentricity:
        parameters.append(ECCENTRICITY.name)

    ranges = ppd_ranges(parameters, args.range)

    nf_samples = pd.read_csv(
        args.sample_filename, delimiter=" ", comment="#", header=None
    ).to_numpy()

    ppd.compute_and_save_ppd(
        NSmoothedPowerlawMSmoothedGaussian,
        nf_samples,
        ranges,
        "rate_scaled_" + args.filename,
        parameters,
        constants,
        nf_samples_mapping,
        args.batch_size,
    )

    if args.per_component:
        compute_per_component_ppd(
            nf_samples,
            ranges,
            constants,
            "rate_scaled_",
            parameters,
            args,
            nf_samples_mapping,
            N_pl,
            N_g,
        )

    nf_samples, constants = ppd.wipe_log_rate(nf_samples, nf_samples_mapping, constants)

    ppd.compute_and_save_ppd(
        NSmoothedPowerlawMSmoothedGaussian,
        nf_samples,
        ranges,
        args.filename,
        parameters,
        constants,
        nf_samples_mapping,
        args.batch_size,
    )

    if args.per_component:
        compute_per_component_ppd(
            nf_samples,
            ranges,
            constants,
            "",
            parameters,
            args,
            nf_samples_mapping,
            N_pl,
            N_g,
        )
