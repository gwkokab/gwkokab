# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd
from numpyro.distributions.distribution import DistributionLike

from gwkokab.models import SmoothedPowerlawPeakAndPowerlawRedshift
from gwkokab.parameters import PRIMARY_MASS_SOURCE, REDSHIFT, SECONDARY_MASS_SOURCE
from gwkokab.utils.tools import error_if
from kokab.utils import ppd, ppd_parser
from kokab.utils.common import ppd_ranges, read_json


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


def model(**params) -> DistributionLike:
    _model = SmoothedPowerlawPeakAndPowerlawRedshift(**params)
    _model._component_distributions[0].marginal_distributions[0] = (
        _model._component_distributions[0].marginal_distributions[0].base_dist
    )
    _model._component_distributions[1].marginal_distributions[0] = (
        _model._component_distributions[1].marginal_distributions[0].base_dist
    )
    return _model


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    error_if(
        not str(args.filename).endswith(".hdf5"),
        msg="Output file must be an HDF5 file.",
    )

    constants = read_json(args.constants)
    nf_samples_mapping = read_json(args.nf_samples_mapping)

    parameters = [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name, REDSHIFT.name]

    ranges = ppd_ranges(parameters, args.range)

    nf_samples = pd.read_csv(
        args.sample_filename, delimiter=" ", comment="#", header=None
    ).to_numpy()

    ppd.compute_and_save_ppd(
        model,
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
        model,
        nf_samples,
        ranges,
        args.filename,
        parameters,
        constants,
        nf_samples_mapping,
        args.batch_size,
    )
