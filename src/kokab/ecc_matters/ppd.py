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


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd

from gwkokab.parameters import ECCENTRICITY, PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE
from gwkokab.utils.tools import error_if
from kokab.ecc_matters.common import EccentricityMattersModel
from kokab.utils import ppd, ppd_parser
from kokab.utils.common import ppd_ranges, read_json


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = ppd_parser.get_parser(parser)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    error_if(
        not str(args.filename).endswith(".hdf5"),
        msg="Output file must be an HDF5 file.",
    )

    constants = read_json(args.constants)
    nf_samples_mapping = read_json(args.nf_samples_mapping)

    parameters = [
        PRIMARY_MASS_SOURCE.name,
        SECONDARY_MASS_SOURCE.name,
        ECCENTRICITY.name,
    ]
    ranges = ppd_ranges(parameters, args.range)

    nf_samples = pd.read_csv(
        args.sample_filename, delimiter=" ", comment="#", header=None
    ).to_numpy()

    ppd.compute_and_save_ppd(
        EccentricityMattersModel,
        nf_samples,
        ranges,
        "rate_scaled_" + args.filename,
        parameters,
        constants,
        nf_samples_mapping,
    )

    nf_samples, constants = ppd.wipe_log_rate(nf_samples, nf_samples_mapping, constants)

    ppd.compute_and_save_ppd(
        EccentricityMattersModel,
        nf_samples,
        ranges,
        args.filename,
        parameters,
        constants,
        nf_samples_mapping,
    )
