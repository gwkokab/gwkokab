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
from numpyro.distributions.distribution import DistributionLike

from gwkokab.models import SmoothedPowerlawPeakAndPowerlawRedshift
from gwkokab.parameters import PRIMARY_MASS_SOURCE, REDSHIFT, SECONDARY_MASS_SOURCE
from gwkokab.utils.tools import error_if
from kokab.utils import ppd, ppd_parser
from kokab.utils.common import read_json


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
    validate_args = params.pop("validate_args", True)
    _model = SmoothedPowerlawPeakAndPowerlawRedshift(
        **params, validate_args=validate_args
    )
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
        "Output file must be an HDF5 file.",
    )

    constants = read_json(args.constants)
    nf_samples_mapping = read_json(args.nf_samples_mapping)

    parameters = [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name, REDSHIFT.name]

    nf_samples = pd.read_csv("sampler_data/nf_samples.dat", delimiter=" ").to_numpy()

    ppd.compute_and_save_ppd(
        model,
        nf_samples,
        args.range,
        "rate_scaled_" + args.filename,
        parameters,
        constants,
        nf_samples_mapping,
    )

    nf_samples, constants = ppd.wipe_log_rate(nf_samples, nf_samples_mapping, constants)

    ppd.compute_and_save_ppd(
        model,
        nf_samples,
        args.range,
        args.filename,
        parameters,
        constants,
        nf_samples_mapping,
    )
