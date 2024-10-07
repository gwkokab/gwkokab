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

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing_extensions import Dict

import pandas as pd

from gwkokab.models import NPowerLawMGaussian
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)

from ..utils import ppd, ppd_parser


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = ppd_parser.get_parser(parser)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    with open(args.constants, "r") as f:
        constants: Dict[str, int | float | bool] = json.loads(f)

    with open(args.nf_samples_mapping, "r") as f:
        nf_samples_mapping: Dict[str, int] = json.loads(f)

    has_spin = constants["use_spin"]
    has_tilt = constants["use_tilt"]

    parameters = [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name]
    if has_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE.name, SECONDARY_SPIN_MAGNITUDE.name])
    if has_tilt:
        parameters.extend([COS_TILT_1.name, COS_TILT_2.name])

    nf_samples = pd.read_csv(
        "sampler_data/nf_samples.dat", delimiter=" ", skiprows=1
    ).to_numpy()

    model = NPowerLawMGaussian(
        **constants,
        **{name: nf_samples[..., i] for name, i in nf_samples_mapping.items()},
    )

    ppd_ranges = args.range

    ppd_values = ppd.compute_ppd(model.log_prob, ppd_ranges)
    ppd.save_ppd(
        ppd_values,
        args.filename,
        ppd_ranges,
        [parameter.name for parameter in parameters],
    )
