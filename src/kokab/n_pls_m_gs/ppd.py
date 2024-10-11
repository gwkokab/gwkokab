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
from typing_extensions import Dict, List, Tuple, Union

import pandas as pd
from jaxtyping import Bool, Float, Int
from numpyro.distributions import MixtureGeneral

from gwkokab.models import NPowerLawMGaussian
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    ECCENTRICITY,
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


def load_configuration(
    constants_file: str, nf_samples_mapping_file: str
) -> Tuple[
    Dict[str, Union[Int[int, ""], Float[float, ""], Bool[bool, ""]]],
    Dict[str, Int[int, ""]],
]:
    try:
        with open(constants_file, "r") as f:
            constants: Dict[str, Union[int, float, bool]] = json.load(f)
        with open(nf_samples_mapping_file, "r") as f:
            nf_samples_mapping: Dict[str, int] = json.load(f)
        return constants, nf_samples_mapping
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading configuration: {e}")


def setup_model(
    constants: Dict[str, Union[Int[int, ""], Float[float, ""], Bool[bool, ""]]],
    nf_samples_mapping: Dict[str, Int[int, ""]],
) -> MixtureGeneral:
    nf_samples = pd.read_csv(
        "sampler_data/nf_samples.dat", delimiter=" ", skiprows=1
    ).to_numpy()
    return NPowerLawMGaussian(
        **constants,
        **{name: nf_samples[..., i] for name, i in nf_samples_mapping.items()},
    )


def compute_and_save_ppd(
    model: MixtureGeneral,
    domains: List[Tuple[Float[float, ""], Float[float, ""], Int[int, ""]]],
    output_file: str,
    parameters: List[str],
) -> None:
    prob_values = ppd.compute_probs(model.log_prob, domains)
    ppd_values = ppd.get_ppd(prob_values, axis=-1)
    marginals = ppd.get_all_marginals(prob_values, domains)
    ppd.save_probs(ppd_values, marginals, output_file, domains, parameters)


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if not str(args.filename).endswith(".hdf5"):
        raise ValueError("Output file must be an HDF5 file.")

    constants, nf_samples_mapping = load_configuration(
        args.constants, args.nf_samples_mapping
    )

    has_spin = constants.get("use_spin", False)
    has_tilt = constants.get("use_tilt", False)
    has_eccentricity = constants.get("use_eccentricity", False)

    parameters = [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name]
    if has_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE.name, SECONDARY_SPIN_MAGNITUDE.name])
    if has_tilt:
        parameters.extend([COS_TILT_1.name, COS_TILT_2.name])
    if has_eccentricity:
        parameters.append(ECCENTRICITY.name)

    model = setup_model(constants, nf_samples_mapping)
    compute_and_save_ppd(model, args.range, args.filename, parameters)
