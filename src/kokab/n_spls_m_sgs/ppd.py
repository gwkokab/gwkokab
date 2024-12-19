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


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing_extensions import Callable, Dict, List, Tuple, Union

import pandas as pd
from jaxtyping import Array

import gwkokab
from gwkokab.models import NSmoothedPowerlawMSmoothedGaussian
from gwkokab.models.utils import create_truncated_normal_distributions
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

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--spin-truncated-normal",
        action="store_true",
        help="Use truncated normal distributions for spin parameters.",
    )

    return parser


def load_configuration(
    constants_file: str, nf_samples_mapping_file: str
) -> Tuple[
    Dict[str, Union[int, float, bool]],
    Dict[str, int],
]:
    try:
        with open(constants_file, "r") as f:
            constants: Dict[str, Union[int, float, bool]] = json.load(f)
        with open(nf_samples_mapping_file, "r") as f:
            nf_samples_mapping: Dict[str, int] = json.load(f)
        return constants, nf_samples_mapping
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_pdf(
    constants: Dict[str, Union[int, float, bool]],
    nf_samples_mapping: Dict[str, int],
    rate_scaled: bool = False,
) -> Callable[[Array], Array]:
    nf_samples = pd.read_csv(
        "sampler_data/nf_samples.dat", delimiter=" ", skiprows=1
    ).to_numpy()

    if not rate_scaled:
        model = NSmoothedPowerlawMSmoothedGaussian(
            **constants,
            **{
                name: (nf_samples[..., i] if not name.startswith("log_rate") else 0.0)
                for name, i in nf_samples_mapping.items()
            },
        )
    else:
        model = NSmoothedPowerlawMSmoothedGaussian(
            **constants,
            **{name: nf_samples[..., i] for name, i in nf_samples_mapping.items()},
        )

    return model.log_prob


def compute_and_save_ppd(
    logpdf: Callable[[Array], Array],
    domains: List[Tuple[float, float, int]],
    output_file: str,
    parameters: List[str],
) -> None:
    prob_values = ppd.compute_probs(logpdf, domains)
    ppd_values = ppd.get_ppd(prob_values, axis=-1)
    marginals = ppd.get_all_marginals(prob_values, domains)
    ppd.save_probs(ppd_values, marginals, output_file, domains, parameters)


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.spin_truncated_normal:
        gwkokab.models.npowerlawmgaussian._model.build_spin_distributions = (
            create_truncated_normal_distributions
        )

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

    model_without_rate_pdf = get_model_pdf(constants, nf_samples_mapping)
    compute_and_save_ppd(model_without_rate_pdf, args.range, args.filename, parameters)

    model_with_rate_pdf = get_model_pdf(constants, nf_samples_mapping, rate_scaled=True)
    compute_and_save_ppd(
        model_with_rate_pdf, args.range, "rate_scaled_" + args.filename, parameters
    )
