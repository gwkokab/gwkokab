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
from functools import partial
from typing_extensions import List, Tuple

import jax.numpy as jnp
from jaxtyping import Int
from numpyro import distributions as dist

from gwkokab.errors import banana_error_m1_m2
from gwkokab.models import NPowerLawMGaussian_TruncatedNormalEccentricity
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    ECCENTRICITY,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)
from gwkokab.population import error_magazine, popfactory, popmodel_magazine

from ..utils import genie_parser
from ..utils.common import expand_arguments
from ..utils.regex import match_all
from .common import constraint, get_logVT


m1_source_name = PRIMARY_MASS_SOURCE.name
m2_source_name = SECONDARY_MASS_SOURCE.name
chi1_name = PRIMARY_SPIN_MAGNITUDE.name
chi2_name = SECONDARY_SPIN_MAGNITUDE.name
cos_tilt_1_name = COS_TILT_1.name
cos_tilt_2_name = COS_TILT_2.name
ecc_name = ECCENTRICITY.name


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = genie_parser.get_parser(parser)

    model_group = parser.add_argument_group("Model Options")

    model_group.add_argument(
        "--model-json",
        help="Path to the JSON file containing the model parameters",
        type=str,
        required=True,
    )
    model_group.add_argument(
        "--no-spin",
        action="store_true",
        help="Do not include spin parameters in the model.",
    )
    model_group.add_argument(
        "--no-tilt",
        action="store_true",
        help="Do not include tilt parameters in the model.",
    )
    model_group.add_argument(
        "--no-eccentricity",
        action="store_true",
        help="Do not include eccentricity parameters in the model.",
    )

    err_group = parser.add_argument_group("Error Options")
    err_group.add_argument(
        "--err-json",
        help="Path to the JSON file containing the error parameters",
        type=str,
        required=True,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    with open(args.model_json, "r") as f:
        model_json = json.load(f)

    with open(args.err_json, "r") as f:
        err_json = json.load(f)

    N_pl = model_json["N_pl"]
    N_g = model_json["N_g"]

    has_spin = not args.no_spin
    has_tilt = not args.no_tilt
    has_eccentricity = not args.no_eccentricity

    all_params: List[Tuple[str, Int[int, "N_pl", "N_g"]]] = [
        ("alpha", N_pl),
        ("beta", N_pl),
        ("loc_m1", N_g),
        ("loc_m2", N_g),
        ("mmax", N_pl),
        ("mmin", N_pl),
        ("scale_m1", N_g),
        ("scale_m2", N_g),
    ]

    parameters_name = (m1_source_name, m2_source_name)
    if has_spin:
        parameters_name += (chi1_name, chi2_name)
        all_params.extend(
            [
                ("mean_chi1_g", N_g),
                ("mean_chi1_pl", N_pl),
                ("mean_chi2_g", N_g),
                ("mean_chi2_pl", N_pl),
                ("variance_chi1_g", N_g),
                ("variance_chi1_pl", N_pl),
                ("variance_chi2_g", N_g),
                ("variance_chi2_pl", N_pl),
            ]
        )
    if has_tilt:
        parameters_name += (cos_tilt_1_name, cos_tilt_2_name)
        all_params.extend(
            [
                ("std_dev_tilt1_g", N_g),
                ("std_dev_tilt1_pl", N_pl),
                ("std_dev_tilt2_g", N_g),
                ("std_dev_tilt2_pl", N_pl),
            ]
        )
    if has_eccentricity:
        parameters_name += (ecc_name,)
        all_params.extend([("scale_ecc_g", N_g), ("scale_ecc_pl", N_pl)])

    extended_params = []
    for params in all_params:
        extended_params.extend(expand_arguments(*params))

    model_param = match_all(extended_params, model_json)

    popmodel_magazine.register(
        parameters_name,
        NPowerLawMGaussian_TruncatedNormalEccentricity(
            N_pl=N_pl,
            N_g=N_g,
            use_spin=has_spin,
            use_tilt=has_tilt,
            use_eccentricity=has_eccentricity,
            **model_param,
        ),
    )

    err_param = match_all(
        [
            "chi1_high",
            "chi1_loc",
            "chi1_low",
            "chi1_scale",
            "chi2_high",
            "chi2_loc",
            "chi2_low",
            "chi2_scale",
            "cos_tilt_1_high",
            "cos_tilt_1_loc",
            "cos_tilt_1_low",
            "cos_tilt_1_scale",
            "cos_tilt_2_high",
            "cos_tilt_2_loc",
            "cos_tilt_2_low",
            "cos_tilt_2_scale",
            "eccentricity_err_high",
            "eccentricity_err_loc",
            "eccentricity_err_low",
            "eccentricity_err_scale",
            "scale_eta",
            "scale_Mc",
        ],
        err_json,
    )

    error_magazine.register(
        (m1_source_name, m2_source_name),
        lambda x, size, key: banana_error_m1_m2(
            x,
            size,
            key,
            scale_Mc=err_param["scale_Mc"],
            scale_eta=err_param["scale_eta"],
        ),
    )

    if has_spin:

        @error_magazine.register(chi1_name)
        def chi1_error_fn(x, size, key):
            err_x = x + dist.TruncatedNormal(
                loc=err_param["chi1_loc"],
                scale=err_param["chi1_scale"],
                low=err_param["chi1_low"],
                high=err_param["chi1_high"],
            ).sample(key=key, sample_shape=(size,))
            return err_x

        @error_magazine.register(chi2_name)
        def chi2_error_fn(x, size, key):
            err_x = x + dist.TruncatedNormal(
                loc=err_param["chi2_loc"],
                scale=err_param["chi2_scale"],
                low=err_param["chi2_low"],
                high=err_param["chi2_high"],
            ).sample(key=key, sample_shape=(size,))
            return err_x

    if has_tilt:

        @error_magazine.register(cos_tilt_1_name)
        def cos_tilt_1_error_fn(x, size, key):
            err_x = x + dist.TruncatedNormal(
                loc=err_param["cos_tilt_1_loc"],
                scale=err_param["cos_tilt_1_scale"],
                low=err_param["cos_tilt_1_low"],
                high=err_param["cos_tilt_1_high"],
            ).sample(key=key, sample_shape=(size,))
            return err_x

        @error_magazine.register(cos_tilt_2_name)
        def cos_tilt_2_error_fn(x, size, key):
            err_x = x + dist.TruncatedNormal(
                loc=err_param["cos_tilt_2_loc"],
                scale=err_param["cos_tilt_2_scale"],
                low=err_param["cos_tilt_2_low"],
                high=err_param["cos_tilt_2_high"],
            ).sample(key=key, sample_shape=(size,))
            return err_x

    if has_eccentricity:

        @error_magazine.register(ecc_name)
        def ecc_error_fn(x, size, key):
            err_x = x + dist.TruncatedNormal(
                loc=err_param["eccentricity_err_loc"],
                scale=err_param["eccentricity_err_scale"],
                low=err_param["eccentricity_err_low"],
                high=err_param["eccentricity_err_high"],
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < 0.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

    popfactory.analysis_time = args.analysis_time
    popfactory.constraint = partial(
        constraint,
        has_spin=has_spin,
        has_tilt=has_tilt,
        has_eccentricity=has_eccentricity,
    )
    popfactory.error_size = args.error_size
    popfactory.log_VT_fn = get_logVT(args.vt_path)
    popfactory.num_realizations = args.num_realizations
    popfactory.rate = args.rate
    popfactory.VT_params = [m1_source_name, m2_source_name]

    popfactory.produce()
