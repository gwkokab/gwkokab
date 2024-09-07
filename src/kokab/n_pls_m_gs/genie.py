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

from gwkokab.errors import banana_error_m1_m2
from gwkokab.models import NPowerLawMGaussianWithDefaultSpinMagnitudeAndSpinMisalignment
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)
from gwkokab.population import error_magazine, popfactory, popmodel_magazine
from gwkokab.vts.neuralvt import load_model
from jax import vmap
from jaxtyping import Array, Bool
from numpyro import distributions as dist

from ..utils import genie_parser
from ..utils.regex import match_all


m1_source_name = PRIMARY_MASS_SOURCE.name
m2_source_name = SECONDARY_MASS_SOURCE.name
chi1_name = PRIMARY_SPIN_MAGNITUDE.name
chi2_name = SECONDARY_SPIN_MAGNITUDE.name
cos_tilt_1_name = COS_TILT_1.name
cos_tilt_2_name = COS_TILT_2.name


def get_logVT(vt_path):
    _, logVT = load_model(vt_path)

    def m1m2_trimmed_logVT(x: Array) -> Array:
        m1m2 = x[..., 0:2]
        return vmap(logVT)(m1m2)

    return m1m2_trimmed_logVT


def constraint(x: Array) -> Bool:
    m1 = x[..., 0]
    m2 = x[..., 1]
    chi1 = x[..., 2]
    chi2 = x[..., 3]
    cos_tilt_1 = x[..., 4]
    cos_tilt_2 = x[..., 5]

    mask = m1 > 0.0
    mask &= m2 > 0.0
    mask &= m1 >= m2

    mask &= chi1 >= 0.0
    mask &= chi1 <= 1.0

    mask &= chi2 >= 0.0
    mask &= chi2 <= 1.0

    mask &= cos_tilt_1 >= -1.0
    mask &= cos_tilt_1 <= 1.0

    mask &= cos_tilt_2 >= -1.0
    mask &= cos_tilt_2 <= 1.0

    return mask


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

    extend_args = lambda arg, n: list(map(lambda i: arg + f"_{i}", range(n)))

    model_param = match_all(
        extend_args("alpha", N_pl)
        + extend_args("beta", N_pl)
        + extend_args("mmin", N_pl)
        + extend_args("mmax", N_pl)
        + extend_args("mean_chi1_pl", N_pl)
        + extend_args("mean_chi2_pl", N_pl)
        + extend_args("std_dev_tilt1_pl", N_pl)
        + extend_args("std_dev_tilt2_pl", N_pl)
        + extend_args("variance_chi1_pl", N_pl)
        + extend_args("variance_chi2_pl", N_pl)
        + extend_args("loc_m1", N_g)
        + extend_args("loc_m2", N_g)
        + extend_args("scale_m1", N_g)
        + extend_args("scale_m2", N_g)
        + extend_args("mean_chi1_g", N_g)
        + extend_args("mean_chi2_g", N_g)
        + extend_args("std_dev_tilt1_g", N_g)
        + extend_args("std_dev_tilt2_g", N_g)
        + extend_args("variance_chi1_g", N_g)
        + extend_args("variance_chi2_g", N_g),
        model_json,
    )

    popmodel_magazine.register(
        (
            m1_source_name,
            m2_source_name,
            chi1_name,
            chi2_name,
            cos_tilt_1_name,
            cos_tilt_2_name,
        ),
        NPowerLawMGaussianWithDefaultSpinMagnitudeAndSpinMisalignment(
            N_pl=N_pl, N_g=N_g, **model_param
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

    popfactory.analysis_time = args.analysis_time
    popfactory.constraint = constraint
    popfactory.error_size = args.error_size
    popfactory.log_VT_fn = get_logVT(args.vt_path)
    popfactory.num_realizations = args.num_realizations
    popfactory.rate = args.rate
    popfactory.VT_params = [m1_source_name, m2_source_name]

    popfactory.produce()
