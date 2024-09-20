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

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from jax import numpy as jnp, vmap
from jaxtyping import Array, Bool

from gwkokab.errors import banana_error_m1_m2
from gwkokab.models import PowerLawPrimaryMassRatio
from gwkokab.parameters import MASS_RATIO, PRIMARY_MASS_SOURCE
from gwkokab.population import error_magazine, popfactory, popmodel_magazine
from gwkokab.utils.transformations import m1_q_to_m2, mass_ratio
from gwkokab.vts._neuralvt import load_model

from ..utils import genie_parser


m1_source_name = PRIMARY_MASS_SOURCE.name
mass_ratio_name = MASS_RATIO.name


def get_logVT(vt_path):
    _, logVT = load_model(vt_path)

    def m1q_logVT(x: Array) -> Array:
        m1 = x[..., 0]
        q = x[..., 1]
        m2 = m1_q_to_m2(m1=m1, q=q)
        m1m2 = jnp.column_stack([m1, m2])
        return vmap(logVT)(m1m2)

    return m1q_logVT


def constraint(x: Array) -> Bool:
    m1 = x[..., 0]
    q = x[..., 1]
    mask = m1 > 0.0
    mask &= q >= 0.0
    mask &= q <= 1.0
    return mask


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = genie_parser.get_parser(parser)

    model_group = parser.add_argument_group("Model Options")

    model_group.add_argument(
        "--alpha",
        help="Power law index for primary mass",
        type=float,
        required=True,
    )
    model_group.add_argument(
        "--beta",
        help="Power law index for mass ratio",
        type=float,
        required=True,
    )
    model_group.add_argument(
        "--mmin",
        help="Minimum mass",
        type=float,
        required=True,
    )
    model_group.add_argument(
        "--mmax",
        help="Maximum mass",
        type=float,
        required=True,
    )

    err_group = parser.add_argument_group("Error Options")
    err_group.add_argument(
        "--err_scale_Mc",
        help="Scale of the error in chirp mass.",
        default=1.0,
        type=float,
        required=True,
    )
    err_group.add_argument(
        "--err_scale_eta",
        help="Scale of the error in symmetric mass ratio.",
        default=1.0,
        type=float,
        required=True,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    popmodel_magazine.register(
        (m1_source_name, mass_ratio_name),
        PowerLawPrimaryMassRatio(
            alpha=args.alpha,
            beta=args.beta,
            mmin=args.mmin,
            mmax=args.mmax,
        ),
    )

    @error_magazine.register((m1_source_name, mass_ratio_name))
    def banana_error_m1_q(x, size, key):
        m1 = x[..., 0]
        q = x[..., 1]
        m2 = m1_q_to_m2(m1=m1, q=q)
        m1m2 = jnp.asarray((m1, m2))
        m1m2_err = banana_error_m1_m2(
            m1m2,
            size,
            key,
            scale_Mc=args.err_scale_Mc,
            scale_eta=args.err_scale_eta,
        )
        m1_err = m1m2_err[..., 0]
        m2_err = m1m2_err[..., 1]
        q_err = mass_ratio(m1=m1_err, m2=m2_err)
        return jnp.stack((m1_err, q_err), axis=-1)

    popfactory.analysis_time = args.analysis_time
    popfactory.constraint = constraint
    popfactory.error_size = args.error_size
    popfactory.log_VT_fn = get_logVT(args.vt_path)
    popfactory.num_realizations = args.num_realizations
    popfactory.rate = args.rate
    popfactory.VT_params = [m1_source_name, mass_ratio_name]

    popfactory.produce()
