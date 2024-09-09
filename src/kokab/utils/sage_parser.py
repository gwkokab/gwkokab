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


from argparse import ArgumentParser


def get_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Sage script.

    :param parser: Parser to add the arguments to
    :return: the command line argument parser
    """

    sage_group = parser.add_argument_group("Sage Options")

    sage_group.add_argument(
        "--vt-path",
        help="Path to the neural VT",
        type=str,
        required=True,
    )
    sage_group.add_argument(
        "--posterior-regex",
        help="Regex for the posterior samples.",
        type=str,
        required=True,
    )
    sage_group.add_argument(
        "--posterior-columns",
        help="Columns of the posterior samples.",
        nargs="+",
        type=str,
        required=True,
    )
    sage_group.add_argument(
        "--analysis_time",
        help="Analysis time of the VT",
        default=0.0,
        type=float,
        required=True,
    )
    sage_group.add_argument(
        "--n-chains",
        help="Number of chains.",
        default=5,
        type=int,
        required=True,
    )
    sage_group.add_argument(
        "--seed",
        help="Seed for the random number generator.",
        default=37,
        type=int,
        required=True,
    )
    sage_group.add_argument(
        "--n-dim",
        help="Number of dimensions.",
        type=int,
        required=True,
    )

    local_sampler_group = parser.add_argument_group(
        "Local Sampler Options",
        description="At the moment we are only supporting MALA Sampler.",
    )

    local_sampler_group.add_argument(
        "--step-size",
        help="Step size for the MALA sampler.",
        type=float,
        default=1e-2,
        required=True,
    )
    local_sampler_group.add_argument(
        "--jit",
        help="Just-in-time compilation.",
        action="store_true",
    )

    global_sampler_group = parser.add_argument_group(
        "Global Sampler Options",
        description="At the moment we are only supporting MaskedCouplingRQSpline Model.",
    )

    global_sampler_group.add_argument(
        "--n-layers",
        help="Number of layers in the model.",
        type=int,
        default=5,
        required=True,
    )
    global_sampler_group.add_argument(
        "--hidden-size",
        help="Hidden size of the model.",
        type=int,
        nargs="+",
        default=[32, 32],
        required=True,
    )
    global_sampler_group.add_argument(
        "--num-bins",
        help="Number of bins in the model.",
        type=int,
        default=8,
        required=True,
    )

    sampler_group = parser.add_argument_group("Sampler Options")

    sampler_group.add_argument(
        "--n-local-steps",
        help="Number of local steps.",
        type=int,
        default=100,
        required=True,
    )
    sampler_group.add_argument(
        "--n-global-steps",
        help="Number of global steps.",
        type=int,
        default=100,
        required=True,
    )
    sampler_group.add_argument(
        "--n-loop-training",
        help="Number of loop training.",
        type=int,
        default=10,
        required=True,
    )
    sampler_group.add_argument(
        "--n-loop-production",
        help="Number of loop production.",
        type=int,
        default=10,
        required=True,
    )
    sampler_group.add_argument(
        "--batch-size",
        help="Batch size.",
        type=int,
        default=10000,
        required=True,
    )
    sampler_group.add_argument(
        "--n-epochs",
        help="Number of epochs.",
        type=int,
        default=5,
        required=True,
    )
    sampler_group.add_argument(
        "--learning-rate",
        help="Learning rate.",
        type=float,
        default=0.001,
    )
    sampler_group.add_argument(
        "--momentum",
        help="Momentum.",
        type=float,
        default=0.9,
    )
    sampler_group.add_argument(
        "--precompile",
        help="Precompile the model.",
        action="store_true",
        default=False,
    )
    sampler_group.add_argument(
        "--verbose",
        help="Verbose.",
        action="store_true",
        default=False,
    )
    sampler_group.add_argument(
        "--use-global",
        help="Use global.",
        action="store_true",
    )
    sampler_group.add_argument(
        "--logging",
        help="Logging.",
        action="store_true",
    )

    data_dump_group = parser.add_argument_group("Data Dump Options")

    data_dump_group.add_argument(
        "--out-dir",
        help="Output directory.",
        type=str,
        default="sampler_data",
    )
    data_dump_group.add_argument(
        "--labels",
        help="Labels for the data.",
        type=str,
        nargs="+",
    )
    data_dump_group.add_argument(
        "--n-samples",
        help="Number of samples.",
        type=int,
        default=20000,
    )

    return parser
