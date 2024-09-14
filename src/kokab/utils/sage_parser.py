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
        default=1.0,
        type=float,
    )
    sage_group.add_argument(
        "--n-chains",
        help="Number of chains.",
        default=5,
        type=int,
    )
    sage_group.add_argument(
        "--seed",
        help="Seed for the random number generator.",
        default=37,
        type=int,
    )

    flowMC_group = parser.add_argument_group("flowMC Options")

    flowMC_group.add_argument(
        "--flowMC-json",
        help="Path to a JSON file containing the flowMC options. It should contains"
        "keys: local_sampler_kwargs, nf_model_kwargs, sampler_kwargs, data_dump_kwargs,"
        " and their respective values.",
    )

    return parser