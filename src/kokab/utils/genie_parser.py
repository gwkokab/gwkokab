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

from numpyro.distributions.distribution import enable_validation


def get_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Genie
    script.

    :param parser: Parser to add the arguments to
    :return: the command line argument parser
    """

    # Global enable validation for all distributions
    enable_validation()

    genie_group = parser.add_argument_group("Genie Options")

    genie_group.add_argument(
        "--error-size",
        help="Size of the error.",
        default=2000,
        type=int,
    )
    genie_group.add_argument(
        "--num-realizations",
        help="Number of realizations.",
        default=5,
        type=int,
    )

    vt_group = parser.add_argument_group("VT Options")

    vt_group.add_argument(
        "--vt-json",
        help="Path to the JSON file containing the VT options.",
        type=str,
        required=True,
    )

    pmean_group = parser.add_argument_group("Poisson Mean Options")

    pmean_group.add_argument(
        "--pmean-json",
        help="Path to the JSON file containing the Poisson mean options.",
        type=str,
        required=True,
    )

    return parser
