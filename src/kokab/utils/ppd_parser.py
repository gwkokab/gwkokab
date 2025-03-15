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

    ppd_group = parser.add_argument_group("PPD Options")

    ppd_group.add_argument(
        "--sample-filename",
        help="Path of the file to save the samples.",
        type=str,
        required=True,
    )
    ppd_group.add_argument(
        "--filename",
        help="Path of the file to save the PPD.",
        type=str,
        required=True,
    )
    ppd_group.add_argument(
        "--constants",
        help="Path to the JSON file containing the constant parameters.",
        type=str,
        required=True,
    )
    ppd_group.add_argument(
        "--nf-samples-mapping",
        help="Path to the JSON file containing the mapping of the number of samples.",
        type=str,
        required=True,
    )
    ppd_group.add_argument(
        "--range",
        help="Range of the PPD for each parameter. The format is 'name min max step'. "
        "Repeat for each parameter.",
        nargs=4,
        action="append",
        type=str,
        required=True,
    )
    ppd_group.add_argument(
        "--batch-size",
        help="Batch size for the computation of log prob per sample.",
        type=int,
        default=1000,
    )

    return parser
