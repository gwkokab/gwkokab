#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def get_parser() -> ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    :return: the command line argument parser
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    genie_group = parser.add_argument_group("Genie Options")

    genie_group.add_argument(
        "--vt_path",
        help="Path to the neural VT",
        type=str,
        required=True,
    )
    genie_group.add_argument(
        "--analysis_time",
        help="Analysis time of the VT",
        default=0.0,
        type=float,
        required=True,
    )
    genie_group.add_argument(
        "--error_size",
        help="Size of the error.",
        default=2000,
        type=int,
        required=True,
    )
    genie_group.add_argument(
        "--num_realizations",
        help="Number of realizations.",
        default=5,
        type=int,
        required=True,
    )
    genie_group.add_argument(
        "--rate",
        help="Rate of binary mergers.",
        type=float,
        required=True,
    )

    return parser
