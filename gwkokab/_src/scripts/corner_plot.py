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


from __future__ import annotations

import argparse

import corner
import numpy as np


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corner plotter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots a corner plot.",
    )
    parser.add_argument(
        "-d",
        "--data",
        help="data file path. Only .dat files are supported.",
        required=True,
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file path",
        required=True,
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        help="labels for the corner plot",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--truths",
        help="truth values for the corner plot",
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "-q",
        "--quantiles",
        help="quantiles for the corner plot",
        default=(0.25, 0.5, 0.75),
        type=list[float],
    )
    parser.add_argument(
        "-b",
        "--bins",
        help="number of bins for the corner plot",
        default=20,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--smooth",
        help="smooth the corner plot",
        action="store_true",
    )
    parser.add_argument(
        "-scale",
        "--scale",
        help="scale the corner plot",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-size",
        "--size",
        help="size of the corner plot in inches",
        default=(6, 6),
        type=tuple[float, float],
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    data = np.loadtxt(args.data.name)
    figure = corner.corner(
        data,
        labels=args.labels,
        truths=args.truths if args.truths is not None else None,
        show_titles=True,
        quantiles=args.quantiles,
        bins=args.bins,
        smooth=args.smooth,
    )
    scaling_factor = args.scale
    figure.set_size_inches(scaling_factor * args.size[0], scaling_factor * args.size[1])
    figure.savefig(args.output.name)
