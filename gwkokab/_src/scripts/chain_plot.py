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
import glob

import numpy as np
from matplotlib import pyplot as plt


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command line interface for plotting chains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots chains from .dat files.",
    )
    parser.add_argument(
        "-d",
        "--data-regex",
        help="regex pattern for the data files. Only .dat files are supported.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file path",
        required=True,
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "-dim",
        "--dimension",
        help="dimension of the data",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        help="labels for the chains",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        help="transparency of the chains",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "-t",
        "--title",
        help="title for the plot",
        type=str,
    )
    parser.add_argument(
        "-width",
        "--width",
        help="width of the plot in inches",
        default=20,
        type=int,
    )
    parser.add_argument(
        "-height",
        "--height",
        help="height of the plot in inches",
        default=None,
        type=int,
    )

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    files = glob.glob(args.data_regex)
    n_dim = args.dimension

    if args.height is None:
        figsize = (args.width, n_dim * 2.5)
    else:
        figsize = (args.width, args.height)

    fig, ax = plt.subplots(n_dim, 1, figsize=figsize, sharex=True)
    for file in files:
        data = np.loadtxt(file)
        for j, data_ in enumerate(data.T):
            ax[j].plot(
                data_,
                alpha=args.alpha,
            )
            ax[j].set_ylabel(args.labels[j])
    if args.title:
        plt.suptitle(args.title)
    plt.tight_layout()
    fig.savefig(args.output.name)
