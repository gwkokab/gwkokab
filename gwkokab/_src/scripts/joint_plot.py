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

import numpy as np
import seaborn as sns


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots a joint plot of two columns of a data file.",
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
        "-x",
        "--x-column",
        help="column index for x-axis",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-y",
        "--y-column",
        help="column index for y-axis",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-xl",
        "--xlabel",
        help="label for x-axis",
        type=str,
    )
    parser.add_argument(
        "-yl",
        "--ylabel",
        help="label for y-axis",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--title",
        help="title of the plot",
        type=str,
    )
    parser.add_argument(
        "-size",
        "--size",
        help="size of the corner plot in inches",
        nargs=2,
        default=(6, 6),
        type=float,
    )
    parser.add_argument(
        "-cmap",
        "--cmap",
        help="color map",
        default="rocket",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--color",
        help="color of the corner plot",
        default="#180F29",
        type=str,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    data = np.loadtxt(args.data.name)

    g = sns.jointplot(
        x=data[:, args.x_column],
        y=data[:, args.y_column],
        marginal_ticks=True,
        ratio=2,
    )
    g.plot_marginals(sns.histplot, color=args.color)
    g.plot_joint(sns.kdeplot, fill=True, thresh=0, cmap=args.cmap)
    g.set_axis_labels(args.xlabel, args.ylabel)
    g.fig.suptitle(args.title)
    g.savefig(args.output.name)
