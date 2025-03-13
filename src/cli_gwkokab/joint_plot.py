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


import argparse

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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
        "--x-column-name",
        help="column name for x-axis",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-y",
        "--y-column-name",
        help="column name for y-axis",
        required=True,
        type=str,
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
    parser.add_argument(
        "--use-latex",
        help="use LaTeX for rendering text",
        action="store_true",
    )
    parser.add_argument(
        "--font-family",
        help="font family to use",
        type=str,
        default=None,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    data = pd.read_csv(args.data.name, delimiter=" ")

    g = sns.jointplot(
        x=data[args.x_column_name].to_numpy(),
        y=data[args.y_column_name].to_numpy(),
        marginal_ticks=True,
        ratio=2,
    )
    g.plot_marginals(sns.histplot, color=args.color)
    g.plot_joint(sns.kdeplot, fill=True, thresh=0, cmap=args.cmap)
    g.set_axis_labels(args.xlabel, args.ylabel)
    g.fig.suptitle(args.title)
    g.savefig(args.output.name, bbox_inches="tight")
