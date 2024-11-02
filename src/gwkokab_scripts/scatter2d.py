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

import argparse

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def make_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    :return: the command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Scatter 2D plotter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots a 2D scatter plot.",
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
        "--x-value-column-name",
        help="name of the x-axis values",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-y",
        "--y-value-column-name",
        help="name of the y-axis values",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--title",
        help="title of the plot",
        type=str,
    )
    parser.add_argument(
        "-xl",
        "--xlabel",
        help="label of the x-axis",
        default="x",
        type=str,
    )
    parser.add_argument(
        "-yl",
        "--ylabel",
        help="label of the y-axis",
        default="y",
        type=str,
    )
    parser.add_argument(
        "--x-scale",
        help="scale of the x-axis",
        default="linear",
        type=str,
    )
    parser.add_argument(
        "--y-scale",
        help="scale of the y-axis",
        default="linear",
        type=str,
    )
    parser.add_argument(
        "--color",
        help="path to the file containing the color values",
        type=str,
    )
    parser.add_argument(
        "--legend",
        help="legend of the plot",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--pointer-size",
        help="size of the pointer",
        type=float,
    )
    parser.add_argument(
        "--override-color",
        help="override the color of the pointers",
        nargs="+",
        type=str,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    if args.color:
        color = np.loadtxt(args.color)

    data = pd.read_csv(args.data.name, delimiter=" ")
    x = data[args.x_value_column_name].to_numpy()
    y = data[args.y_value_column_name].to_numpy()

    if not args.color:
        plt.scatter(x, y, s=args.pointer_size)
    else:
        if args.override_color:
            ALL_COLORS = args.override_color
        else:
            ALL_COLORS = (
                list(mcolors.TABLEAU_COLORS.values())
                + list(mcolors.XKCD_COLORS.values())
                + list(mcolors.CSS4_COLORS.values())
                + list(mcolors.BASE_COLORS.values())
            )
        unique_colors = np.unique(color)
        for i, unique_color in enumerate(unique_colors):
            mask = color == unique_color
            print(i, unique_color, args.legend[i])
            plt.scatter(
                x[mask],
                y[mask],
                label=args.legend[i] if args.legend else None,
                c=ALL_COLORS[int(unique_color)],
                s=args.pointer_size,
            )
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.xscale(args.x_scale)
    plt.yscale(args.y_scale)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output.name)

    args.data.close()
