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

import mplcursors
import pandas as pd
from matplotlib import pyplot as plt


def make_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    :return: the command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Scatter batch 3D plotter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots a batch 3D scatter plot.",
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
        "-z",
        "--z-value-column-name",
        help="name of the z-axis values",
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
        "-zl",
        "--zlabel",
        help="label of the z-axis",
        default="z",
        type=str,
    )
    parser.add_argument(
        "-cmap",
        "--color-map",
        help="color map for the plot",
        default="plasma",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--color",
        help="index of the color values",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-cbar",
        "--color-bar",
        help="show color bar",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--scatter-size",
        help="size of the scatter points",
        default=5,
        type=int,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        help="transparency of the scatter points",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "-m",
        "--marker",
        help="marker style for the scatter points",
        default=".",
        type=str,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    file_list = glob.glob(args.data_regex)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for file_path in file_list:
        data = pd.read_csv(file_path, delimiter=" ")
        x = data[args.x_value_column_name].to_numpy()
        y = data[args.y_value_column_name].to_numpy()
        z = data[args.z_value_column_name].to_numpy()
        ax.scatter(
            x,
            y,
            z,
            c=data[:, args.color] if args.color is not None else z,
            cmap=args.color_map,
            s=args.scatter_size,
            alpha=args.alpha,
            marker=args.marker,
        )
    if args.color_bar:
        cursor = mplcursors.cursor(ax, hover=True)
        cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                sel.annotation.get_text() + " " + str(sel.target.get_array()[sel.index])
            ),
        )

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.output.name, bbox_inches="tight")
    plt.close("all")