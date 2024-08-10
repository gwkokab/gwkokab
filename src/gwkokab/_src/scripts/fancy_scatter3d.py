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
from plotly import graph_objects as go


def make_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    :return: the command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Scatter 3D plotter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots a 3D scatter plot.",
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
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "-rn",
        "--random-sample",
        help="randomly sample the data",
        type=int,
    )
    parser.add_argument(
        "-x",
        "--x-value-column-index",
        help="index of the x-axis values",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-y",
        "--y-value-column-index",
        help="index of the y-axis values",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-z",
        "--z-value-column-index",
        help="index of the z-axis values",
        required=True,
        type=int,
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
        "-a",
        "--alpha",
        help="transparency of the points",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "-s",
        "--show",
        help="show the plot",
        action="store_true",
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    data = np.loadtxt(args.data.name)
    if args.random_sample:
        data = data[
            np.random.choice(data.shape[0], args.random_sample, replace=False),
            :,
        ]
    x = data[:, args.x_value_column_index]
    y = data[:, args.y_value_column_index]
    z = data[:, args.z_value_column_index]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={
                "size": 3,
                "opacity": args.alpha,
                "color": data[:, args.color],
                "colorscale": args.color_map,
            },
        ),
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=args.xlabel,
            yaxis_title=args.ylabel,
            zaxis_title=args.zlabel,
        ),
    )

    if args.title:
        fig.update_layout(title=args.title)

    if args.output:
        fig.write_html(args.output.name)

    if args.show:
        fig.show()
