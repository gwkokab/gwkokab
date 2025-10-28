# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse


def make_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    Returns
    -------
    argparse.ArgumentParser
        the command line argument parser
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
        default=None,
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
    parser.add_argument(
        "--dpi",
        help="dots per inch to save file",
        type=int,
        default=100,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    import os

    import glasbey
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    if args.color:
        color = np.loadtxt(args.color)

    data = pd.read_csv(args.data.name, delimiter=" ")
    x = data[args.x_value_column_name].to_numpy()
    y = data[args.y_value_column_name].to_numpy()

    if not args.color:
        plt.scatter(x, y, s=args.pointer_size)
    else:
        unique_colors = np.unique(color)
        if args.override_color:
            ALL_COLORS = args.override_color
        else:
            ALL_COLORS = glasbey.create_palette(palette_size=unique_colors.shape[0])
        for i, unique_color in enumerate(unique_colors):
            mask = color == unique_color
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
    plt.tight_layout()
    # Determine output file type and save accordingly
    output_ext = os.path.splitext(args.output.name)[1].lower()
    plt_savefig_kwargs = dict()
    if output_ext == ".png":
        plt_savefig_kwargs["dpi"] = args.dpi
    plt.savefig(args.output.name, bbox_inches="tight", **plt_savefig_kwargs)
    plt.close("all")
    args.data.close()
