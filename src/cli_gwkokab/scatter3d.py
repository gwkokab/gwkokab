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
        "-a",
        "--alpha",
        help="transparency of the points",
        default=0.3,
        type=float,
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

    import mplcursors
    import pandas as pd
    from matplotlib import pyplot as plt

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    data = pd.read_csv(args.data.name, delimiter=" ")
    x = data[args.x_value_column_name].to_numpy()
    y = data[args.y_value_column_name].to_numpy()
    z = data[args.z_value_column_name].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x,
        y,
        z,
        c=data[:, args.color] if args.color is not None else z,
        cmap=args.color_map,
        marker="o",
        alpha=args.alpha,
    )
    if args.title is not None:
        ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_zlabel(args.zlabel)
    if args.color_bar:
        cbar = fig.colorbar(sc)
        cbar.set_label("Color")
        cursor = mplcursors.cursor(sc, hover=True)
        cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"{sel.annotation.get_text()}\n{sel.artist.get_array()}"
            ),
        )
    plt.tight_layout()
    # Determine output file type and save accordingly
    output_ext = os.path.splitext(args.output.name)[1].lower()
    plt_savefig_kwargs = dict()
    if output_ext == ".png":
        plt_savefig_kwargs["dpi"] = args.dpi
    plt.savefig(args.output.name, bbox_inches="tight", **plt_savefig_kwargs)
    plt.close("all")
    args.data.close()
