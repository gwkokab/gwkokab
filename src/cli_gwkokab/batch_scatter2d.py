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

    import glob
    import os

    import glasbey
    import pandas as pd
    from matplotlib import pyplot as plt

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    file_list = glob.glob(args.data_regex)

    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(
                color=glasbey.create_palette(palette_size=len(file_list))
            )
        }
    )

    for file_path in file_list:
        data = pd.read_csv(file_path, delimiter=" ")
        x = data[args.x_value_column_name].to_numpy()
        y = data[args.y_value_column_name].to_numpy()

        plt.scatter(
            x,
            y,
            s=args.scatter_size,
            alpha=args.alpha,
            marker=args.marker,
        )

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.tight_layout()
    # Determine output file type and save accordingly
    output_ext = os.path.splitext(args.output.name)[1].lower()
    plt_savefig_kwargs = dict()
    if output_ext == ".png":
        plt_savefig_kwargs["dpi"] = args.dpi
    plt.savefig(args.output.name, bbox_inches="tight", **plt_savefig_kwargs)
    plt.close("all")
