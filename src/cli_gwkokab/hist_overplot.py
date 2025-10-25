# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Joint plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script creates over plots of histograms for multiple columns in a data file.",
    )
    parser.add_argument(
        "--data",
        help="data file path. Only .dat files are supported.",
        required=True,
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--delimiter",
        help="delimiter for the data file",
        default=" ",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="output file path",
        required=True,
        type=argparse.FileType("w"),
    )
    parser.add_argument(
        "--column-names",
        help="names of the columns to plot",
        required=True,
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--labels",
        help="labels for the columns",
        default=None,
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--xlabel",
        help="label for x-axis",
        type=str,
    )
    parser.add_argument(
        "--ylabel",
        help="label for y-axis",
        type=str,
    )
    parser.add_argument(
        "--title",
        help="title of the plot",
        type=str,
    )
    parser.add_argument(
        "--size",
        help="size of the corner plot in inches",
        nargs=2,
        default=(6, 6),
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
    parser.add_argument(
        "--bins",
        help="number of bins for the histogram",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--alpha",
        help="transparency of the histogram bars",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--x-scale",
        help="scale of the x-axis",
        choices=["linear", "log", "symlog"],
        type=str,
        default="linear",
    )
    parser.add_argument(
        "--y-scale",
        help="scale of the y-axis",
        choices=["linear", "log", "symlog"],
        type=str,
        default="linear",
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    import os

    import pandas as pd
    from matplotlib import pyplot as plt

    if args.labels is None:
        labels = args.column_names
    else:
        assert len(args.labels) == len(args.column_names), (
            "Number of labels must match number of column names."
        )
        labels = args.labels

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    with open(args.data.name) as f:
        header = f.readline().lstrip("#").strip().split()

    data = pd.read_csv(args.data.name, sep=r"\s+", comment="#", header=None)
    data.columns = header

    plt.figure(figsize=args.size)
    for x_column in args.column_names:
        if x_column not in data.columns:
            raise ValueError(f"Column '{x_column}' not found in data.")

    for x_column, label in zip(args.column_names, labels):
        plt.hist(
            data[x_column],
            bins=args.bins,
            density=True,
            alpha=args.alpha,
            label=label,
        )
    if args.xlabel is not None:
        plt.xlabel(args.xlabel)
    if args.ylabel is not None:
        plt.ylabel(args.ylabel)
    if args.title is not None:
        plt.title(args.title)
    plt.legend()
    plt.xscale(args.x_scale)
    plt.yscale(args.y_scale)
    plt.tight_layout()

    output_ext = os.path.splitext(args.output.name)[1].lower()
    plt_savefig_kwargs = dict()
    if output_ext == ".png":
        plt_savefig_kwargs["dpi"] = args.dpi
    plt.savefig(args.output.name, bbox_inches="tight", **plt_savefig_kwargs)
    plt.close("all")
