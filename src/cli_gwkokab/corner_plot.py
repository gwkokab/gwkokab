# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corner plotter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots a corner plot.",
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
        "-l",
        "--labels",
        nargs="+",
        help="labels for the corner plot",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--truths",
        help="truth values for the corner plot",
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "-q",
        "--quantiles",
        help="quantiles for the corner plot",
        default=None,
        type=list[float],  # type: ignore[arg-type]
    )
    parser.add_argument(
        "-b",
        "--bins",
        help="number of bins for the corner plot",
        default=20,
        type=int,
    )
    parser.add_argument(
        "-s",
        "--smooth",
        help="smooth the corner plot",
        action="store_true",
    )
    parser.add_argument(
        "-st",
        "--show-titles",
        help="show titles in the corner plot",
        action="store_true",
    )
    parser.add_argument(
        "-scale",
        "--scale",
        help="scale the corner plot",
        default=1.0,
        type=float,
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
        "--range",
        help="range of the corner plot",
        nargs="+",
        action="append",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--truth-color",
        help="color of the truth values in the corner plot",
        default="red",
        type=str,
    )
    parser.add_argument(
        "--color",
        help="color of the corner plot",
        default="#3498DB",
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
    parser.add_argument(
        "--title-fmt",
        help="format string for titles in the corner plot",
        default=".3f",
        type=str,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    import os

    import corner
    import pandas as pd
    from matplotlib import pyplot as plt

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    data = pd.read_csv(args.data.name, delimiter=" ", skiprows=1).to_numpy()
    figure = corner.corner(
        data,
        labels=args.labels,
        truths=args.truths if args.truths is not None else None,
        quantiles=args.quantiles,
        bins=args.bins,
        smooth=args.smooth,
        show_titles=args.show_titles,
        truth_color=args.truth_color,
        color=args.color,
        plot_datapoints=False,
        range=args.range,
        title_fmt=args.title_fmt,
    )
    scaling_factor = args.scale
    figure.set_size_inches(scaling_factor * args.size[0], scaling_factor * args.size[1])
    # Determine output file type and save accordingly
    output_ext = os.path.splitext(args.output.name)[1].lower()
    plt_savefig_kwargs = dict()
    if output_ext == ".png":
        plt_savefig_kwargs["dpi"] = args.dpi
    plt.savefig(args.output.name, bbox_inches="tight", **plt_savefig_kwargs)
    plt.close("all")
