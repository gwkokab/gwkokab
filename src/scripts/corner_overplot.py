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
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--data-labels",
        help="labels for the data files",
        nargs="+",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-l",
        "--column-labels",
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
        "--colors",
        help="colors of the corner plot",
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

    import corner
    import glasbey
    import pandas as pd
    from matplotlib import pyplot as plt

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    N = len(args.data)
    if args.colors is None:
        colors = glasbey.create_palette(palette_size=N)
    else:
        assert len(args.colors) == N, (
            "Number of colors must match number of data files."
        )
        colors = args.colors

    with open(args.data[0]) as f:
        header = f.readline().lstrip("#").strip().split()
    assert len(header) > 0, "Header must not be empty."

    if args.column_labels is None:
        column_labels = header
    else:
        column_labels = args.column_labels

    assert len(header) == len(column_labels), (
        f"Number of columns in the header ({len(header)}) does not match the number "
        f"of column labels provided ({len(column_labels)})."
    )

    data = []
    for file in args.data:
        data_i = pd.read_csv(file, delimiter=" ", skiprows=1).to_numpy()
        with open(file) as f:
            header_i = f.readline().lstrip("#").strip().split()
        assert len(header_i) == len(header), (
            f"Number of columns in {file} are not equal to the expected number {len(header)}."
        )
        reordered_indices = [header.index(col) for col in header_i]
        data_i = data_i[:, reordered_indices]

        data.append(data_i)

    scaling_factor = args.scale
    fig = plt.figure(
        figsize=(scaling_factor * args.size[0], scaling_factor * args.size[1]),
        dpi=args.dpi,
    )

    for i in range(N):
        fig = corner.corner(
            data[i],
            fig=fig,
            labels=column_labels,
            truths=args.truths if args.truths is not None else None,
            quantiles=args.quantiles,
            bins=args.bins,
            smooth=args.smooth,
            show_titles=args.show_titles,
            truth_color=args.truth_color,
            color=colors[i],
            plot_datapoints=False,
            range=args.range,
            title_fmt=args.title_fmt,
            plot_density=False,
            hist_kwargs={"density": True},
        )

    legend_handles = [plt.Line2D([], [], color=color, linewidth=2) for color in colors]
    fig.legend(
        legend_handles,
        args.data_labels,
        loc="upper right",
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close("all")
