# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse
import glob
import os

import pandas as pd
from matplotlib import pyplot as plt


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command line interface for plotting chains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots chains from .dat files.",
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
        "-dim",
        "--dimension",
        help="dimension of the data",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        help="labels for the chains",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        help="transparency of the chains",
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "-t",
        "--title",
        help="title for the plot",
        type=str,
    )
    parser.add_argument(
        "-width",
        "--width",
        help="width of the plot in inches",
        default=20,
        type=int,
    )
    parser.add_argument(
        "-height",
        "--height",
        help="height of the plot in inches",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-xs",
        "--x-scale",
        help="scale of the x-axis",
        default="linear",
        type=str,
    )
    parser.add_argument(
        "-ys",
        "--y-scale",
        help="scale of the y-axis",
        default="linear",
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
    parser = make_parser()
    args = parser.parse_args()

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    files = glob.glob(args.data_regex)
    n_dim = args.dimension

    if args.height is None:
        figsize = (args.width, n_dim * 2.5)
    else:
        figsize = (args.width, args.height)
    fig, ax = plt.subplots(n_dim, 1, figsize=figsize, sharex=True)
    if n_dim == 1:
        for file in files:
            data = pd.read_csv(file, delimiter=" ", skiprows=1).to_numpy()
            ax.plot(
                data,
                alpha=args.alpha,
            )
            ax.set_ylabel(args.labels[0])
            plt.tick_params(
                axis="both",
                which="both",
                labelleft=True,
                labelright=True,
                labeltop=True,
                labelbottom=True,
            )
            plt.grid(visible=True, which="both", axis="both", alpha=0.5)
    else:
        for file in files:
            data = pd.read_csv(file, delimiter=" ", skiprows=1).to_numpy()
            n = data.T.shape[0]
            for j, data_ in enumerate(data.T):
                ax[j].plot(
                    data_,
                    alpha=args.alpha,
                )
                ax[j].set_ylabel(args.labels[j])
                ax[j].tick_params(
                    axis="both",
                    which="both",
                    labelleft=True,
                    labelright=True,
                    labeltop=j == 0,
                    labelbottom=j == n - 1,
                )
                ax[j].grid(visible=True, which="both", axis="both", alpha=0.5)
    if args.title:
        plt.suptitle(args.title)
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
