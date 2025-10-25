# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse


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
        "--band",
        help="plot the band instead of the chains",
        action="store_true",
    )
    parser.add_argument(
        "--band-color",
        help="color of the band",
        default="#1f77b4",
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

    import glob
    import os

    import glasbey
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    files = glob.glob(args.data_regex)
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(
                color=glasbey.create_palette(palette_size=len(files))
            )
        }
    )

    n_dim = pd.read_csv(files[0], delimiter=" ", skiprows=1).to_numpy().shape[1]

    if args.height is None:
        figsize = (args.width, n_dim * 2.5)
    else:
        figsize = (args.width, args.height)

    band = args.band
    band_color = args.band_color
    _, ax = plt.subplots(n_dim, 1, figsize=figsize, sharex=True)
    if n_dim == 1:
        for file in files:
            data = pd.read_csv(file, delimiter=" ", skiprows=1).to_numpy()
            n_points = data.shape[0]
            if not band:
                ax.plot(data, alpha=args.alpha)
            else:
                quantiles = np.quantile(data, [0.05, 0.25, 0.75, 0.95], axis=-1)
                mean_values = np.mean(data, axis=-1)
                ax.fill_between(
                    range(n_points),
                    quantiles[0],
                    quantiles[-1],
                    alpha=0.2,
                    color=band_color,
                )
                ax.fill_between(
                    range(n_points),
                    quantiles[1],
                    quantiles[-2],
                    alpha=0.5,
                    color=band_color,
                )
                ax.plot(
                    range(n_points),
                    mean_values,
                    color=band_color,
                    alpha=1.0,
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
        data = np.stack(
            [pd.read_csv(file, delimiter=" ", skiprows=1).to_numpy() for file in files],
            axis=-1,
        )
        n_points = data.shape[0]
        if band:
            quantiles = np.quantile(data, [0.05, 0.25, 0.75, 0.95], axis=-1)
            mean_values = np.mean(data, axis=-1)

        for i in range(n_dim):
            ax[i].set_ylabel(args.labels[i])
            ax[i].tick_params(
                axis="both",
                which="both",
                labelleft=True,
                labelright=True,
                labeltop=i == 0,
                labelbottom=i == n_dim - 1,
            )
            ax[i].grid(visible=True, which="both", axis="both", alpha=0.5)
            if not band:
                data_ = data[:, i, :]
                ax[i].plot(data_, alpha=args.alpha)
            else:
                quantiles_ = quantiles[:, :, i]
                mean_values_ = mean_values[:, i]
                ax[i].fill_between(
                    range(n_points),
                    quantiles_[0],
                    quantiles_[-1],
                    alpha=0.2,
                    color=band_color,
                )
                ax[i].fill_between(
                    range(n_points),
                    quantiles_[1],
                    quantiles_[-2],
                    alpha=0.5,
                    color=band_color,
                )
                ax[i].plot(
                    range(n_points),
                    mean_values_,
                    color=band_color,
                    alpha=1.0,
                )
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
