# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import argparse
from typing import Any, List, Tuple

import numpy as np


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPD plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots ppd plots.",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        help="data files paths (one or more). Only .hdf5 files are supported.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory to save the plots",
    )
    parser.add_argument(
        "--prefix",
        help="prefix for the output file",
        type=str,
    )

    pretty_group = parser.add_argument_group("Pretty Options")
    pretty_group.add_argument(
        "--titles",
        nargs="+",
        type=str,
        help="titles for the plots in order of the parameters",
    )
    pretty_group.add_argument(
        "--x-scale",
        help="scale of the x-axis",
        default="linear",
        choices=["linear", "log"],
        type=str,
    )
    pretty_group.add_argument(
        "--y-scale",
        help="scale of the y-axis",
        default="linear",
        choices=["linear", "log"],
        type=str,
    )
    pretty_group.add_argument(
        "--x-range",
        nargs=2,
        action="append",
        type=float,
        help="range of the x axis plot in the form of start end for each parameter",
    )
    pretty_group.add_argument(
        "--y-range",
        nargs=2,
        action="append",
        type=float,
        help="range of the y axis plot in the form of start end for each parameter",
    )
    pretty_group.add_argument(
        "--font-size",
        default=16,
        type=int,
        help="font size for the x,y axis labels and titles",
    )
    pretty_group.add_argument(
        "--x-labels",
        nargs="+",
        type=str,
        help="labels for the x axis in order of the parameters",
    )
    pretty_group.add_argument(
        "--y-labels",
        nargs="+",
        type=str,
        help="labels for the y axis in order of the parameters",
    )
    pretty_group.add_argument(
        "--size",
        help="size of the ppd plot in inches",
        nargs=2,
        default=(12, 12),
        type=float,
    )
    pretty_group.add_argument(
        "--use-latex",
        help="use LaTeX for rendering text",
        action="store_true",
    )
    pretty_group.add_argument(
        "--font-family",
        help="font family to use",
        type=str,
        default=None,
    )
    pretty_group.add_argument(
        "--median-color",
        help="color of the median line",
        type=str,
        default=r"#7B8794",
    )
    pretty_group.add_argument(
        "--ppd-color",
        help="color of the ppd line",
        type=str,
        default=r"#b41f78",
    )
    pretty_group.add_argument(
        "--ninety-ci-color",
        help="color of the 90 percentile CI",
        type=str,
        default=r"#C5C9C7",
    )
    pretty_group.add_argument(
        "--fifty-ci-color",
        help="color of the 50 percentile CI",
        type=str,
        default=r"#BBF90F",
    )
    pretty_group.add_argument(
        "--median-alpha",
        help="alpha of the median line",
        type=float,
        default=0.8,
    )
    pretty_group.add_argument(
        "--ppd-alpha",
        help="alpha of the ppd line",
        type=float,
        default=1.0,
    )
    pretty_group.add_argument(
        "--ninety-ci-alpha",
        help="alpha of the 90 percentile CI",
        type=float,
        default=0.5,
    )
    pretty_group.add_argument(
        "--fifty-ci-alpha",
        help="alpha of the 50 percentile CI",
        type=float,
        default=0.7,
    )
    pretty_group.add_argument(
        "--median-linestyle",
        help="linestyle of the median line",
        type=str,
        default="--",
    )
    pretty_group.add_argument(
        "--grid",
        help="show grid",
        action="store_true",
    )
    pretty_group.add_argument(
        "--grid-which",
        help="which grid to show",
        type=str,
        default="both",
        choices=["both", "major", "minor"],
    )
    pretty_group.add_argument(
        "--grid-alpha",
        help="alpha of the grid",
        type=float,
        default=0.7,
    )
    pretty_group.add_argument(
        "--grid-linestyle",
        help="linestyle of the grid",
        type=str,
        default="--",
    )
    parser.add_argument(
        "--dpi",
        help="dots per inch to save file",
        type=int,
        default=100,
    )

    return parser


def get_domain(
    domain_as_read: Any,
) -> List[Tuple[float, float, int]]:
    """Get the domain of the model per axis.

    Parameters
    ----------
    domain_as_read : Any
        The domain of the model per axis.

    Returns
    -------
    List[Tuple[float, float, int]]
        The domain of the model per axis.
    """
    return [(float(s), float(e), int(n)) for s, e, n in domain_as_read]


def get_utf8_decoded_headers(headers_as_read: Any) -> List[str]:
    """Get utf-8 decoded headers.

    Parameters
    ----------
    headers_as_read : Any
        The headers as read from the file.

    Returns
    -------
    List[str]
        The utf-8 decoded headers.
    """
    return [h.decode("utf-8") for h in headers_as_read]


def get_quantiles(data: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Get the quantiles of the data.

    Parameters
    ----------
    data : np.ndarray
        The data to get the quantiles of.
    quantiles : np.ndarray
        The quantiles to get. The quantiles should be between 0 and 1.

    Returns
    -------
    np.ndarray
        The quantiles of the data.
    """
    return np.quantile(data, quantiles, axis=1)


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    import h5py
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from kokab.utils.ppd import get_all_marginals

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    prefix = "" if args.prefix is None else args.prefix

    x_range = args.x_range
    y_range = args.y_range
    font_size = args.font_size
    x_labels = args.x_labels
    y_labels = args.y_labels
    titles = args.titles

    quantiles = [
        0.05,  # 90% CI
        0.25,  # 50% CI
        0.5,  # median
        0.75,  # 50% CI
        0.95,  # 90% CI
    ]

    axes: List[Tuple[Figure, Axes]] = []

    for j in range(len(args.data)):
        with h5py.File(args.data[j], "r") as f:
            domains = get_domain(f["domains"])
            headers = get_utf8_decoded_headers(f["headers"])
            marginals = [f["marginals"][head][:] for head in headers]
            ppd = f["ppd"][()]

        marginal_ppds = get_all_marginals(ppd, domains)

        if not axes:
            axes = [plt.subplots(figsize=args.size) for _ in range(len(headers))]

        for i in range(len(headers)):
            _, ax = axes[i]
            head = headers[i]
            marginal = marginals[i]
            marginal_ppd = marginal_ppds[i]
            domain = domains[i]

            start, end, num_points = domain
            quant = get_quantiles(marginal, quantiles)
            xx = np.linspace(start, end, num_points)

            ax.plot(
                xx,
                quant[2],
                # label="Median",
                color=args.median_color,
                alpha=args.median_alpha,
                linestyle=args.median_linestyle,
            )
            ax.plot(
                xx,
                marginal_ppd,
                # label="PPD",
                color=args.ppd_color,
                alpha=args.ppd_alpha,
            )
            ax.fill_between(
                xx,
                quant[0],
                quant[-1],
                alpha=args.ninety_ci_alpha,
                # label="90\% CI" if args.use_latex else "90% CI",
                color=args.ninety_ci_color,
            )
            ax.fill_between(
                xx,
                quant[1],
                quant[-2],
                alpha=args.fifty_ci_alpha,
                # label="50\% CI" if args.use_latex else "50% CI",
                color=args.fifty_ci_color,
            )
            ax.set_yscale(args.y_scale)
            ax.set_xscale(args.x_scale)
            if titles is not None:
                if i < len(titles):
                    ax.set_title(titles[i], fontsize=font_size)
            if x_labels is None:
                ax.set_xlabel(head, fontsize=font_size)
            else:
                if i < len(x_labels):
                    ax.set_xlabel(x_labels[i], fontsize=font_size)

            if y_labels is None:
                ax.set_ylabel(f"ppd({head})", fontsize=font_size)
            else:
                if i < len(y_labels):
                    ax.set_ylabel(y_labels[i], fontsize=font_size)

            if x_range is not None and i < len(x_range):
                ax.set_xlim(x_range[i][0], x_range[i][1])
            if y_range is not None and i < len(y_range):
                ax.set_ylim(y_range[i][0], y_range[i][1])

    for i in range(len(axes)):
        fig, ax = axes[i]
        filename = f"{args.dir}/{prefix}{headers[i]}_ppd_plot"
        if len(axes) > 1:
            filename += f"_{i}.pdf"
        else:
            filename += ".pdf"

        if args.grid:
            ax.grid(
                args.grid,
                which=args.grid_which,
                linestyle=args.grid_linestyle,
                alpha=args.grid_alpha,
            )
        ax.legend()
        fig.savefig(filename, dpi=args.dpi, bbox_inches="tight")
    plt.close("all")
