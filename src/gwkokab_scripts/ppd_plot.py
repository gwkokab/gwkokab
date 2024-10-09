# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import argparse
from typing_extensions import Any, List, Tuple

import h5py
import numpy as np
from jaxtyping import Float, Int
from matplotlib import pyplot as plt

from kokab.utils.ppd import get_all_marginals


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPD plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots ppd plots.",
    )
    parser.add_argument(
        "--data",
        help="data file path. Only .hdf5 files are supported.",
        required=True,
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--prefix",
        help="prefix for the output file",
        type=str,
    )
    parser.add_argument(
        "--size",
        help="size of the corner plot in inches",
        nargs=2,
        default=(10, 10),
        type=float,
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
        "--x-range",
        nargs=2,
        action="append",
        type=float,
        help="range of the x axis plot in the form of start end for each parameter",
    )
    parser.add_argument(
        "--y-range",
        nargs=2,
        action="append",
        type=float,
        help="range of the y axis plot in the form of start end for each parameter",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="directory to save the plots",
    )

    return parser


def get_domain(
    domain_as_read: Any,
) -> Tuple[Float[float, ""], Float[float, ""], Int[int, ""]]:
    return [(float(s), float(e), int(n)) for s, e, n in domain_as_read]


def get_headers(headers_as_read: Any) -> List[str]:
    return [h.decode("utf-8") for h in headers_as_read]


def get_quantiles(
    data: Float[np.ndarray, "..."], quantiles: Float[np.ndarray, "..."]
) -> Float[np.ndarray, "..."]:
    return np.quantile(data, quantiles, axis=1)


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    prefix = "" if args.prefix is None else args.prefix

    x_range = args.x_range
    y_range = args.y_range

    with h5py.File(args.data.name, "r") as f:
        domains = get_domain(f["domains"])
        headers = get_headers(f["headers"])
        marginals = [f["marginals"][head][:] for head in headers]
        ppd = f["ppd"][()]

    marginal_ppds = get_all_marginals(ppd, domains)

    i = 0
    for head, marginal, marginal_ppd, domain in zip(
        headers, marginals, marginal_ppds, domains
    ):
        start, end, num_points = domain
        quant = get_quantiles(marginal, [0.05, 0.25, 0.5, 0.75, 0.95])
        xx = np.linspace(start, end, num_points)

        fig, ax = plt.subplots(figsize=args.size)
        ax.plot(
            xx, quant[2], label="median", color="#7B8794", alpha=0.8, linestyle="--"
        )
        ax.plot(xx, marginal_ppd, label="PPD", color="#FF6F61")
        ax.fill_between(
            xx, quant[0], quant[-1], alpha=0.5, label="90% CI", color="#76C7C0"
        )
        ax.fill_between(
            xx, quant[1], quant[-2], alpha=0.7, label="50% CI", color="#8FD694"
        )
        ax.set_yscale(args.y_scale)
        ax.set_xscale(args.x_scale)
        ax.set_title(f"PPD plot of {prefix}{head}")
        ax.set_xlabel(head)
        ax.set_ylabel(f"ppd({head})")
        if x_range is not None:
            if i < len(x_range):
                ax.set_xlim(x_range[i][0], x_range[i][1])
        if y_range is not None:
            if i < len(y_range):
                ax.set_ylim(y_range[i][0], y_range[i][1])

        plt.legend()
        plt.tight_layout()
        fig.savefig(f"{args.dir}/{prefix}{head}_ppd_plot.png")
        plt.close("all")
        i += 1
