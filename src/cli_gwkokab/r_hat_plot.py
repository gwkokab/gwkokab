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


import argparse
from glob import glob

import arviz as az
import numpy as np
from arviz.utils import _var_names, get_coords
from matplotlib import pyplot as plt


def make_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    :return: the command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Pearson correlation coefficient plotter. "
        "source: https://github.com/kazewong/flowMC/blob/main/example/notebook/dualmoon.ipynb (last cell)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots a Pearson correlation coefficients of the chains.",
    )
    parser.add_argument(
        "--chains-regex",
        help="regex pattern for the files containing chains. Only .dat files are supported.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output",
        help="output file path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--n-split",
        help="number of splits",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--title",
        help="title of the plot",
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
        "--labels",
        help="labels of the chains",
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

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    plt.rcParams.update({"text.usetex": args.use_latex})
    if args.font_family is not None:
        plt.rcParams.update({"font.family": args.font_family})

    chains_filenames = glob(args.chains_regex)
    chains = np.array([np.loadtxt(filename) for filename in chains_filenames])

    ## Load data as arviz InferenceData class
    idata = az.convert_to_inference_data(chains)
    coords = {}
    data = get_coords(az.convert_to_dataset(idata, group="posterior"), coords)
    var_names = None
    filter_vars = None
    var_names = _var_names(var_names, data, filter_vars)
    n_draws = data.sizes["draw"]

    first_draw = data.draw.values[0]  # int of where where things should start

    ## Compute where to split the data to diagnostic the convergence
    n_split = args.n_split
    draw_divisions = np.linspace(n_draws // n_split, n_draws, n_split, dtype=int)

    rhat_s = np.stack(
        [
            np.array(
                az.rhat(
                    data.sel(draw=slice(first_draw + draw_div)),
                    var_names=var_names,
                    method="rank",
                )["x"]
            )
            for draw_div in draw_divisions
        ]
    )
    labels = args.labels if args.labels is not None else chains_filenames

    plt.plot(draw_divisions, rhat_s, "-o", label=labels)
    plt.axhline(1, c="k", ls="--")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\hat{R}$")
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fancybox=True,
        shadow=True,
    )
    plt.yscale(args.y_scale)
    plt.xscale(args.x_scale)
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
