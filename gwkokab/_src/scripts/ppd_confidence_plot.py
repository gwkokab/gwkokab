#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

import argparse

import jax
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt
from numpyro.distributions import *

from ..models import *


def make_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    :return: the command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="posterior predictive distributions plotter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This script plots a posterior predictive distributions plot.",
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
        "-m",
        "--model",
        help="model name",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-a",
        "--args-column-index",
        nargs="+",
        help="model arguments column index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-x",
        "--x-limit",
        help="x-axis limit",
        required=True,
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "-xs",
        "--x-step",
        help="x-axis step",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-q",
        "--quantiles",
        nargs="+",
        help="quantiles for the confidence plot",
        type=float,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--fill-between",
        nargs="+",
        help="fill between quantiles",
        type=int,
        action="append",
        required=True,
    )
    parser.add_argument(
        "-xlog",
        "--x-log-scale",
        help="x-axis on log scale",
        action="store_true",
    )
    parser.add_argument(
        "-ylog",
        "--y-log-scale",
        help="y-axis on log scale",
        action="store_true",
    )
    parser.add_argument(
        "-xl",
        "--x-label",
        help="x-axis label",
        type=str,
    )
    parser.add_argument(
        "-yl",
        "--y-label",
        help="y-axis label",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--title",
        help="plot title",
        type=str,
    )
    parser.add_argument(
        "-fw",
        "--fig-width",
        help="figure width in inches",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-fh",
        "--fig-height",
        help="figure height",
        type=int,
        default=7,
    )

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    data = np.loadtxt(args.data.name)

    model_name = args.model

    xx_limit = args.x_limit
    steps = args.x_step
    if args.y_log_scale:
        plt.yscale("log")
    if args.x_log_scale:
        plt.xscale("log")
        xx = np.logspace(np.log10(xx_limit[0]), np.log10(xx_limit[1]), steps)
    else:
        xx = np.linspace(xx_limit[0], xx_limit[1], steps)

    def model_fn(*args):
        model = eval(model_name)(*args)
        model._validate_args = True
        prob = jnp.exp(model.log_prob(xx))
        return prob

    args_column_index = args.args_column_index
    yy = jax.vmap(model_fn)(*tuple(data[:, i] for i in args_column_index))
    ppd = np.mean(yy, axis=0)
    plt.plot(xx, ppd, linestyle="--", label="Posterior Predictive Distribution")
    del ppd
    del yy

    model: Distribution = eval(model_name)(
        *tuple(data[:, i] for i in args_column_index)
    )
    model._validate_args = True
    xx_grid = np.repeat(xx[:, None], len(data), axis=1)
    yy = np.exp(model.log_prob(xx_grid))
    q = np.quantile(yy, args.quantiles, axis=1)
    for ends in args.fill_between:
        if len(ends) == 1:
            plt.plot(xx, q[ends[0]], label=f"Quantile {args.quantiles[ends[0]]*100}%")
        elif len(ends) == 2:
            plt.fill_between(
                xx,
                q[ends[0]],
                q[ends[1]],
                alpha=0.5,
                label=f"CI {(args.quantiles[ends[1]] - args.quantiles[ends[0]])*100}%",
            )

    if args.x_label:
        plt.xlabel(args.x_label)
    if args.y_label:
        plt.ylabel(args.y_label)
    if args.title:
        plt.title(args.title)

    plt.legend()
    plt.gcf().set_size_inches(args.fig_width, args.fig_height)
    plt.tight_layout()
    plt.savefig(args.output.name)
