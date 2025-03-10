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
import glob

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command line interface for plotting Effective Sample Size per draw.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-chain-regex",
        help="regex pattern for the train chains files. Only .dat files are supported.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--production-chain-regex",
        help="regex pattern for the production chains files. Only .dat files are supported.",
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
        "--labels",
        nargs="+",
        help="labels for the chains",
        required=True,
    )
    parser.add_argument(
        "--width",
        help="width of the plot in inches",
        default=12,
        type=int,
    )
    parser.add_argument(
        "--height",
        help="height of the plot in inches",
        default=6,
        type=int,
    )
    parser.add_argument(
        "--use-latex",
        help="use LaTeX for rendering text",
        action="store_true",
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    plt.rcParams.update({"text.usetex": args.use_latex})

    train_paths = glob.glob(args.train_chain_regex)
    prod_paths = glob.glob(args.production_chain_regex)

    train_chains = []

    for file_path in train_paths:
        chain = np.loadtxt(file_path)
        train_chains.append(chain)

    train_chains = np.array(train_chains)

    train_data = az.convert_to_inference_data(train_chains)

    prod_chains = []

    for file_path in prod_paths:
        chain = np.loadtxt(file_path)
        prod_chains.append(chain)

    prod_chains = np.array(prod_chains)

    prod_data = az.convert_to_inference_data(prod_chains)

    ess_train = az.ess(train_data)
    ess_prod = az.ess(prod_data)

    n_draws = train_data.posterior.dims["draw"]

    sns.set_theme(style="whitegrid")
    labels = args.labels
    df = pd.DataFrame(
        data={
            "Parameter": labels,
            "Training": ess_train.to_array().data.squeeze() / n_draws,
            "Production": ess_prod.to_array().data.squeeze() / n_draws,
        }
    )

    height, width = args.height, args.width
    aspect = width / height

    ax = sns.catplot(
        data=df.melt(id_vars="Parameter"),
        x="Parameter",
        y="value",
        hue="variable",
        kind="bar",
        height=height,
        aspect=aspect,
        margin_titles=True,
    )
    ax.set_xticklabels(rotation=45)
    ax.set_axis_labels("Parameter", "ESS per draw")
    ax.legend.draw_frame(True)
    ax.legend.set_title("")
    ax.legend.set_bbox_to_anchor((1, 0.9))

    sub_plot = ax.facet_axis(0, 0)

    for c in sub_plot.containers:
        labels = [f"{h.get_height():.3f}" for h in c]
        sub_plot.bar_label(c, labels=labels, label_type="edge")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
