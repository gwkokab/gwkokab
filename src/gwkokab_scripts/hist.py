# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Plots a histogram by reading data from HDF5 files. "
            "It assumes that the data available at the dataset path is a "
            "compound array, where the '--variable' argument specifies the field to plot."
        )
    )
    parser.add_argument(
        "-i",
        "--input-files",
        nargs="+",
        required=True,
        help="Path to one or more HDF5 input files.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=str,
        help="Path of the compound dataset inside the HDF5 files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to save the output plot file.",
    )
    parser.add_argument(
        "-v",
        "--variable",
        required=True,
        type=str,
        help="Name of the field/variable within the compound array to be plotted.",
    )
    parser.add_argument(
        "--bins",
        default=30,
        type=int,
        help="Number of bins for histogram.",
    )
    parser.add_argument(
        "-l",
        "--xlabel",
        type=str,
        help="Custom label for the x-axis.",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Transparency for each histogram layer (0.0 = transparent, 1.0 = opaque).",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot probability density instead of raw frequency counts.",
    )

    args = parser.parse_args()

    import h5py
    import numpy as np
    from matplotlib import pyplot as plt
    import glasbey

    plt.rcParams["figure.constrained_layout.use"] = True

    dataset = args.dataset
    variable = args.variable

    hist_kwargs = {
        "density": args.density,
        "alpha": args.alpha,
        "bins": args.bins,
    }

    colors = glasbey.create_palette(len(args.input_files), optimize_palette=False)

    for color, filename in zip(colors, args.input_files):
        with h5py.File(filename, "r") as f:
            data = np.asarray(f[dataset][variable])
        plt.hist(data, **hist_kwargs, color=color)

    # Fallback logic using the new argument naming scheme
    x_label: str = args.xlabel or args.variable
    y_label = "Density" if hist_kwargs["density"] else "Frequency"

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(args.output, dpi=200)
