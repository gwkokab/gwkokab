# Copyright 2026 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Plots a 2D scatter plot by reading data from HDF5 files. "
            "It assumes that the data available at the dataset path is a "
            "compound array, where the '--x-variable' and '--y-variable' "
            "arguments specify the fields to plot against each other."
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
        "-x",
        "--x-variable",
        required=True,
        type=str,
        help="Field name within the compound array for the X-axis data.",
    )
    parser.add_argument(
        "-y",
        "--y-variable",
        required=True,
        type=str,
        help="Field name within the compound array for the Y-axis data.",
    )
    parser.add_argument(
        "-xl",
        "--xlabel",
        type=str,
        help="Custom label for the x-axis (defaults to --x-variable).",
    )
    parser.add_argument(
        "-yl",
        "--ylabel",
        type=str,
        help="Custom label for the y-axis (defaults to --y-variable).",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.7,
        help="Transparency for the scatter points (0.0 = transparent, 1.0 = opaque).",
    )
    parser.add_argument(
        "-s",
        "--marker-size",
        type=float,
        default=20.0,
        help="Size of the scatter plot markers.",
    )

    args = parser.parse_args()

    import h5py
    import numpy as np
    from matplotlib import pyplot as plt
    import glasbey

    plt.rcParams["figure.constrained_layout.use"] = True

    scatter_kwargs = {
        "alpha": args.alpha,
        "s": args.marker_size,
    }

    colors = glasbey.create_palette(len(args.input_files), optimize_palette=False)

    for color, filename in zip(colors, args.input_files):
        with h5py.File(filename, "r") as f:
            compound_dataset = f[args.dataset]
            x_data = np.asarray(compound_dataset[args.x_variable])
            y_data = np.asarray(compound_dataset[args.y_variable])

        plt.scatter(x_data, y_data, **scatter_kwargs, color=color)

    x_label: str = args.xlabel or args.x_variable
    y_label: str = args.ylabel or args.y_variable

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(args.output, dpi=200)
