# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def trace_plot(
    filenames,
    labels=None,
    output="trace_plot.png",
    dpi=300,
    alpha=0.3,
    size=0.5,
    width=8,
    height_per_dim=2,
):
    import numpy as np

    # Load data
    chains = np.stack([np.loadtxt(f, skiprows=1) for f in filenames], axis=0)

    del np

    n_chains, n_samples, n_dims = chains.shape

    # Validation for labels
    if labels and len(labels) != n_dims:
        import warnings

        warnings.warn(
            f"Warning: {len(labels)} labels provided for {n_dims} dimensions. Ignoring labels."
        )
        labels = None

    import glasbey

    chains_colors = glasbey.create_palette(n_chains)

    del glasbey

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(
        n_dims,
        1,
        figsize=(width, height_per_dim * n_dims),
        sharex=True,
        constrained_layout=True,
    )

    # Handle single dimension case
    if n_dims == 1:
        axs = [axs]

    for n_dim in range(n_dims):
        ax = axs[n_dim]  # type: ignore
        ax.set_xlim(0, n_samples)

        for n_chain in range(n_chains):
            # Using plot with markers is significantly faster than scatter for large MCMC chains
            ax.plot(
                chains[n_chain, :, n_dim],
                color=chains_colors[n_chain],
                alpha=alpha,
                marker="o",
                markersize=size,
                linestyle="None",  # Keep it as points, or change to '-' for lines
            )

        if labels:
            ax.set_ylabel(labels[n_dim])
        else:
            ax.set_ylabel(f"Dim {n_dim}")

    axs[-1].set_xlabel("Iteration")  # type: ignore

    fig.savefig(output, dpi=dpi, bbox_inches="tight")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate MCMC trace plots from .dat files."
    )

    # File & Label Arguments
    parser.add_argument(
        "filenames", nargs="+", help="Space-separated list of .dat files."
    )
    parser.add_argument(
        "--labels", nargs="+", help="List of LaTeX labels for the y-axes."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="trace_plot.png",
        help="Output filename for the plot.",
    )

    # Aesthetic Arguments
    parser.add_argument("--dpi", type=int, default=300, help="DPI of the output image.")
    parser.add_argument(
        "--alpha", type=float, default=0.3, help="Opacity of the points (0 to 1)."
    )
    parser.add_argument("--size", type=float, default=0.5, help="Size of the points.")
    parser.add_argument(
        "--width", type=float, default=8.0, help="Width of the figure in inches."
    )
    parser.add_argument(
        "--height-per-dim",
        type=float,
        default=2.0,
        help="Height per subplot dimension.",
    )

    args = parser.parse_args()

    trace_plot(
        filenames=args.filenames,
        labels=args.labels,
        output=args.output,
        dpi=args.dpi,
        alpha=args.alpha,
        size=args.size,
        width=args.width,
        height_per_dim=args.height_per_dim,
    )
