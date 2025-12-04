# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute diagonal mass matrix from pilot-run samples"
    )
    parser.add_argument(
        "filename", help="Path to pilot-run .dat file (rows=samples, columns=params)"
    )
    parser.add_argument(
        "--delimiter",
        default=" ",
        help="Delimiter used in the .dat file (default: space)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Regularization for tiny std values (default 1e-12)",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Number of rows to skip at the beginning of the file (default 0)",
    )

    args = parser.parse_args()

    import numpy as np

    # Load samples
    samples = np.loadtxt(
        args.filename, delimiter=args.delimiter, skiprows=args.skip_rows
    )
    samples = np.atleast_2d(samples)

    if samples.ndim != 2:
        raise RuntimeError(f"Expected 2D samples array, got shape {samples.shape}")

    # Compute per-dimension std
    sigma = np.std(samples, axis=0, ddof=1)

    # Convert to condition matrix
    condition_matrix = np.reciprocal(np.square(sigma) + args.eps)

    # Write one-line comma-separated numbers
    values = ", ".join(f"{v:.8g}" for v in condition_matrix)

    print(values)
