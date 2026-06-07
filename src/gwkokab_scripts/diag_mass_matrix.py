# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def main():
    """Compute diagonal mass matrix from pilot-run samples."""
    import argparse
    from argparse import ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(
        description="Compute diagonal mass matrix from pilot-run samples",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Path to pilot-run .hdf5 file")
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Regularization for tiny std values (default 1e-12)",
    )

    args = parser.parse_args()

    import h5py
    import numpy as np

    with h5py.File(args.filename) as f:
        samples = f["samples"][:]

    samples = np.atleast_2d(samples)

    if samples.shape[0] < 2:
        raise RuntimeError(
            "At least 2 samples are required to compute standard deviation."
        )

    # Compute per-dimension std
    sigma = np.std(samples, axis=0, ddof=1)

    # Convert to condition matrix
    condition_matrix = np.reciprocal(np.square(sigma) + args.eps)

    # Write one-line comma-separated numbers
    values = ", ".join(f"{v:.8g}" for v in condition_matrix)

    print(values)
