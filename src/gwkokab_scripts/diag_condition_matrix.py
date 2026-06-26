# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def main():
    """Compute diagonal condition matrix from pilot-run samples."""
    import argparse
    from argparse import ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(
        description="Compute diagonal condition matrix from pilot-run samples",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Path to pilot-run .hdf5 file")

    args = parser.parse_args()

    import sys

    import h5py
    import numpy as np

    with h5py.File(args.filename) as f:
        samples = f["samples"][:]

    samples = np.atleast_2d(samples)

    if samples.shape[0] < 2:
        raise RuntimeError(
            "At least 2 samples are required to compute standard deviation."
        )

    condition_matrix = np.var(samples, axis=0, ddof=1)

    values = ", ".join(f"{v:.8g}" for v in condition_matrix)
    sys.stdout.write(f"Diagonal condition matrix: [{values}]\n")
