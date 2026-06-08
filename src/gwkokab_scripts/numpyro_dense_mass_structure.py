#!/usr/bin/env python3

import argparse
from collections import defaultdict

import h5py
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


DEFAULT_H5_FILE = "./inference_data.hdf5"
DEFAULT_SAMPLES_PATH = "/samples"
DEFAULT_VARIABLES_INDEX_PATH = "/variables_index"
DEFAULT_CORR_THRESHOLD = 0.5
DEFAULT_MIN_BLOCK_SIZE = 2


def source_label(args, name):
    return "user" if name in args._provided else "default"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Suggest dense_mass parameter blocks from posterior-sample correlations. "
            "Useful when posterior samples are available but the model geometry is not "
            "well understood. The suggested blocks can be used as an initial dense_mass "
            "configuration for NumPyro/NUTS. If the model geometry and parameter "
            "dependencies are already known, manually specifying dense_mass blocks is "
            "generally preferred."
        )
    )
    parser.add_argument(
        "--posterior-file",
        default=DEFAULT_H5_FILE,
        help=f"HDF5 inference file containing posteriors. Default: {DEFAULT_H5_FILE}",
    )

    parser.add_argument(
        "--samples-path",
        default=DEFAULT_SAMPLES_PATH,
        help=f"path to samples dataset in posterior h5 file. Default: {DEFAULT_SAMPLES_PATH}",
    )

    parser.add_argument(
        "--variables-index-path",
        default=DEFAULT_VARIABLES_INDEX_PATH,
        help=f"path to variables_index group in posteriro h5 file. Default: {DEFAULT_VARIABLES_INDEX_PATH}",
    )

    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=DEFAULT_CORR_THRESHOLD,
        help=f"Absolute correlation threshold. Default: {DEFAULT_CORR_THRESHOLD}",
    )

    parser.add_argument(
        "--min-block-size",
        type=int,
        default=DEFAULT_MIN_BLOCK_SIZE,
        help=f"Minimum number of parameters in a dense block. Default: {DEFAULT_MIN_BLOCK_SIZE}",
    )

    args = parser.parse_args()

    provided = set()

    args._provided = provided
    return args


def main():
    args = parse_args()

    print("\nDense mass suggestion settings:")
    print(
        f"  posterior_file              = {args.posterior_file} ({source_label(args, 'posterior_file')})"
    )
    print(
        f"  samples_path in posterior_file         = {args.samples_path} ({source_label(args, 'samples_path')})"
    )
    print(
        f"  variables_index_path = {args.variables_index_path} ({source_label(args, 'variables_index_path')})"
    )
    print(
        f"  correlation_threshold       = {args.corr_threshold} ({source_label(args, 'corr_threshold')})"
    )
    print(
        f"  min_block_size       = {args.min_block_size} ({source_label(args, 'min_block_size')})"
    )

    with h5py.File(args.posterior_file, "r") as f:
        samples = np.asarray(f[args.samples_path])
        var_index = dict(f[args.variables_index_path].attrs)

    if samples.ndim > 2:
        samples = samples.reshape(-1, samples.shape[-1])

    if samples.shape[0] < samples.shape[1]:
        samples = samples.T

    index_to_names = defaultdict(list)

    for name, idx in var_index.items():
        index_to_names[int(idx)].append(name)

    recovered_params = []
    column_indices = []

    for idx in sorted(index_to_names):
        names = sorted(index_to_names[idx])
        canonical_name = names[0]

        recovered_params.append(canonical_name)
        column_indices.append(idx)

    data = samples[:, column_indices]

    good = []

    for i in range(data.shape[1]):
        x = data[:, i]
        good.append(np.all(np.isfinite(x)) and np.std(x) > 0)

    good = np.array(good)

    data = data[:, good]
    recovered_params = [p for p, keep in zip(recovered_params, good) if keep]

    corr = np.corrcoef(data.T)
    corr = np.nan_to_num(corr, nan=0.0)

    distance = 1.0 - np.abs(corr)
    distance = 0.5 * (distance + distance.T)
    distance = np.clip(distance, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    Z = linkage(squareform(distance, checks=False), method="average")

    clusters = fcluster(
        Z,
        t=1.0 - args.corr_threshold,
        criterion="distance",
    )

    cluster_to_params = defaultdict(list)

    for p, c in zip(recovered_params, clusters):
        cluster_to_params[int(c)].append(p)

    dense_mass = [
        block
        for block in cluster_to_params.values()
        if len(block) >= args.min_block_size
    ]

    param_order = {p: i for i, p in enumerate(recovered_params)}

    dense_mass = sorted(
        dense_mass,
        key=lambda block: min(param_order[p] for p in block),
    )

    print("\nList of parameters in posterior file:\n")
    for p in recovered_params:
        print(p)

    print("[")

    lines = ["    [" + ", ".join(f'"{p}"' for p in block) + "]" for block in dense_mass]
    print(",\n".join(lines))
    print("]")

    print("\nStrong pairwise correlations:\n")
    for i, pi in enumerate(recovered_params):
        for j in range(i + 1, len(recovered_params)):
            pj = recovered_params[j]
            cij = corr[i, j]

            if abs(cij) >= args.corr_threshold:
                print(f"{pi:35s} {pj:35s} corr = {cij: .3f}")


if __name__ == "__main__":
    main()
