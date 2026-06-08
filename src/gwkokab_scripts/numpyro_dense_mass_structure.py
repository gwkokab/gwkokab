# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Suggest dense_mass parameter blocks from posterior-sample correlations. "
            "Useful when posterior samples are available but the model geometry is not "
            "well understood. The suggested blocks can be used as an initial dense_mass "
            "configuration for NumPyro/NUTS. If the model geometry and parameter "
            "dependencies are already known, manually specifying dense_mass blocks is "
            "generally preferred."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "posterior_file",
        type=str,
        help=(
            "Path to the HDF5 file containing posterior samples. The file should have a "
            "dataset named 'samples' with shape (num_samples, num_parameters) and an "
            "attribute 'variables_index' mapping parameter names to their column indices."
        ),
    )

    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.5,
        help="Minimum absolute correlation to consider parameters as strongly correlated"
        " and suggest them for the same dense_mass block.",
    )

    parser.add_argument(
        "--min-block-size",
        type=int,
        default=2,
        help="Minimum number of parameters in a cluster to be suggested as a dense_mass"
        " block. Clusters with fewer parameters will be ignored.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    from collections import defaultdict

    import h5py
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    with h5py.File(args.posterior_file, "r") as f:
        samples = np.asarray(f["samples"])
        var_index: dict[str, int] = dict(f["variables_index"].attrs)

    index_to_names = defaultdict(list)

    for name, idx in var_index.items():
        index_to_names[int(idx)].append(name)

    good = []

    for i in range(samples.shape[1]):
        x = samples[:, i]
        good.append(np.all(np.isfinite(x)) and np.std(x) > 0)

    recovered_params = []
    for i in range(samples.shape[1]):
        if not good[i]:
            continue
        names = index_to_names.get(i)
        canonical_name = sorted(names)[0] if names else f"unnamed_param_{i}"
        recovered_params.append(canonical_name)
    good = np.array(good)
    samples = samples[:, good]

    corr = np.nan_to_num(np.corrcoef(samples.T), nan=0.0)

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
