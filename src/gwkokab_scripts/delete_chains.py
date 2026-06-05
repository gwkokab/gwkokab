# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import h5py


def _delete_chains(f: h5py.File, chains_idx: list[int], n_chains: int) -> None:
    import numpy as np

    from gwkokab.analysis.utils.literals import SAMPLES_GROUP_NAME

    samples = f[SAMPLES_GROUP_NAME][()]
    del f[SAMPLES_GROUP_NAME]

    cleaned_chains_flatten = np.vstack([
        chain
        for idx, chain in enumerate(np.array_split(samples, n_chains, axis=0))
        if idx not in chains_idx
    ])

    f.create_dataset(SAMPLES_GROUP_NAME, data=cleaned_chains_flatten)


def _dc_sampler_is_numpyro(f: h5py.File, chains_idx: list[int]) -> None:
    n_chains = f["/sampler_cfg/mcmc"].attrs["num_chains"]
    f["/sampler_cfg/mcmc"].attrs["num_chains"] = n_chains - len(chains_idx)
    chains = [
        f["/chains/chain_{}".format(idx)][()]
        for idx in range(n_chains)
        if idx not in chains_idx
    ]

    from gwkokab.analysis.core.utils import write_to_hdf5

    del f["/chains"]

    for idx, chain in enumerate(chains):
        write_to_hdf5(f, f"/chains/chain_{idx}", chain)


def _dc_sampler_is_flowMC(f: h5py.File, chains_idx: list[int]) -> None:
    n_chains = f["sampler_cfg"].attrs["n_chains"]
    f["sampler_cfg"].attrs["n_chains"] = n_chains - len(chains_idx)

    from gwkokab.analysis.core.utils import write_to_hdf5

    for phase in ("train", "prod"):
        chains = [
            f[f"/chains/{phase}/chain_{idx}"]
            for idx in range(n_chains)
            if idx not in chains_idx
        ]

        del f[f"/chains/{phase}"]

        for idx, chain in enumerate(chains):
            group_name = f"/chains/{phase}/chain_{idx}"
            f.create_group(group_name)
            for key in chain.keys():
                write_to_hdf5(f, f"{group_name}/{key}", chain[key][()])


def _delete_chains_factory(f: h5py.File, chains_idx: list[int]) -> None:
    n_chains = 0
    sampler_name = f["sampler_cfg"].attrs["sampler_name"]

    if sampler_name == "numpyro":
        n_chains = len(f["chains"].keys())
        _dc_sampler_is_numpyro(f, chains_idx)
    elif sampler_name == "flowMC":
        n_chains = int(f["sampler_cfg"].attrs["n_chains"])
        _dc_sampler_is_flowMC(f, chains_idx)
    else:
        raise ValueError(f"Unrecognized Sampler: {sampler_name}")

    _delete_chains(f, chains_idx, n_chains)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Delete chains from a given HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--numbers",
        nargs="+",
        type=int,
        required=True,
        help="List of chain numbers to delete. Chain numbering starts from 0.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Name of the input HDF5 file containing the chains to be deleted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Name of the output HDF5 file where the modified chains will be saved.",
    )

    args = parser.parse_args()

    import os

    env_vars = {
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    for var, value in env_vars.items():
        os.environ[var] = value

    import shutil

    input_filename = args.input
    output_filename = args.output
    numbers = args.numbers

    shutil.copy(input_filename, output_filename)

    with h5py.File(output_filename, "a") as f:
        _delete_chains_factory(f, numbers)
