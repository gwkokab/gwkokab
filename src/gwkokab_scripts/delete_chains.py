# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import re

import h5py


def _parse_chain_names(chains: list[str]) -> list[str]:
    """Parse chain identifiers supplied as names like chain_0."""
    chain_names: list[str] = []
    for chain in chains:
        value = str(chain).strip()
        if re.fullmatch(r"chain_\d+", value):
            chain_names.append(value)
            continue

        raise ValueError(
            f"Invalid chain identifier '{chain}'. Use the full chain name, for example chain_0."
        )
    return chain_names


def _delete_chains(
    f: h5py.File,
    chains_to_delete: list[str],
    chain_names: list[str],
) -> None:
    import numpy as np

    from gwkokab.analysis.utils.literals import SAMPLES_GROUP_NAME

    samples = f[SAMPLES_GROUP_NAME][()]
    del f[SAMPLES_GROUP_NAME]

    n_chains = len(chain_names)
    cleaned_chains_flatten = np.vstack([
        chain
        for chain_name, chain in zip(
            chain_names, np.array_split(samples, n_chains, axis=0), strict=True
        )
        if chain_name not in chains_to_delete
    ])

    f.create_dataset(SAMPLES_GROUP_NAME, data=cleaned_chains_flatten)


def _dc_sampler_is_numpyro(f: h5py.File, chains_to_delete: list[str]) -> list[str]:
    chain_names = sorted(f["/chains"].keys(), key=lambda name: int(name.split("_")[-1]))
    remaining_chain_names = [
        chain_name for chain_name in chain_names if chain_name not in chains_to_delete
    ]

    for chain_name in chains_to_delete:
        del f[f"/chains/{chain_name}"]

    f["/sampler_cfg/mcmc"].attrs["num_chains"] = len(remaining_chain_names)

    return chain_names


def _dc_sampler_is_flowMC(f: h5py.File, chains_to_delete: list[str]) -> list[str]:
    phase_chain_names: list[list[str]] = []

    for phase in ("train", "prod"):
        phase_group = f[f"/chains/{phase}"]
        chain_names = sorted(
            phase_group.keys(), key=lambda name: int(name.split("_")[-1])
        )
        phase_chain_names.append(chain_names)

    if phase_chain_names[0] != phase_chain_names[1]:
        raise ValueError(
            f"flowMC train/prod chain names do not match: {phase_chain_names}"
        )

    chain_names = phase_chain_names[0]
    remaining_chain_names = [
        chain_name for chain_name in chain_names if chain_name not in chains_to_delete
    ]

    for phase in ("train", "prod"):
        for chain_name in chains_to_delete:
            del f[f"/chains/{phase}/{chain_name}"]

    f["sampler_cfg"].attrs["n_chains"] = len(remaining_chain_names)

    return chain_names


def _delete_chains_factory(f: h5py.File, chains_to_delete: list[str]) -> None:
    sampler_name = f["sampler_cfg"].attrs["sampler_name"]

    if sampler_name == "numpyro":
        chain_names = sorted(
            f["/chains"].keys(), key=lambda name: int(name.split("_")[-1])
        )
    elif sampler_name == "flowMC":
        phase_chain_names = [
            sorted(
                f[f"/chains/{phase}"].keys(), key=lambda name: int(name.split("_")[-1])
            )
            for phase in ("train", "prod")
        ]
        if phase_chain_names[0] != phase_chain_names[1]:
            raise ValueError(
                f"flowMC train/prod chain names do not match: {phase_chain_names}"
            )
        chain_names = phase_chain_names[0]
    else:
        raise ValueError(f"Unrecognized Sampler: {sampler_name}")

    unique_chains_to_delete = set(chains_to_delete)
    if not unique_chains_to_delete:
        raise ValueError("No chain names specified for deletion.")
    missing_chains = sorted(unique_chains_to_delete - set(chain_names))
    if missing_chains:
        raise ValueError(
            f"Requested chains do not exist: {missing_chains}. Available chains: {chain_names}"
        )
    if len(unique_chains_to_delete) >= len(chain_names):
        raise ValueError(
            f"Cannot delete all chains. Total chains: {len(chain_names)}, requested to delete: {len(unique_chains_to_delete)}"
        )

    if sampler_name == "numpyro":
        original_chain_names = _dc_sampler_is_numpyro(f, chains_to_delete)
    elif sampler_name == "flowMC":
        original_chain_names = _dc_sampler_is_flowMC(f, chains_to_delete)

    _delete_chains(f, chains_to_delete, original_chain_names)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Delete chains from a given HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--chains",
        nargs="+",
        type=str,
        required=True,
        help="List of chains to delete. Use full names like chain_0 chain_2.",
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Name of the input HDF5 file containing the chains to be deleted.",
    )
    parser.add_argument(
        "-o",
        type=str,
        default="clean_data.hdf5",
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

    input_filename = args.i
    output_filename = args.o
    chains_to_delete = _parse_chain_names(args.chains)

    shutil.copy(input_filename, output_filename)

    with h5py.File(output_filename, "a") as f:
        _delete_chains_factory(f, chains_to_delete)
