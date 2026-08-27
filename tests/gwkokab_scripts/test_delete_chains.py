# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`gwkokab_scripts.delete_chains`.

The script surgically removes chains from a finished ``inference_data.hdf5``: it drops
the per-chain groups, rewrites the flattened ``samples`` dataset without the deleted
chains' rows, and corrects the chain count recorded in ``sampler_cfg``. The two sampler
backends store chains under different paths, so both layouts are exercised.

Every chain is written with all of its rows equal to its own index, which makes the
surviving rows of ``samples`` traceable back to the chain they came from.
"""

import os

import h5py
import numpy as np
import pytest

from gwkokab_scripts import delete_chains
from gwkokab_scripts.delete_chains import _delete_chains_factory, _parse_chain_names


N_SAMPLES = 4
N_DIMS = 2


@pytest.fixture(autouse=True)
def _restore_jax_env(monkeypatch):
    """``main`` writes JAX environment variables straight into ``os.environ``.

    Re-setting them through monkeypatch first makes pytest restore the original values
    (or remove them again) once the test is over.
    """
    for var in (
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    ):
        monkeypatch.setenv(var, os.environ.get(var, ""))


def _chain_rows(chain_id: int) -> np.ndarray:
    return np.full((N_SAMPLES, N_DIMS), float(chain_id))


def _flat_samples(chain_ids) -> np.ndarray:
    return np.vstack([_chain_rows(chain_id) for chain_id in chain_ids])


def _write_numpyro_file(path, n_chains: int = 3) -> str:
    """An ``inference_data.hdf5`` as ``numpyro_base`` leaves it: one dataset per chain
    directly under ``/chains`` and the chain count on ``/sampler_cfg/mcmc``.
    """
    path = str(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=_flat_samples(range(n_chains)))
        for chain_id in range(n_chains):
            f.create_dataset(f"chains/chain_{chain_id}", data=_chain_rows(chain_id))
        f.require_group("sampler_cfg").attrs["sampler_name"] = "numpyro"
        f.require_group("sampler_cfg/mcmc").attrs["num_chains"] = n_chains
    return path


def _write_flowMC_file(path, n_chains: int = 3, prod_chain_ids=None) -> str:
    """An ``inference_data.hdf5`` as ``flowMC_base`` leaves it: a ``positions`` and a
    ``log_probs`` dataset per chain, under both the ``train`` and the ``prod`` phase.
    """
    path = str(path)
    phases = {"train": range(n_chains), "prod": prod_chain_ids or range(n_chains)}
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=_flat_samples(range(n_chains)))
        for phase, chain_ids in phases.items():
            for chain_id in chain_ids:
                group = f.require_group(f"chains/{phase}/chain_{chain_id}")
                group.create_dataset("positions", data=_chain_rows(chain_id))
                group.create_dataset("log_probs", data=np.full(N_SAMPLES, chain_id))
        f.require_group("sampler_cfg").attrs.update({
            "sampler_name": "flowMC",
            "n_chains": n_chains,
        })
    return path


def _surviving_chain_ids(path) -> set:
    with h5py.File(path, "r") as f:
        return set(np.unique(f["samples"][()]).astype(int).tolist())


# ---------------------------------------------------------------------------
# _parse_chain_names
# ---------------------------------------------------------------------------


def test_parse_chain_names_accepts_full_names_and_strips_whitespace():
    assert _parse_chain_names(["chain_0", "  chain_12\n"]) == ["chain_0", "chain_12"]


def test_parse_chain_names_preserves_the_given_order():
    """The order matters only for the error messages, but it should not be shuffled."""
    assert _parse_chain_names(["chain_2", "chain_0"]) == ["chain_2", "chain_0"]


@pytest.mark.parametrize(
    "chain",
    ["0", "chain", "chain_", "chain_x", "chains_0", "chain_0_1", "", "Chain_0"],
)
def test_parse_chain_names_rejects_anything_but_chain_n(chain):
    """A bare index is the tempting shorthand and is deliberately not accepted."""
    with pytest.raises(ValueError, match="Invalid chain identifier"):
        _parse_chain_names([chain])


def test_parse_chain_names_reports_the_offending_identifier():
    with pytest.raises(ValueError, match="'chain_one'"):
        _parse_chain_names(["chain_0", "chain_one"])


# ---------------------------------------------------------------------------
# _delete_chains_factory: numpyro layout
# ---------------------------------------------------------------------------


def test_numpyro_layout_drops_the_requested_chain(tmp_path):
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5")

    with h5py.File(path, "a") as f:
        _delete_chains_factory(f, ["chain_1"])

    with h5py.File(path, "r") as f:
        assert sorted(f["chains"].keys()) == ["chain_0", "chain_2"]
        assert f["sampler_cfg/mcmc"].attrs["num_chains"] == 2
        assert f["samples"].shape == (2 * N_SAMPLES, N_DIMS)

    assert _surviving_chain_ids(path) == {0, 2}


def test_numpyro_layout_keeps_the_surviving_samples_in_order(tmp_path):
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5", n_chains=4)

    with h5py.File(path, "a") as f:
        _delete_chains_factory(f, ["chain_0", "chain_2"])
        samples = f["samples"][()]

    np.testing.assert_array_equal(samples, _flat_samples([1, 3]))


def test_numpyro_layout_handles_double_digit_chain_names(tmp_path):
    """``chain_10`` must sort after ``chain_9``: the chain names are ordered by their
    numeric suffix, and it is that order the flat ``samples`` block is split along.
    """
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5", n_chains=11)

    with h5py.File(path, "a") as f:
        _delete_chains_factory(f, ["chain_10"])
        samples = f["samples"][()]

    np.testing.assert_array_equal(samples, _flat_samples(range(10)))


# ---------------------------------------------------------------------------
# _delete_chains_factory: flowMC layout
# ---------------------------------------------------------------------------


def test_flowMC_layout_drops_the_chain_from_both_phases(tmp_path):
    path = _write_flowMC_file(tmp_path / "flowMC.hdf5")

    with h5py.File(path, "a") as f:
        _delete_chains_factory(f, ["chain_1"])

    with h5py.File(path, "r") as f:
        for phase in ("train", "prod"):
            assert sorted(f[f"chains/{phase}"].keys()) == ["chain_0", "chain_2"]
        assert f["sampler_cfg"].attrs["n_chains"] == 2
        np.testing.assert_array_equal(
            f["chains/prod/chain_2/positions"][()], _chain_rows(2)
        )
        np.testing.assert_array_equal(
            f["chains/train/chain_0/log_probs"][()], np.zeros(N_SAMPLES)
        )

    assert _surviving_chain_ids(path) == {0, 2}


def test_flowMC_layout_rejects_mismatched_phases(tmp_path):
    """A truncated run can leave ``prod`` short of ``train``; the chain-to-samples
    mapping is then unknowable, so nothing is deleted.
    """
    path = _write_flowMC_file(tmp_path / "flowMC.hdf5", prod_chain_ids=[0, 1])

    with h5py.File(path, "a") as f:
        with pytest.raises(ValueError, match="do not match"):
            _delete_chains_factory(f, ["chain_1"])
        assert sorted(f["chains/train"].keys()) == ["chain_0", "chain_1", "chain_2"]


# ---------------------------------------------------------------------------
# _delete_chains_factory: validation
# ---------------------------------------------------------------------------


def test_unknown_sampler_is_rejected(tmp_path):
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5")
    with h5py.File(path, "a") as f:
        f["sampler_cfg"].attrs["sampler_name"] = "emcee"
        with pytest.raises(ValueError, match="Unrecognized Sampler: emcee"):
            _delete_chains_factory(f, ["chain_1"])


def test_empty_request_is_rejected(tmp_path):
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5")
    with h5py.File(path, "a") as f:
        with pytest.raises(ValueError, match="No chain names specified"):
            _delete_chains_factory(f, [])


def test_missing_chain_is_reported_with_the_available_ones(tmp_path):
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5")
    with h5py.File(path, "a") as f:
        with pytest.raises(ValueError, match=r"do not exist: \['chain_9'\]"):
            _delete_chains_factory(f, ["chain_0", "chain_9"])
        # the failure happens before anything is removed
        assert sorted(f["chains"].keys()) == ["chain_0", "chain_1", "chain_2"]


def test_deleting_every_chain_is_rejected(tmp_path):
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5")
    with h5py.File(path, "a") as f:
        with pytest.raises(ValueError, match="Cannot delete all chains"):
            _delete_chains_factory(f, ["chain_0", "chain_1", "chain_2"])


def test_repeating_a_chain_name_is_idempotent(tmp_path):
    """The request is de-duplicated before anything is removed, so a name typed twice
    does not try to delete the same group twice.
    """
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5")

    with h5py.File(path, "a") as f:
        _delete_chains_factory(f, ["chain_1", "chain_1"])
        assert sorted(f["chains"].keys()) == ["chain_0", "chain_2"]
        assert f["sampler_cfg/mcmc"].attrs["num_chains"] == 2

    assert _surviving_chain_ids(path) == {0, 2}


def test_the_request_order_does_not_matter(tmp_path):
    path = _write_numpyro_file(tmp_path / "numpyro.hdf5", n_chains=4)

    with h5py.File(path, "a") as f:
        _delete_chains_factory(f, ["chain_3", "chain_0", "chain_3"])
        samples = f["samples"][()]

    np.testing.assert_array_equal(samples, _flat_samples([1, 2]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_writes_a_cleaned_copy_and_leaves_the_input_alone(tmp_path, run_main):
    source = _write_numpyro_file(tmp_path / "inference_data.hdf5")
    output = tmp_path / "cleaned.hdf5"

    run_main(delete_chains.main, "-i", source, "-o", output, "-n", "chain_1")

    assert _surviving_chain_ids(output) == {0, 2}
    with h5py.File(source, "r") as f:
        assert sorted(f["chains"].keys()) == ["chain_0", "chain_1", "chain_2"]
        assert f["sampler_cfg/mcmc"].attrs["num_chains"] == 3


def test_main_tolerates_a_repeated_chain(tmp_path, run_main):
    source = _write_numpyro_file(tmp_path / "inference_data.hdf5")
    output = tmp_path / "cleaned.hdf5"

    run_main(delete_chains.main, "-i", source, "-o", output, "-n", "chain_1", "chain_1")

    assert _surviving_chain_ids(output) == {0, 2}


def test_main_accepts_several_chains(tmp_path, run_main):
    source = _write_numpyro_file(tmp_path / "inference_data.hdf5", n_chains=4)
    output = tmp_path / "cleaned.hdf5"

    run_main(delete_chains.main, "-i", source, "-o", output, "-n", "chain_0", "chain_3")

    assert _surviving_chain_ids(output) == {1, 2}


def test_main_defaults_the_output_to_clean_data_hdf5(tmp_path, monkeypatch, run_main):
    monkeypatch.chdir(tmp_path)
    source = _write_numpyro_file(tmp_path / "inference_data.hdf5")

    run_main(delete_chains.main, "-i", source, "-n", "chain_1")

    assert _surviving_chain_ids(tmp_path / "clean_data.hdf5") == {0, 2}


def test_main_pins_jax_to_the_cpu(tmp_path, run_main):
    """The script only rewrites an HDF5 file, so it forces the cheap JAX backend rather
    than letting an import grab a GPU.
    """
    source = _write_numpyro_file(tmp_path / "inference_data.hdf5")

    run_main(
        delete_chains.main, "-i", source, "-o", tmp_path / "out.hdf5", "-n", "chain_1"
    )

    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_main_rejects_a_bare_index_before_touching_any_file(tmp_path, run_main):
    source = _write_numpyro_file(tmp_path / "inference_data.hdf5")
    output = tmp_path / "cleaned.hdf5"

    with pytest.raises(ValueError, match="Invalid chain identifier"):
        run_main(delete_chains.main, "-i", source, "-o", output, "-n", "1")

    assert not output.exists()


@pytest.mark.parametrize("argv", [("-i", "in.hdf5"), ("-n", "chain_0")])
def test_main_requires_both_the_input_and_the_chains(argv, run_main):
    with pytest.raises(SystemExit):
        run_main(delete_chains.main, *argv)
