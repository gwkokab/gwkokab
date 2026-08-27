# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`gwkokab_scripts.numpyro_dense_mass_structure`.

The script reads a pilot posterior and suggests ``dense_mass`` blocks for NUTS by
clustering the columns of ``samples`` on their absolute correlation. Its output is meant
to be pasted into ``sampler_cfg.json``, so the block list is checked by parsing it back
as JSON rather than by matching text.

The ``samples`` each test writes come from a seeded generator with a known correlation
structure: a pair of columns tied together and the rest independent.
"""

import json

import numpy as np
import pytest

from gwkokab_scripts import numpyro_dense_mass_structure as dense_mass


N_SAMPLES = 400


def _correlated_pair(rng, n=N_SAMPLES):
    """A column and a near-copy of it: correlation ~0.999, well above any threshold the
    tests use.
    """
    first = rng.normal(size=n)
    return first, first + 0.05 * rng.normal(size=n)


def _suggested_blocks(captured: str) -> list:
    """Parse the ``dense_mass`` list the script prints between two bare brackets."""
    lines = captured.splitlines()
    start = lines.index("[")
    end = lines.index("]", start)
    return json.loads("\n".join(lines[start : end + 1]))


def _recovered_parameters(captured: str) -> list:
    lines = captured.splitlines()
    start = lines.index("List of parameters in posterior file:") + 1
    end = lines.index("[", start)
    return [line for line in lines[start:end] if line]


@pytest.fixture
def posterior(samples_file):
    """Two correlated parameters and one independent one."""
    rng = np.random.default_rng(0)
    alpha, mmin = _correlated_pair(rng)
    log_rate = rng.normal(size=N_SAMPLES)
    return samples_file(
        np.column_stack([alpha, mmin, log_rate]),
        {"alpha_pl_0": 0, "mmin_pl_0": 1, "log_rate_0": 2},
    )


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults(run_main):
    args = run_main(dense_mass.parse_args, "inference_data.hdf5")

    assert args.posterior_file == "inference_data.hdf5"
    assert args.corr_threshold == 0.5
    assert args.min_block_size == 2


def test_the_posterior_file_is_required(run_main):
    with pytest.raises(SystemExit):
        run_main(dense_mass.parse_args)


# ---------------------------------------------------------------------------
# block suggestions
# ---------------------------------------------------------------------------


def test_correlated_parameters_are_suggested_as_one_block(posterior, run_main, capsys):
    run_main(dense_mass.main, posterior)

    assert _suggested_blocks(capsys.readouterr().out) == [["alpha_pl_0", "mmin_pl_0"]]


def test_an_independent_parameter_is_left_out_of_the_blocks(
    posterior, run_main, capsys
):
    """A block of one is a diagonal mass matrix entry, which NUTS already has."""
    run_main(dense_mass.main, posterior)

    out = capsys.readouterr().out
    assert "log_rate_0" in _recovered_parameters(out)
    assert "log_rate_0" not in [
        name for block in _suggested_blocks(out) for name in block
    ]


def test_blocks_are_ordered_by_their_first_parameter(samples_file, run_main, capsys):
    """Two independent pairs, interleaved across the columns: the block holding the
    earliest column has to be printed first, whatever labels the clustering handed
    out.
    """
    rng = np.random.default_rng(1)
    alpha, mmin = _correlated_pair(rng)
    beta, mmax = _correlated_pair(rng)
    path = samples_file(
        np.column_stack([alpha, beta, mmax, mmin]),
        {"alpha_pl_0": 0, "beta_pl_0": 1, "mmax_pl_0": 2, "mmin_pl_0": 3},
    )

    run_main(dense_mass.main, path)

    assert _suggested_blocks(capsys.readouterr().out) == [
        ["alpha_pl_0", "mmin_pl_0"],
        ["beta_pl_0", "mmax_pl_0"],
    ]


def test_a_higher_threshold_breaks_the_block_apart(posterior, run_main, capsys):
    run_main(dense_mass.main, posterior, "--corr-threshold", "0.99999")

    assert _suggested_blocks(capsys.readouterr().out) == []


def test_min_block_size_filters_out_small_blocks(posterior, run_main, capsys):
    run_main(dense_mass.main, posterior, "--min-block-size", "3")

    assert _suggested_blocks(capsys.readouterr().out) == []


# ---------------------------------------------------------------------------
# column selection
# ---------------------------------------------------------------------------


def test_a_frozen_parameter_is_dropped(samples_file, run_main, capsys):
    """A constant column has zero variance: it cannot correlate with anything and would
    poison ``corrcoef`` with a division by zero.
    """
    rng = np.random.default_rng(0)
    alpha, mmin = _correlated_pair(rng)
    path = samples_file(
        np.column_stack([alpha, mmin, np.full(N_SAMPLES, 3.0)]),
        {"alpha_pl_0": 0, "mmin_pl_0": 1, "mmax_pl_0": 2},
    )

    run_main(dense_mass.main, path)

    out = capsys.readouterr().out
    assert _recovered_parameters(out) == ["alpha_pl_0", "mmin_pl_0"]
    assert _suggested_blocks(out) == [["alpha_pl_0", "mmin_pl_0"]]


def test_a_column_with_a_non_finite_sample_is_dropped(samples_file, run_main, capsys):
    rng = np.random.default_rng(0)
    alpha, mmin = _correlated_pair(rng)
    diverged = rng.normal(size=N_SAMPLES)
    diverged[7] = np.nan
    path = samples_file(
        np.column_stack([alpha, mmin, diverged]),
        {"alpha_pl_0": 0, "mmin_pl_0": 1, "log_rate_0": 2},
    )

    run_main(dense_mass.main, path)

    assert _recovered_parameters(capsys.readouterr().out) == ["alpha_pl_0", "mmin_pl_0"]


def test_tied_parameters_are_reported_under_one_name(samples_file, run_main, capsys):
    """Aliases share a sampled dimension, so the column is named after the first of them
    alphabetically rather than listed twice.
    """
    rng = np.random.default_rng(0)
    alpha, mmin = _correlated_pair(rng)
    path = samples_file(
        np.column_stack([alpha, mmin]),
        {"beta_pl_0": 0, "alpha_pl_0": 0, "mmin_pl_0": 1},
    )

    run_main(dense_mass.main, path)

    assert _recovered_parameters(capsys.readouterr().out) == ["alpha_pl_0", "mmin_pl_0"]


def test_a_column_missing_from_the_index_gets_a_placeholder_name(
    samples_file, run_main, capsys
):
    rng = np.random.default_rng(0)
    alpha, mmin = _correlated_pair(rng)
    path = samples_file(np.column_stack([alpha, mmin]), {"alpha_pl_0": 0})

    run_main(dense_mass.main, path)

    assert _recovered_parameters(capsys.readouterr().out) == [
        "alpha_pl_0",
        "unnamed_param_1",
    ]


def test_fewer_than_two_usable_columns_is_an_error(samples_file, run_main):
    """One free parameter cannot form a block, and ``squareform`` would fail on it."""
    rng = np.random.default_rng(0)
    path = samples_file(
        np.column_stack([rng.normal(size=N_SAMPLES), np.zeros(N_SAMPLES)]),
        {"alpha_pl_0": 0, "mmax_pl_0": 1},
    )

    with pytest.raises(ValueError, match="Not enough valid parameters"):
        run_main(dense_mass.main, path)


# ---------------------------------------------------------------------------
# the correlation listing
# ---------------------------------------------------------------------------


def test_strong_pairs_are_listed_with_their_correlation(posterior, run_main, capsys):
    run_main(dense_mass.main, posterior)

    listing = capsys.readouterr().out.split("Strong pairwise correlations:")[1]
    rows = [line.split() for line in listing.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0][:2] == ["alpha_pl_0", "mmin_pl_0"]
    assert float(rows[0][-1]) == pytest.approx(1.0, abs=0.01)


def test_weak_pairs_are_not_listed(posterior, run_main, capsys):
    run_main(dense_mass.main, posterior, "--corr-threshold", "0.99999")

    listing = capsys.readouterr().out.split("Strong pairwise correlations:")[1]
    assert listing.strip() == ""
