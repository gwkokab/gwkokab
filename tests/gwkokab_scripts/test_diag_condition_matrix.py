# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`gwkokab_scripts.diag_condition_matrix`.

The script turns a pilot run into the diagonal of a mass matrix: it reads ``samples``
and prints the per-parameter sample variance, which is what a follow-up run is seeded
with. The output is meant to be pasted into a config file, so its exact shape is part of
the contract.
"""

import re

import numpy as np
import pytest

from gwkokab_scripts import diag_condition_matrix


def _printed_values(captured: str) -> np.ndarray:
    match = re.fullmatch(r"Diagonal condition matrix: \[(.*)\]\n", captured)
    assert match is not None, f"unexpected output: {captured!r}"
    return np.array([float(value) for value in match.group(1).split(", ")])


def test_prints_one_variance_per_column(samples_file, run_main, capsys):
    samples = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    run_main(diag_condition_matrix.main, samples_file(samples))

    assert capsys.readouterr().out == "Diagonal condition matrix: [1, 100]\n"


def test_uses_the_unbiased_variance(samples_file, run_main, capsys):
    """``ddof=1`` matters: with three samples the biased estimate is a third smaller."""
    samples = np.array([[1.0], [2.0], [3.0]])
    run_main(diag_condition_matrix.main, samples_file(samples))

    values = _printed_values(capsys.readouterr().out)
    np.testing.assert_allclose(values, np.var(samples, axis=0, ddof=1))
    assert not np.allclose(values, np.var(samples, axis=0))


def test_matches_numpy_on_a_wide_posterior(samples_file, run_main, capsys):
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(200, 7)) * np.arange(1.0, 8.0)
    run_main(diag_condition_matrix.main, samples_file(samples))

    values = _printed_values(capsys.readouterr().out)
    np.testing.assert_allclose(values, np.var(samples, axis=0, ddof=1), rtol=1e-7)


def test_prints_eight_significant_digits(samples_file, run_main, capsys):
    """A variance rounded to a couple of decimals would be useless as a mass matrix."""
    # the unbiased variance of these four samples is 35/12 = 2.91666...
    samples = np.array([[0.0], [1.0], [2.0], [4.0]])
    run_main(diag_condition_matrix.main, samples_file(samples))

    assert capsys.readouterr().out == "Diagonal condition matrix: [2.9166667]\n"


def test_a_constant_parameter_reports_zero_variance(samples_file, run_main, capsys):
    samples = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    run_main(diag_condition_matrix.main, samples_file(samples))

    assert _printed_values(capsys.readouterr().out)[0] == 0.0


def test_a_single_sample_is_rejected(samples_file, run_main):
    with pytest.raises(RuntimeError, match="At least 2 samples"):
        run_main(diag_condition_matrix.main, samples_file(np.array([[1.0, 2.0]])))


def test_a_flat_samples_dataset_is_read_as_a_single_sample(samples_file, run_main):
    """``np.atleast_2d`` turns a 1-D dataset into *one* row rather than one column, so a
    file written without the parameter axis is rejected instead of silently misread.
    """
    with pytest.raises(RuntimeError, match="At least 2 samples"):
        run_main(diag_condition_matrix.main, samples_file(np.arange(10.0)))


def test_missing_file_is_an_os_error(tmp_path, run_main):
    with pytest.raises(OSError):
        run_main(diag_condition_matrix.main, tmp_path / "does_not_exist.hdf5")


def test_the_filename_is_required(run_main):
    with pytest.raises(SystemExit):
        run_main(diag_condition_matrix.main)
