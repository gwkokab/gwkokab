# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`gwkokab_scripts.param_lens`.

The script answers the question a chain plot cannot: which hyper-parameters were tied
together, and which were frozen to the same value. It reads the two JSON side files an
inference run leaves behind and clusters their entries by value, so
:func:`~gwkokab_scripts.param_lens.create_grid` — the clustering itself — carries the
logic worth testing, while ``main`` is checked for the file handling around it.
"""

import json

import pytest
from rich.table import Table

from gwkokab_scripts import param_lens
from gwkokab_scripts.param_lens import create_grid


def _labels(table: Table) -> list:
    return list(table.columns[0].cells)


def _params(table: Table) -> list:
    return list(table.columns[1].cells)


# ---------------------------------------------------------------------------
# create_grid
# ---------------------------------------------------------------------------


def test_parameters_sharing_a_value_land_in_one_table():
    """Two names with the same index are an alias: one sampled dimension, two names."""
    grid = create_grid({"alpha_pl_0": 0, "alpha_pl_1": 0, "mmax_pl_0": 1}, "", "IDX")

    assert [_params(table) for table in grid] == [
        ["• alpha_pl_0", "• alpha_pl_1"],
        ["• mmax_pl_0"],
    ]


def test_the_label_is_written_once_per_cluster():
    """Repeating ``IDX 0`` on every row would bury the grouping it is there to show."""
    (table,) = create_grid({"alpha_pl_0": 0, "alpha_pl_1": 0}, "", "IDX")

    assert _labels(table) == ["IDX 0", ""]


def test_the_query_filters_on_the_parameter_name():
    grid = create_grid({"alpha_pl_0": 0, "mmax_pl_0": 1}, "alpha", "IDX")

    assert [_params(table) for table in grid] == [["• alpha_pl_0"]]


def test_the_query_is_case_insensitive():
    grid = create_grid({"Alpha_pl_0": 0}, "ALPHA", "IDX")

    assert [_params(table) for table in grid] == [["• Alpha_pl_0"]]


def test_an_empty_query_keeps_everything():
    grid = create_grid({"alpha_pl_0": 0, "mmax_pl_0": 1}, "", "IDX")

    assert len(grid) == 2


def test_no_match_returns_a_message_instead_of_tables():
    grid = create_grid({"alpha_pl_0": 0}, "eccentricity", "IDX")

    assert grid == ["[dim italic]No matches found for 'eccentricity'[/dim italic]"]


def test_numeric_values_are_ordered_numerically():
    """Sorting the values as text would put index 10 between 1 and 2."""
    grid = create_grid({"a": 1, "b": 2, "c": 10}, "", "IDX")

    assert [_labels(table)[0] for table in grid] == ["IDX 1", "IDX 2", "IDX 10"]


def test_decimal_values_are_ordered_numerically():
    grid = create_grid({"a": 2.5, "b": 10.5}, "", "VAL")

    assert [_labels(table)[0] for table in grid] == ["VAL 2.5", "VAL 10.5"]


def test_non_numeric_values_sort_after_the_numeric_ones():
    """Constants can be strings — a parameter frozen to another parameter's name — and
    those cluster after the numbers, alphabetically.
    """
    grid = create_grid({"a": 1, "b": "mmin_pl_0", "c": "beta_pl_0"}, "", "VAL")

    assert [_labels(table)[0] for table in grid] == [
        "VAL 1",
        "VAL beta_pl_0",
        "VAL mmin_pl_0",
    ]


def test_negative_values_sort_with_the_text_cluster():
    """The numeric test is ``str.isdigit`` on the value, which a leading minus fails, so
    a negative constant is grouped with the strings rather than before the positives.
    """
    grid = create_grid({"a": 1.0, "b": -2.0}, "", "VAL")

    assert [_labels(table)[0] for table in grid] == ["VAL 1.0", "VAL -2.0"]


def test_values_are_clustered_by_their_string_form():
    """``0`` and ``0.0`` are different labels even though they compare equal."""
    grid = create_grid({"a": 0, "b": 0.0}, "", "VAL")

    assert len(grid) == 2


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@pytest.fixture
def json_files(tmp_path):
    """Write the two side files an inference run drops next to its HDF5 output."""

    def _make(constants=None, mapping=None):
        constants_path = tmp_path / "constants.json"
        mapping_path = tmp_path / "nf_samples_mapping.json"
        constants_path.write_text(json.dumps(constants or {"mmin_pl_0": 5.0}))
        mapping_path.write_text(json.dumps(mapping or {"alpha_pl_0": 0}))
        return constants_path, mapping_path

    return _make


def test_main_prints_both_sections(json_files, run_main, capsys):
    constants, mapping = json_files()

    run_main(param_lens.main, "--constants", constants, "--mapping", mapping)

    out = capsys.readouterr().out
    assert "MAPPING" in out
    assert "CONSTANTS" in out
    assert "alpha_pl_0" in out
    assert "mmin_pl_0" in out


def test_main_filters_both_sections_by_the_query(json_files, run_main, capsys):
    constants, mapping = json_files(
        constants={"mmin_pl_0": 5.0, "mmax_pl_0": 100.0},
        mapping={"alpha_pl_0": 0, "mmin_pl_0": 1},
    )

    run_main(param_lens.main, "mmin", "--constants", constants, "--mapping", mapping)

    out = capsys.readouterr().out
    assert "mmin_pl_0" in out
    assert "alpha_pl_0" not in out
    assert "No matches found for 'mmin'" not in out


def test_main_reports_a_query_that_matches_nothing(json_files, run_main, capsys):
    constants, mapping = json_files()

    run_main(param_lens.main, "chi_eff", "--constants", constants, "--mapping", mapping)

    assert capsys.readouterr().out.count("No matches found for 'chi_eff'") == 2


def test_a_missing_file_is_reported_without_a_traceback(tmp_path, run_main, capsys):
    """The defaults point at the current directory, so running the script from the wrong
    place is the common mistake and must not look like a crash.
    """
    mapping = tmp_path / "nf_samples_mapping.json"
    mapping.write_text(json.dumps({"alpha_pl_0": 0}))

    assert (
        run_main(
            param_lens.main,
            "--constants",
            tmp_path / "absent.json",
            "--mapping",
            mapping,
        )
        is None
    )

    out = capsys.readouterr().out
    assert "Could not find" in out
    assert "absent.json" in out
    assert "MAPPING" not in out
