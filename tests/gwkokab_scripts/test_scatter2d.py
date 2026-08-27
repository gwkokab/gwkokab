# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`gwkokab_scripts.scatter2d`.

The script is the two-variable sibling of :mod:`gwkokab_scripts.hist`: one scatter layer
per input file, both axes taken from fields of the same compound dataset. The drawn
offsets are read back from the current figure, which is the only place the plotted data
can be inspected.
"""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from gwkokab_scripts import scatter2d


DATASET = "events/posterior"
MASS_1 = np.linspace(5.0, 45.0, 20)
MASS_2 = np.linspace(3.0, 20.0, 20)


@pytest.fixture
def event_file(compound_file):
    def _make(mass_1=MASS_1, mass_2=MASS_2):
        return compound_file({"mass_1": mass_1, "mass_2": mass_2}, dataset=DATASET)

    return _make


def _run(run_main, files, output, *extra):
    if isinstance(files, str):
        files = [files]
    run_main(
        scatter2d.main,
        "-i",
        *files,
        "-d",
        DATASET,
        "-o",
        output,
        "-x",
        "mass_1",
        "-y",
        "mass_2",
        *extra,
    )


def test_writes_the_output_image(event_file, tmp_path, run_main):
    output = tmp_path / "m1_m2.png"

    _run(run_main, event_file(), output)

    assert output.exists()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plots_the_two_requested_fields_against_each_other(
    event_file, tmp_path, run_main
):
    _run(run_main, event_file(), tmp_path / "out.png")

    offsets = np.asarray(plt.gca().collections[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 0], MASS_1)
    np.testing.assert_allclose(offsets[:, 1], MASS_2)


def test_draws_one_layer_per_input_file_each_in_its_own_colour(
    event_file, tmp_path, run_main
):
    _run(run_main, [event_file(), event_file()], tmp_path / "out.png")

    collections = plt.gca().collections
    assert len(collections) == 2
    colours = {tuple(c.get_facecolor()[0][:3]) for c in collections}
    assert len(colours) == 2


def test_marker_defaults_match_the_documented_ones(event_file, tmp_path, run_main):
    _run(run_main, event_file(), tmp_path / "out.png")

    collection = plt.gca().collections[0]
    assert collection.get_alpha() == 0.7
    np.testing.assert_array_equal(collection.get_sizes(), [20.0])


def test_marker_size_and_alpha_can_be_overridden(event_file, tmp_path, run_main):
    _run(run_main, event_file(), tmp_path / "out.png", "-a", "0.2", "-s", "5")

    collection = plt.gca().collections[0]
    assert collection.get_alpha() == 0.2
    np.testing.assert_array_equal(collection.get_sizes(), [5.0])


def test_axis_labels_fall_back_to_the_field_names(event_file, tmp_path, run_main):
    _run(run_main, event_file(), tmp_path / "out.png")

    axes = plt.gca()
    assert (axes.get_xlabel(), axes.get_ylabel()) == ("mass_1", "mass_2")


def test_axis_labels_can_be_overridden(event_file, tmp_path, run_main):
    _run(
        run_main,
        event_file(),
        tmp_path / "out.png",
        "-xl",
        r"$m_1$",
        "-yl",
        r"$m_2$",
    )

    axes = plt.gca()
    assert (axes.get_xlabel(), axes.get_ylabel()) == (r"$m_1$", r"$m_2$")


def test_an_unknown_field_is_reported(event_file, tmp_path, run_main):
    with pytest.raises(ValueError, match="Field chi_eff does not appear"):
        run_main(
            scatter2d.main,
            "-i",
            event_file(),
            "-d",
            DATASET,
            "-o",
            tmp_path / "out.png",
            "-x",
            "mass_1",
            "-y",
            "chi_eff",
        )


@pytest.mark.parametrize("missing", ["-i", "-d", "-o", "-x", "-y"])
def test_all_five_flags_are_required(missing, event_file, tmp_path, run_main):
    argv = {
        "-i": ["-i", event_file()],
        "-d": ["-d", DATASET],
        "-o": ["-o", str(tmp_path / "out.png")],
        "-x": ["-x", "mass_1"],
        "-y": ["-y", "mass_2"],
    }
    argv.pop(missing)

    with pytest.raises(SystemExit):
        run_main(scatter2d.main, *[arg for pair in argv.values() for arg in pair])
