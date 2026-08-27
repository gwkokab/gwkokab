# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`gwkokab_scripts.hist`.

The script overlays one step histogram per input file, all read from the same field of
the same compound dataset, and saves the figure. Since it draws on the implicit current
figure, the assertions read that figure back through ``plt.gca()`` after ``main()``
returns, alongside checking that the output file was written.
"""

import numpy as np
import pytest
from matplotlib import pyplot as plt

from gwkokab_scripts import hist


DATASET = "events/posterior"


@pytest.fixture
def event_file(compound_file):
    """A single-file factory for the two-field layout the script expects."""

    def _make(mass_1=None, mass_2=None):
        size = 50
        rng = np.random.default_rng(0)
        return compound_file(
            {
                "mass_1": np.linspace(5.0, 45.0, size) if mass_1 is None else mass_1,
                "mass_2": rng.normal(size=size) if mass_2 is None else mass_2,
            },
            dataset=DATASET,
        )

    return _make


def _run(run_main, files, output, *extra):
    if isinstance(files, str):
        files = [files]
    run_main(
        hist.main,
        "-i",
        *files,
        "-d",
        DATASET,
        "-o",
        output,
        "-v",
        "mass_1",
        *extra,
    )


def test_writes_the_output_image(event_file, tmp_path, run_main):
    output = tmp_path / "mass_1.png"

    _run(run_main, event_file(), output)

    assert output.exists()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_draws_one_step_histogram_per_input_file(event_file, tmp_path, run_main):
    _run(run_main, [event_file(), event_file(), event_file()], tmp_path / "out.png")

    assert len(plt.gca().patches) == 3


def test_each_input_file_gets_its_own_colour(event_file, tmp_path, run_main):
    """Overlaid histograms are only readable if the layers are distinguishable."""
    _run(run_main, [event_file(), event_file()], tmp_path / "out.png")

    edge_colours = {tuple(patch.get_edgecolor()) for patch in plt.gca().patches}
    assert len(edge_colours) == 2


@pytest.mark.parametrize("bins", [5, 12])
def test_bins_are_passed_through(event_file, tmp_path, run_main, bins):
    """A step histogram traces two vertices per bin edge, plus the closing pair."""
    _run(run_main, event_file(), tmp_path / "out.png", "--bins", bins)

    assert np.asarray(plt.gca().patches[0].get_xy()).shape == (2 * bins + 2, 2)


def test_the_x_axis_falls_back_to_the_variable_name(event_file, tmp_path, run_main):
    _run(run_main, event_file(), tmp_path / "out.png")

    assert plt.gca().get_xlabel() == "mass_1"


def test_the_x_label_can_be_overridden(event_file, tmp_path, run_main):
    _run(run_main, event_file(), tmp_path / "out.png", "-l", r"$m_1$ [M$_\odot$]")

    assert plt.gca().get_xlabel() == r"$m_1$ [M$_\odot$]"


def test_counts_are_labelled_as_frequency(event_file, tmp_path, run_main):
    _run(run_main, event_file(), tmp_path / "out.png")

    assert plt.gca().get_ylabel() == "Frequency"


def test_density_relabels_and_rescales_the_y_axis(event_file, tmp_path, run_main):
    """``--density`` is not only a label: the drawn heights integrate to one, so over a
    40 solar-mass range they drop far below the raw counts.
    """
    _run(run_main, event_file(), tmp_path / "out.png", "--density", "--bins", "5")

    axes = plt.gca()
    assert axes.get_ylabel() == "Density"
    assert np.asarray(axes.patches[0].get_xy())[:, 1].max() < 1.0


def test_counts_are_drawn_at_their_raw_height(event_file, tmp_path, run_main):
    _run(run_main, event_file(), tmp_path / "out.png", "--bins", "5")

    heights = np.asarray(plt.gca().patches[0].get_xy())[:, 1]
    assert heights.max() == 10.0  # 50 evenly spaced samples over 5 bins


def test_alpha_is_applied_to_each_layer(event_file, tmp_path, run_main):
    _run(run_main, event_file(), tmp_path / "out.png", "-a", "0.4")

    assert plt.gca().patches[0].get_alpha() == 0.4


def test_an_unknown_field_is_reported(event_file, tmp_path, run_main):
    with pytest.raises(ValueError, match="Field spin_1z does not appear"):
        run_main(
            hist.main,
            "-i",
            event_file(),
            "-d",
            DATASET,
            "-o",
            tmp_path / "out.png",
            "-v",
            "spin_1z",
        )


def test_an_unknown_dataset_is_reported(event_file, tmp_path, run_main):
    with pytest.raises(KeyError):
        run_main(
            hist.main,
            "-i",
            event_file(),
            "-d",
            "events/not_there",
            "-o",
            tmp_path / "out.png",
            "-v",
            "mass_1",
        )


@pytest.mark.parametrize("missing", ["-i", "-d", "-o", "-v"])
def test_every_flag_but_the_optional_ones_is_required(
    missing, event_file, tmp_path, run_main
):
    argv = {
        "-i": ["-i", event_file()],
        "-d": ["-d", DATASET],
        "-o": ["-o", str(tmp_path / "out.png")],
        "-v": ["-v", "mass_1"],
    }
    argv.pop(missing)

    with pytest.raises(SystemExit):
        run_main(hist.main, *[arg for pair in argv.values() for arg in pair])
