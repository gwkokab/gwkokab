# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`gwkokab_scripts.flowMC_info`.

The script explains a ``flowMC`` sampler config before it is run: how many samples each
training loop keeps, when the history window saturates and starts forgetting, and what
the heuristics would pick instead. The arithmetic — ``infer_n_dims`` and
``generate_heuristics`` — is tested directly; the rich rendering is tested through the
text it puts on stdout.
"""

import json
import re

import pytest
from matplotlib import pyplot as plt

from gwkokab_scripts import flowMC_info
from gwkokab_scripts.flowMC_info import (
    display_config_table,
    display_diagnostics_panel,
    display_loop_table,
    display_suggestions,
    generate_heuristics,
    infer_n_dims,
    load_config,
    plot_history,
    print_header,
    print_section,
)


# Seven samples kept per loop out of a hundred-sample window, so the history saturates
# after fifteen loops — well past the five training loops the table stops at.
BASE_CONFIG = {
    "n_chains": 1,
    "n_local_steps": 4,
    "local_thinning": 1,
    "n_global_steps": 3,
    "global_thinning": 1,
    "history_window": 100,
    "n_max_examples": 50,
    "n_training_loops": 5,
    "n_production_loops": 2,
    "batch_size": 1024,
    "n_epochs": 10,
    "mass_matrix": [1.0] * 4,
}


@pytest.fixture
def config_file(tmp_path):
    """Factory writing a ``flowMC_config.json`` with the given keys overridden."""
    counter = {"n": 0}

    def _make(**overrides) -> str:
        config = {**BASE_CONFIG, **overrides}
        for key, value in overrides.items():
            if value is None:
                config.pop(key)
        path = tmp_path / f"flowMC_config_{counter['n']}.json"
        counter["n"] += 1
        path.write_text(json.dumps(config))
        return str(path)

    return _make


def _loop_rows(out: str) -> list:
    """The body rows of the loop-simulation table, as lists of stripped cells."""
    table = out.split("Training Loop Simulation", 1)[1].split("Optimization", 1)[0]
    rows = []
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.split("│")]
        if len(cells) == 5 and cells[0].isdigit():
            rows.append(cells)
    return rows


def _config_value(out: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}\s{{2,}}(\S+)", out)
    assert match is not None, f"{label!r} missing from:\n{out}"
    return match.group(1)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_returns_the_parsed_json(config_file):
    assert load_config(config_file())["n_chains"] == 1


def test_load_config_reports_a_missing_file(tmp_path):
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match="Config file not found at"):
        load_config(str(missing))


def test_load_config_propagates_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{'n_chains': 1,}")
    with pytest.raises(json.JSONDecodeError):
        load_config(str(path))


# ---------------------------------------------------------------------------
# infer_n_dims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "cli_n_dims", "expected"),
    [
        ({"n_dims": 7, "mass_matrix": [1.0] * 4}, 3, 3),
        ({"n_dims": 7}, None, 7),
        ({"n_dims": "9"}, None, 9),
        ({"n_dims": None, "mass_matrix": [1.0] * 4}, None, 4),
        ({"mass_matrix": [1.0] * 12}, None, 12),
        ({"mass_matrix": 4}, None, 0),
        ({}, None, 0),
        ({}, 5, 5),
    ],
    ids=[
        "cli-wins",
        "from-n-dims",
        "n-dims-as-string",
        "unusable-n-dims-falls-back",
        "from-mass-matrix",
        "mass-matrix-not-a-list",
        "nothing-to-go-on",
        "cli-only",
    ],
)
def test_infer_n_dims(config, cli_n_dims, expected):
    assert infer_n_dims(config, cli_n_dims) == expected


# ---------------------------------------------------------------------------
# generate_heuristics
# ---------------------------------------------------------------------------


def test_no_heuristics_without_a_dimension():
    assert generate_heuristics(0, 20, 2000) == {}


@pytest.mark.parametrize(
    ("n_dims", "hidden_units", "n_layers", "history_loops"),
    [
        (1, [64, 64], 4, 5),
        (15, [64, 64], 4, 5),
        (16, [128, 128], 6, 6),
        (30, [128, 128], 6, 6),
        (31, [256, 256], 8, 8),
    ],
)
def test_the_architecture_tier_follows_the_dimension(
    n_dims, hidden_units, n_layers, history_loops
):
    """The tiers switch at 15 and 30 dimensions inclusive."""
    heuristics = generate_heuristics(n_dims, 10, 1000)

    assert heuristics["hidden_units"] == hidden_units
    assert heuristics["n_layers"] == n_layers
    assert heuristics["history_loops"] == history_loops


def test_the_window_holds_a_few_loops_worth_of_every_chain():
    heuristics = generate_heuristics(10, 20, 8000)

    assert heuristics["history_window"] == 5 * (8000 // 20)
    assert heuristics["candidate_capacity"] == 5 * (8000 // 20) * 20


def test_the_training_set_is_capped_by_the_tier_target():
    """Forty thousand candidates, but a small model only wants twenty thousand."""
    heuristics = generate_heuristics(10, 20, 8000)

    assert heuristics["candidate_capacity"] == 40_000
    assert heuristics["n_max_examples"] == 20_000


def test_the_training_set_is_capped_by_what_the_window_holds():
    heuristics = generate_heuristics(10, 20, 2000)

    assert heuristics["candidate_capacity"] == 10_000
    assert heuristics["n_max_examples"] == 10_000


def test_the_batch_is_an_eighth_of_half_the_training_set():
    heuristics = generate_heuristics(10, 20, 8000)

    assert heuristics["batch_size"] == int(0.5 * 20_000 / 8)


def test_the_batch_never_drops_below_a_thousand():
    """A handful of samples per batch would make the flow's gradients pure noise."""
    heuristics = generate_heuristics(10, 1, 10)

    assert heuristics["n_max_examples"] == 50
    assert heuristics["batch_size"] == 1024


def test_the_batch_never_exceeds_an_eighth_of_the_largest_target():
    """The explicit 10,000 ceiling is unreachable: the biggest tier targets 40,000
    examples, which puts the batch at 2,500.
    """
    assert generate_heuristics(100, 1000, 10**7)["batch_size"] == 2500


@pytest.mark.parametrize(
    ("n_chains", "kept_total_per_loop"),
    [(0, 2000), (4, 0)],
    ids=["no-chains", "nothing-kept"],
)
def test_an_empty_history_falls_back_to_a_default_batch(n_chains, kept_total_per_loop):
    """``n_chains`` of zero must not divide by zero, and either way there is no history
    to size the batch against.
    """
    heuristics = generate_heuristics(10, n_chains, kept_total_per_loop)

    assert heuristics["candidate_capacity"] == 0
    assert heuristics["n_max_examples"] == 0
    assert heuristics["batch_size"] == 4096


@pytest.mark.parametrize(
    ("n_dims", "training_loops", "production_loops"), [(10, 24, 12), (31, 32, 16)]
)
def test_the_loop_counts_scale_with_the_history_depth(
    n_dims, training_loops, production_loops
):
    heuristics = generate_heuristics(n_dims, 10, 1000)

    assert heuristics["training_loops"] == training_loops
    assert heuristics["production_loops"] == production_loops


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def test_print_header_frames_the_title(capsys):
    print_header("FlowMC Configuration Analysis")

    out = capsys.readouterr().out
    assert "FlowMC Configuration Analysis" in out
    assert "╭" in out


def test_print_section_rules_off_the_title(capsys):
    print_section("Optimization")

    out = capsys.readouterr().out
    assert "Optimization" in out
    assert "─" in out


def test_display_config_table_prints_every_row(capsys):
    display_config_table([("Chains", 20), ("Config Path", "flowMC_config.json")])

    out = capsys.readouterr().out
    assert _config_value(out, "Chains") == "20"
    assert _config_value(out, "Config Path") == "flowMC_config.json"


def test_display_diagnostics_panel_bullets_every_note(capsys):
    display_diagnostics_panel(
        {"TOTAL new samples per loop": "7"},
        [
            "Step count where forgetting matters: 105 (15 loops × 4 local)",
            "History saturation: 15 loop(s)",
            "a note without a colon",
        ],
    )

    out = capsys.readouterr().out
    assert "TOTAL new samples per loop" in out
    assert out.count("•") == 3
    assert "Step count where forgetting matters" in out
    assert "a note without a colon" in out


def test_display_diagnostics_panel_without_notes(capsys):
    display_diagnostics_panel({"Kept LOCAL samples per loop": "4"}, [])

    out = capsys.readouterr().out
    assert "Kept LOCAL samples per loop" in out
    assert "•" not in out


def test_display_loop_table_formats_and_flags_the_rows(capsys):
    display_loop_table(
        [(1, 1000, 1000, 1000, "Saturated"), (2, 2000, 2000, 1500, "Forgetting")], 2
    )

    out = capsys.readouterr().out
    assert _loop_rows(out) == [
        ["1", "1,000", "1,000", "1,000", "Saturated"],
        ["2", "2,000", "2,000", "1,500", "Forgetting"],
    ]


def test_display_suggestions_shows_both_columns(capsys):
    current = {
        "history_window": 100,
        "n_max_examples": 50,
        "batch_size": 1024,
        "n_training_loops": 5,
        "n_production_loops": 2,
    }
    suggested = generate_heuristics(4, 1, 7)

    display_suggestions(current, suggested)

    out = capsys.readouterr().out
    for label in (
        "History Window",
        "N Max Examples",
        "Batch Size",
        "Training Loops",
        "Production Loops",
    ):
        assert label in out
    assert "Hidden Units: [64, 64]" in out
    assert "Layers: 4" in out


def test_display_suggestions_accepts_a_non_numeric_current_value(capsys):
    """Nothing validates the config, so a string where a count belongs must still render
    rather than blow up in the thousands separator.
    """
    current = {
        "history_window": "auto",
        "n_max_examples": 50,
        "batch_size": 1024,
        "n_training_loops": 5,
        "n_production_loops": 2,
    }

    display_suggestions(current, generate_heuristics(4, 1, 7))

    assert "auto" in capsys.readouterr().out


def test_plot_history_writes_a_file_and_closes_the_figure(tmp_path):
    path = tmp_path / "flowmc_history_plot.png"

    plot_history([1, 2, 3], [7, 14, 21], [7, 14, 21], str(path))

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_renders_every_section(config_file, run_main, capsys):
    run_main(flowMC_info.main, config_file())

    out = capsys.readouterr().out
    for section in (
        "FlowMC Configuration Analysis",
        "Configuration Overview",
        "History & Memory Dynamics",
        "Detailed Dynamics & Diagnostics",
        "Training Loop Simulation",
        "Optimization",
        "Heuristic Tuning Suggestions",
        "Suggested Architecture",
    ):
        assert section in out


def test_main_reports_the_kept_samples_per_loop(config_file, run_main, capsys):
    """Four local and three global samples survive the thinning, on a single chain."""
    run_main(flowMC_info.main, config_file())

    out = capsys.readouterr().out
    assert _config_value(out, "Kept LOCAL samples per loop") == "4"
    assert _config_value(out, "Kept GLOBAL samples per loop") == "3"
    assert _config_value(out, "TOTAL new samples per loop") == "7"
    assert _config_value(out, "Total NF candidate capacity (window)") == "100"


def test_main_takes_the_dimension_from_the_mass_matrix(config_file, run_main, capsys):
    run_main(flowMC_info.main, config_file())

    assert _config_value(capsys.readouterr().out, "Dimensions (n_dims)") == "4"


def test_main_honours_the_dimension_override(config_file, run_main, capsys):
    run_main(flowMC_info.main, config_file(), "--n-dims", "40")

    out = capsys.readouterr().out
    assert _config_value(out, "Dimensions (n_dims)") == "40"
    assert "Hidden Units: [256, 256]" in out


def test_main_warns_and_skips_the_suggestions_without_a_dimension(
    config_file, run_main, capsys
):
    run_main(flowMC_info.main, config_file(mass_matrix=None))

    out = capsys.readouterr().out
    assert "'mass_matrix' missing" in out
    assert _config_value(out, "Dimensions (n_dims)") == "Unknown"
    assert "Cannot generate suggestions" in out
    assert "Heuristic Tuning Suggestions" not in out


def test_main_reports_the_history_saturation(config_file, run_main, capsys):
    """A hundred-sample window fed seven samples a loop fills after fifteen loops, and
    forgets from the sixteenth — which is the 150th epoch and the 105th sampler step.
    """
    run_main(flowMC_info.main, config_file())

    out = capsys.readouterr().out
    assert "History saturation: 15 loop(s) (effective depth ~14.29 loops)" in out
    assert (
        "First epoch index where forgetting matters: 150 (0-based), 151 (1-based)"
        in out
    )
    assert "Step count where forgetting matters: 105" in out


def test_main_simulates_every_training_loop(config_file, run_main, capsys):
    """Seven new samples a loop, none of them yet crowded out of the window or the
    training subset, so all three counts track each other.
    """
    run_main(flowMC_info.main, config_file())

    assert _loop_rows(capsys.readouterr().out) == [
        ["1", "7", "7", "7", "-"],
        ["2", "14", "14", "14", "-"],
        ["3", "21", "21", "21", "-"],
        ["4", "28", "28", "28", "-"],
        ["5", "35", "35", "35", "-"],
    ]


def test_main_flags_saturation_and_forgetting(config_file, run_main, capsys):
    """With a window of seven the very first loop fills it, the second starts pushing
    samples out, and the table runs two loops past that before stopping.
    """
    run_main(flowMC_info.main, config_file(history_window=7, n_training_loops=10))

    rows = _loop_rows(capsys.readouterr().out)
    assert [row[0] for row in rows] == ["1", "2", "3", "4"]
    assert [row[4] for row in rows] == [
        "Saturated",
        "Forgetting",
        "Forgetting",
        "Forgetting",
    ]
    assert [row[2] for row in rows] == ["7"] * 4


def test_main_truncates_the_table_on_request(config_file, run_main, capsys):
    run_main(flowMC_info.main, config_file(), "--max-loops-table", "2")

    assert [row[0] for row in _loop_rows(capsys.readouterr().out)] == ["1", "2"]


def test_main_never_shows_more_loops_than_are_run(config_file, run_main, capsys):
    run_main(flowMC_info.main, config_file(), "--max-loops-table", "99")

    assert [row[0] for row in _loop_rows(capsys.readouterr().out)] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]


def test_main_caps_the_training_set_at_n_max_examples(config_file, run_main, capsys):
    """Once the pool passes ``n_max_examples`` the flow only ever sees a subset."""
    run_main(flowMC_info.main, config_file(n_max_examples=20))

    out = capsys.readouterr().out
    assert [row[2] for row in _loop_rows(out)] == ["7", "14", "21", "28", "35"]
    assert [row[3] for row in _loop_rows(out)] == ["7", "14", "20", "20", "20"]


def test_main_notes_when_the_whole_history_is_used(config_file, run_main, capsys):
    run_main(flowMC_info.main, config_file(n_max_examples=1000))

    assert "NF trains on ALL visible samples each loop" in capsys.readouterr().out


def test_main_notes_when_only_a_subset_is_used(config_file, run_main, capsys):
    run_main(flowMC_info.main, config_file(n_max_examples=50))

    assert "random subset of up to 50 samples" in capsys.readouterr().out


def test_main_handles_a_configuration_that_keeps_nothing(config_file, run_main, capsys):
    """Thinning coarser than the loop length keeps no samples at all; the report must
    still come out, without a forgetting boundary.
    """
    run_main(flowMC_info.main, config_file(local_thinning=100, global_thinning=100))

    out = capsys.readouterr().out
    assert "History saturation: 0 loop(s)" in out
    assert "forgetting matters" not in out
    assert [row[1] for row in _loop_rows(out)] == ["0"] * 5


def test_main_survives_a_zero_thinning(config_file, run_main, capsys):
    """``local_thinning`` of zero means "keep everything", not a division by zero."""
    run_main(flowMC_info.main, config_file(local_thinning=0, global_thinning=0))

    out = capsys.readouterr().out
    assert _config_value(out, "TOTAL new samples per loop") == "7"


def test_main_defaults_the_production_loops_to_zero(config_file, run_main, capsys):
    run_main(flowMC_info.main, config_file(n_production_loops=None))

    out = capsys.readouterr().out
    row = re.search(r"Production Loops\s+│\s+(\S+)\s+│", out)
    assert row is not None and row.group(1) == "0"


def test_main_does_not_plot_unless_asked(config_file, run_main, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(flowMC_info, "plot_history", lambda *args: calls.append(args))

    run_main(flowMC_info.main, config_file())

    assert calls == []
    assert "Plot saved to" not in capsys.readouterr().out


def test_main_writes_the_requested_plot(config_file, tmp_path, run_main, capsys):
    path = tmp_path / "history.png"

    run_main(flowMC_info.main, config_file(), "--plot", "--plot-path", path)

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert f"Plot saved to: {path}" in capsys.readouterr().out


def test_the_plot_covers_the_loops_the_table_leaves_out(
    config_file, run_main, capsys, monkeypatch
):
    calls = []
    monkeypatch.setattr(flowMC_info, "plot_history", lambda *args: calls.append(args))

    run_main(flowMC_info.main, config_file(), "--plot", "--max-loops-table", "2")

    assert [row[0] for row in _loop_rows(capsys.readouterr().out)] == ["1", "2"]
    loops, candidates, n_train, save_path = calls[0]
    assert loops == [1, 2, 3, 4, 5]
    assert candidates == [7, 14, 21, 28, 35]
    assert n_train == [7, 14, 21, 28, 35]
    assert save_path == "flowmc_history_plot.png"


def test_main_exits_when_the_config_cannot_be_read(tmp_path, run_main, capsys):
    with pytest.raises(SystemExit) as excinfo:
        run_main(flowMC_info.main, tmp_path / "absent.json")

    assert excinfo.value.code == 1
    assert "Error loading config" in capsys.readouterr().out


def test_main_needs_the_keys_it_reads(config_file, run_main):
    """Nothing validates the config, so a missing key surfaces as a plain KeyError."""
    with pytest.raises(KeyError, match="n_epochs"):
        run_main(flowMC_info.main, config_file(n_epochs=None))


def test_main_defaults_to_flowmc_config_json(tmp_path, monkeypatch, run_main, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "flowMC_config.json").write_text(json.dumps(BASE_CONFIG))

    run_main(flowMC_info.main)

    assert "FlowMC Configuration Analysis" in capsys.readouterr().out
