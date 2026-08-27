# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JSON helpers every CLI uses to read its four configuration files.

The two file helpers exist to funnel every failure mode into a single ``ValueError``, so
the interesting assertions here are about what they *convert*, not what they let
through.
"""

import json

import pytest

from gwkokab.analysis.utils.common import expand_arguments, read_json, write_json


@pytest.mark.parametrize(
    ("arg", "n", "expected"),
    [
        ("physics", 3, ["physics_0", "physics_1", "physics_2"]),
        ("alpha_pl", 1, ["alpha_pl_0"]),
        ("log_rate", 0, []),
    ],
)
def test_expand_arguments(arg, n, expected):
    assert expand_arguments(arg, n) == expected


def test_read_json_round_trips_write_json(tmp_path):
    path = str(tmp_path / "cfg.json")
    content = {"a": 1, "b": [1.0, 2.0], "c": {"d": "e"}, "f": None, "g": True}

    write_json(path, content)

    assert read_json(path) == content


def test_write_json_is_indented(tmp_path):
    """The dumps are meant to be edited by hand, so they are pretty-printed."""
    path = tmp_path / "cfg.json"

    write_json(str(path), {"a": {"b": 1}})

    assert path.read_text().splitlines()[1].startswith("    ")


def test_read_json_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Error loading configuration"):
        read_json(str(tmp_path / "absent.json"))


def test_read_json_malformed_raises_value_error(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{not json,}")

    with pytest.raises(ValueError, match="Error loading configuration"):
        read_json(str(path))


def test_write_json_unserializable_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Error writing configuration"):
        write_json(str(tmp_path / "cfg.json"), {"a": object()})


def test_write_json_missing_directory_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Error writing configuration"):
        write_json(str(tmp_path / "absent" / "cfg.json"), {"a": 1})


def test_read_json_accepts_a_top_level_list(tmp_path):
    """``read_json`` is annotated ``-> Dict`` but never checks; a list comes back as
    is.
    """
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps([1, 2, 3]))

    assert read_json(str(path)) == [1, 2, 3]
