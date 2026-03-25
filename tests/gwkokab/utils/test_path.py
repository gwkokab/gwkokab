# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from gwkokab.utils.path import normalize_path


@pytest.fixture
def mock_home(monkeypatch, tmp_path):
    """Simulates a clean home directory to avoid touching your real ~/ during
    testing.
    """
    fake_home = tmp_path / "home" / "linuxuser"
    fake_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    return fake_home


def test_empty_string():
    """Should return the current working directory (standard Path behavior)."""
    result = normalize_path("")
    assert result == Path.cwd()


def test_non_existent_env_variable():
    """os.path.expandvars leaves $VAR as-is if not found.

    Ensure the function doesn't crash but returns the literal string path.
    """
    raw = "/tmp/$NON_EXISTENT_VAR/file.txt"
    result = normalize_path(raw)
    assert "$NON_EXISTENT_VAR" in str(result)


def test_multiple_slashes():
    """Ensure redundant slashes are collapsed (e.g., //// becomes /)."""
    raw = "/tmp///folder//file.txt"
    result = normalize_path(raw)
    # .resolve() or Path stringification handles this
    assert "//" not in str(result)[1:]


def test_relative_path_resolution():
    """Verify that a simple relative path is converted to an absolute path."""
    raw = "local_file.txt"
    result = normalize_path(raw)
    assert result.is_absolute()
    assert result == Path.cwd() / "local_file.txt"


def test_nested_variables(monkeypatch):
    """Test variables that contain other variables or tildes."""
    monkeypatch.setenv("DOCS", "~/my_docs")
    # Note: expandvars doesn't recurse by default, but let's check behavior
    raw = "$DOCS/report.pdf"
    result = normalize_path(raw)
    # Since expandvars runs before expanduser, this should work!
    expected = Path.home() / "my_docs" / "report.pdf"
    assert result == expected


@pytest.mark.parametrize(
    "weird_path",
    [
        "/tmp/space in name/file.csv",
        "/tmp/special-chars-!@#/file.txt",
        "/tmp/trailing_slash/",
    ],
)
def test_path_formatting_edge_cases(weird_path):
    """Ensure spaces and special characters don't break the Path object."""
    result = normalize_path(weird_path)
    assert result.name or result.parent


def test_linux_env_expansion(monkeypatch):
    """Test standard $VAR and ${VAR} syntax common in Bash."""
    monkeypatch.setenv("APP_LOGS", "/var/log/myapp")

    assert str(normalize_path("$APP_LOGS/error.log")) == "/var/log/myapp/error.log"
    assert str(normalize_path("${APP_LOGS}/debug.log")) == "/var/log/myapp/debug.log"


def test_tilde_expansion(mock_home):
    """Test that ~ correctly maps to the Linux HOME variable."""
    raw = "~/downloads/video.mp4"
    result = normalize_path(raw)

    assert result == mock_home / "downloads" / "video.mp4"
    assert result.is_absolute()


def test_root_path_resolution():
    """Ensure absolute paths starting with / are preserved and cleaned."""
    raw = "///etc/../etc/systemd/../systemd/journald.conf"
    result = normalize_path(raw)

    # Expected: /etc/systemd/journald.conf
    assert str(result) == "/etc/systemd/journald.conf"


def test_hidden_files_and_spaces():
    """Linux allows spaces and dots in filenames; ensure they aren't mangled."""
    raw = "/tmp/.hidden folder/my config.conf"
    result = normalize_path(raw)

    assert result.name == "my config.conf"
    assert ".hidden folder" in result.parts


def test_undefined_variable_behavior():
    """On Linux, if a bash var is undefined, it usually evaluates to empty.

    However, os.path.expandvars leaves it literal. We test for this consistency.
    """
    raw = "/tmp/$TOTALLY_FAKE_VARIABLE/data"
    result = normalize_path(raw)
    assert "$TOTALLY_FAKE_VARIABLE" in str(result)


def test_current_dir_context():
    """Test that relative paths resolve based on the current working directory."""
    raw = "./logs/current.log"
    result = normalize_path(raw)

    expected = Path.cwd() / "logs" / "current.log"
    assert result == expected


def test_non_string_input():
    """Edge case: What if someone passes an integer or None?"""
    with pytest.raises(TypeError):
        normalize_path(None)
    with pytest.raises(TypeError):
        normalize_path(123)


def test_path_object_input(mock_home):
    """Ensure Path objects are handled correctly."""
    raw = Path("~/some_dir/file.txt")
    result = normalize_path(raw)
    assert result == mock_home / "some_dir" / "file.txt"
