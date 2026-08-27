# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`gwkokab_scripts.h5repack`.

The script is a guarded wrapper around the external ``h5repack`` binary: repack into a
sibling ``.tmp`` file, move it over the original on success, and delete it on failure.
That temporary file is the whole point — an interrupted repack must never leave a half-
written HDF5 file where the finished run used to be — so the tests focus on what
survives each outcome.

The binary itself is an HDF5 build artifact that need not be installed wherever the
tests run, so both the PATH lookup and the subprocess call are stubbed out.
"""

import shutil
import subprocess

import pytest

from gwkokab_scripts import h5repack


ORIGINAL = "original bytes"
REPACKED = "repacked bytes"


@pytest.fixture
def h5repack_on_path(monkeypatch):
    """Pretend the ``h5repack`` executable is installed."""
    monkeypatch.setattr(
        shutil, "which", lambda cmd: "/usr/bin/h5repack" if cmd == "h5repack" else None
    )


@pytest.fixture
def fake_run(monkeypatch):
    """Install a ``subprocess.run`` stub and return the list of commands it receives.

    The stub is handed the side effect to perform in place of the repack, which is how a
    test picks between a successful run, a non-zero exit and a run that never produced a
    temporary file.
    """

    def _install(side_effect=None):
        calls: list = []

        def _run(cmd, check=False, **kwargs):
            calls.append(list(cmd))
            if side_effect is not None:
                side_effect(list(cmd))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", _run)
        return calls

    return _install


def _write_repacked(cmd) -> None:
    with open(cmd[-1], "w") as f:
        f.write(REPACKED)


@pytest.fixture
def hdf5_file(tmp_path):
    path = tmp_path / "inference_data.hdf5"
    path.write_text(ORIGINAL)
    return path


def test_successful_repack_replaces_the_original(
    hdf5_file, h5repack_on_path, fake_run, run_main
):
    calls = fake_run(_write_repacked)

    run_main(h5repack.main, hdf5_file)

    assert calls == [["h5repack", str(hdf5_file), f"{hdf5_file}.tmp"]]
    assert hdf5_file.read_text() == REPACKED
    assert not (hdf5_file.parent / f"{hdf5_file.name}.tmp").exists()


def test_a_failed_repack_keeps_the_original_and_drops_the_temporary(
    hdf5_file, h5repack_on_path, fake_run, run_main
):
    def _fail(cmd):
        _write_repacked(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    fake_run(_fail)

    with pytest.raises(subprocess.CalledProcessError):
        run_main(h5repack.main, hdf5_file)

    assert hdf5_file.read_text() == ORIGINAL
    assert not (hdf5_file.parent / f"{hdf5_file.name}.tmp").exists()


def test_a_failure_before_any_output_still_propagates(
    hdf5_file, h5repack_on_path, fake_run, run_main
):
    """There is nothing to clean up when the binary died early; the cleanup must not
    trip over the missing temporary file.
    """

    def _die(cmd):
        raise OSError("h5repack died")

    fake_run(_die)

    with pytest.raises(OSError, match="h5repack died"):
        run_main(h5repack.main, hdf5_file)

    assert hdf5_file.read_text() == ORIGINAL


def test_a_failed_move_cleans_up_the_temporary(
    hdf5_file, h5repack_on_path, fake_run, run_main, monkeypatch
):
    """A full or read-only filesystem fails at the move, not at the repack."""

    def _read_only(*args):
        raise OSError("read-only")

    fake_run(_write_repacked)
    monkeypatch.setattr(shutil, "move", _read_only)

    with pytest.raises(OSError, match="read-only"):
        run_main(h5repack.main, hdf5_file)

    assert hdf5_file.read_text() == ORIGINAL
    assert not (hdf5_file.parent / f"{hdf5_file.name}.tmp").exists()


def test_a_missing_binary_is_reported(hdf5_file, monkeypatch, run_main):
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    with pytest.raises(OSError, match="'h5repack' command not found"):
        run_main(h5repack.main, hdf5_file)

    assert hdf5_file.read_text() == ORIGINAL


def test_a_missing_file_is_reported(tmp_path, h5repack_on_path, run_main):
    missing = tmp_path / "absent.hdf5"

    with pytest.raises(FileNotFoundError, match="absent.hdf5"):
        run_main(h5repack.main, missing)


def test_the_binary_is_looked_up_before_the_file(tmp_path, monkeypatch, run_main):
    """Without ``h5repack`` nothing can be repacked, so that is the failure worth
    reporting even when the input path is wrong as well.
    """
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    with pytest.raises(OSError, match="not found in PATH"):
        run_main(h5repack.main, tmp_path / "absent.hdf5")


def test_the_file_argument_is_required(run_main):
    with pytest.raises(SystemExit):
        run_main(h5repack.main)


def test_extra_options_are_forwarded_to_h5repack(
    hdf5_file, h5repack_on_path, fake_run, run_main
):
    """The usage the epilog documents: everything after ``--`` belongs to h5repack, in
    the order it was given and ahead of the two filenames.
    """
    calls = fake_run(_write_repacked)

    run_main(h5repack.main, "--", "-f", "GZIP=9", "-f", "SHUF", hdf5_file)

    assert calls == [
        [
            "h5repack",
            "-f",
            "GZIP=9",
            "-f",
            "SHUF",
            str(hdf5_file),
            f"{hdf5_file}.tmp",
        ]
    ]


def test_a_bare_separator_adds_no_options(
    hdf5_file, h5repack_on_path, fake_run, run_main
):
    calls = fake_run(_write_repacked)

    run_main(h5repack.main, "--", hdf5_file)

    assert calls == [["h5repack", str(hdf5_file), f"{hdf5_file}.tmp"]]


def test_options_without_the_separator_are_rejected(
    hdf5_file, h5repack_on_path, fake_run, run_main, capsys
):
    """Read as this script's own flags they mean nothing, and silently repacking without
    them would leave a file that only looks compressed.
    """
    calls = fake_run(_write_repacked)

    with pytest.raises(SystemExit):
        run_main(h5repack.main, "-f", "GZIP=9", hdf5_file)

    assert calls == []
    assert "unrecognized arguments: -f" in capsys.readouterr().err
