# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the :mod:`gwkokab_scripts` console entry points.

Every script in the package is a thin ``main()`` wrapped around ``sys.argv``: it parses
arguments, reads an HDF5 or JSON file, and then either rewrites a file or prints a
report. Testing one therefore means driving ``main()`` with an argument vector and
inspecting what it left behind — a file, the current matplotlib figure, or stdout — so
the fixtures here cover exactly those three.
"""

import sys
import warnings
from typing import Callable, Mapping, Optional, Sequence

import h5py
import matplotlib
import numpy as np
import pytest


# The plotting scripts must never try to open a window, on a developer machine or in
# CI. Selected here, before any fixture imports ``pyplot``.
matplotlib.use("Agg")

# ``glasbey``, which ``hist`` and ``scatter2d`` import for their palettes, pulls in
# ``colorspacious``, whose docstrings carry invalid escape sequences. The SyntaxWarning
# that raises is emitted when the module is *compiled*, so it only appears on a cold
# bytecode cache — a fresh install, which is to say every CI run — and this project
# turns warnings into errors. Compile it once here, quietly, rather than inside a test.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import glasbey  # noqa: F401


@pytest.fixture(autouse=True)
def _isolated_pyplot():
    """Keep the global pyplot state from leaking between tests.

    ``hist``, ``scatter2d`` and ``flowMC_info`` draw on the implicit *current* figure
    and mutate ``rcParams`` (``figure.constrained_layout.use``, the ``bmh`` style),
    neither of which they undo. Without this fixture a test would inherit the axes and
    the styling of whichever test ran before it.
    """
    from matplotlib import pyplot as plt

    plt.close("all")
    with matplotlib.rc_context():
        yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Widen the console :mod:`rich` renders into.

    Under captured output rich sees a non-terminal and falls back to 80 columns, which
    wraps the tables and panels the scripts print and would break assertions that match
    a whole line.
    """
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def run_main(monkeypatch) -> Callable[..., None]:
    """Invoke a script's ``main()`` under a chosen argument vector.

    ``argparse`` reads ``sys.argv`` directly and treats its first entry as the program
    name, so the returned callable prepends one. Every argument is stringified, which
    lets tests pass :class:`~pathlib.Path` objects and numbers as they are.
    """

    def _run(main: Callable[[], None], *argv) -> None:
        monkeypatch.setattr(sys, "argv", ["gwk_script", *(str(arg) for arg in argv)])
        return main()

    return _run


@pytest.fixture
def samples_file(tmp_path) -> Callable[..., str]:
    """Factory writing the ``samples`` half of an ``inference_data.hdf5`` file.

    ``variables_index`` is a *group* carrying one attribute per hyper-parameter name,
    the layout :func:`gwkokab.analysis.core.utils.write_to_hdf5` produces when it is
    handed attributes but no data.
    """
    counter = {"n": 0}

    def _make(
        samples: np.ndarray,
        variables_index: Optional[Mapping[str, int]] = None,
    ) -> str:
        path = str(tmp_path / f"posterior_{counter['n']}.hdf5")
        counter["n"] += 1
        with h5py.File(path, "w") as f:
            f.create_dataset("samples", data=np.asarray(samples))
            if variables_index is not None:
                group = f.require_group("variables_index")
                for name, index in variables_index.items():
                    group.attrs[name] = index
        return path

    return _make


@pytest.fixture
def compound_file(tmp_path) -> Callable[..., str]:
    """Factory writing the compound-dataset layout the plotting scripts read.

    ``hist`` and ``scatter2d`` both index a dataset by field name, so the dataset has to
    be a structured array rather than a plain 2-D block.
    """
    counter = {"n": 0}

    def _make(
        fields: Mapping[str, Sequence[float]],
        dataset: str = "events/posterior",
    ) -> str:
        arrays = {
            name: np.asarray(values, dtype=float) for name, values in fields.items()
        }
        size = len(next(iter(arrays.values())))
        data = np.zeros(size, dtype=np.dtype([(name, "f8") for name in arrays]))
        for name, values in arrays.items():
            data[name] = values

        path = str(tmp_path / f"events_{counter['n']}.hdf5")
        counter["n"] += 1
        with h5py.File(path, "w") as f:
            f.create_dataset(dataset, data=data)
        return path

    return _make
