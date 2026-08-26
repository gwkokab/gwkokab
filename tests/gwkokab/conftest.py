# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Dict, Optional, Sequence

import h5py
import numpy as np
import pytest

from gwkokab.constants import SECONDS_PER_YEAR


def _write_linear_model(
    path: str, names: Sequence[str], coefficients: Sequence[float], bias: float
) -> None:
    r"""Write an HDF5 file readable by :func:`gwkokab.utils.train.load_model` whose MLP
    is exactly the affine map :math:`x \mapsto c \cdot x + b`.

    The MLP built by ``load_model`` has a ReLU between its two layers, so the identity is
    encoded through the first layer as :math:`\mathrm{ReLU}(x) - \mathrm{ReLU}(-x) = x`:
    the first layer stacks :math:`+I` on top of :math:`-I`, and the second layer contracts
    the two halves with :math:`+c` and :math:`-c`.
    """
    n = len(names)
    weight_0 = np.concatenate([np.eye(n), -np.eye(n)], axis=0)
    bias_0 = np.zeros(2 * n)
    weight_1 = np.concatenate([coefficients, np.negative(coefficients)])[None, :]
    bias_1 = np.asarray([bias], dtype=float)

    with h5py.File(path, "w") as f:
        f.create_dataset("names", data=np.array(names, dtype="S"))
        f.create_dataset("in_size", data=n)
        f.create_dataset("out_size", data=1)
        f.create_dataset("width_size", data=2 * n)
        f.create_dataset("depth", data=1)
        for i, (weight, bias_i) in enumerate(((weight_0, bias_0), (weight_1, bias_1))):
            group = f.create_group(f"layer_{i}")
            group.create_dataset(f"weight_{i}", data=weight)
            group.create_dataset(f"bias_{i}", data=bias_i)


@pytest.fixture
def linear_model_file(tmp_path) -> Callable[..., str]:
    """Factory writing a neural network file whose output is a known affine function.

    The returned callable takes the parameter `names` the model was trained on, the per-
    parameter `coefficients` and an additive `bias`, and returns the path of the written
    file. The network then evaluates to ``sum(coefficients * x) + bias``, so a zero
    coefficient vector yields a *constant* network, for which the Poisson mean and its
    variance are available in closed form.
    """
    counter = {"n": 0}

    def _make(
        names: Sequence[str], coefficients: Sequence[float], bias: float = 0.0
    ) -> str:
        path = str(tmp_path / f"model_{counter['n']}.hdf5")
        counter["n"] += 1
        _write_linear_model(path, names, np.asarray(coefficients, dtype=float), bias)
        return path

    return _make


@pytest.fixture
def injections_file(tmp_path) -> Callable[..., str]:
    """Factory writing a sensitivity-injection file in the O3 ``injections`` layout.

    ``sampling_pdf`` is chosen so that the ``prior`` recovered by
    :func:`~gwkokab.poisson_mean._injection_based_helper.load_injection_data` is exactly
    `prior_value` for every injection, which keeps the Poisson mean analytic.
    """
    counter = {"n": 0}

    def _make(
        *,
        mass_1: Optional[np.ndarray] = None,
        mass_2: Optional[np.ndarray] = None,
        redshift: Optional[np.ndarray] = None,
        ifar: Optional[np.ndarray] = None,
        total_generated: int = 100,
        analysis_time_years: float = 2.0,
        prior_value: float = 1.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        mass_1 = np.linspace(10.0, 50.0, 8) if mass_1 is None else np.asarray(mass_1)
        n = mass_1.size
        mass_2 = np.linspace(5.0, 25.0, n) if mass_2 is None else np.asarray(mass_2)
        redshift = (
            np.linspace(0.05, 0.9, n) if redshift is None else np.asarray(redshift)
        )
        ifar = np.full(n, 1e3) if ifar is None else np.asarray(ifar)

        # in-plane spins are zero, so a_i reduces to |spin_iz|
        spin_1z = np.full(n, 0.3)
        spin_2z = np.full(n, -0.2)
        # load_injection_data multiplies sampling_pdf by (2 pi a_1 a_2)**2
        sampling_pdf = prior_value / np.square(
            2.0 * np.pi * np.abs(spin_1z) * np.abs(spin_2z)
        )

        path = str(tmp_path / f"injections_{counter['n']}.hdf5")
        counter["n"] += 1
        with h5py.File(path, "w") as f:
            group = f.create_group("injections")
            group.attrs["total_generated"] = np.int64(total_generated)
            group.attrs["analysis_time_s"] = np.float64(
                analysis_time_years * SECONDS_PER_YEAR
            )
            datasets = {
                "mass1_source": mass_1,
                "mass2_source": mass_2,
                "redshift": redshift,
                "spin1x": np.zeros(n),
                "spin1y": np.zeros(n),
                "spin1z": spin_1z,
                "spin2x": np.zeros(n),
                "spin2y": np.zeros(n),
                "spin2z": spin_2z,
                "sampling_pdf": sampling_pdf,
                "ifar_gstlal": ifar,
            }
            datasets.update(extra or {})
            for key, value in datasets.items():
                group.create_dataset(key, data=value)
        return path

    return _make
