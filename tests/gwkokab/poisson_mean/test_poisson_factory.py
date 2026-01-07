# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os

import numpy as np
import pytest

from gwkokab.models import PowerlawPeak
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean.poisson_factory import (
    get_selection_fn_and_poisson_mean_estimator,
)


def test_poisson_mean_estimator_invalid_type():
    with pytest.raises(ValueError, match="estimator_type must be one of"):
        get_selection_fn_and_poisson_mean_estimator(
            estimator_type="invalid_type",
            key=None,
            parameters=None,
            batch_size=None,
            filename=None,
        )


def test_custom_poisson_mean_estimator_no_python_module_path():
    with pytest.raises(
        ValueError,
        match="For custom estimator_type, 'python_module_path' must be provided.",
    ):
        get_selection_fn_and_poisson_mean_estimator(
            estimator_type="custom",
            key=None,
            parameters=None,
            batch_size=None,
            filename=None,
        )


def test_custom_poisson_mean_estimator_invalid_python_module_path():
    with pytest.raises(
        ValueError,
        match="'python_module_path' must be a string, got",
    ):
        get_selection_fn_and_poisson_mean_estimator(
            estimator_type="custom",
            key=None,
            parameters=None,
            batch_size=None,
            filename=None,
            python_module_path=123,  # type: ignore
        )

    with pytest.raises(
        ValueError,
        match="'python_module_path' must point to a .py file, got",
    ):
        get_selection_fn_and_poisson_mean_estimator(
            estimator_type="custom",
            key=None,
            parameters=None,
            batch_size=None,
            filename=None,
            python_module_path="",  # type: ignore
        )


def test_custom_poisson_mean_estimator_no_estimator_function():
    with pytest.raises(
        ValueError,
        match="The custom module must have a 'custom_poisson_mean_estimator' function.",
    ):
        get_selection_fn_and_poisson_mean_estimator(
            estimator_type="custom",
            key=None,
            parameters=None,
            batch_size=None,
            filename=None,
            python_module_path=os.path.abspath(__file__),
        )


@pytest.mark.xfail(
    reason="OSError: Unable to synchronously open file (file signature not found)"
)
def test_custom_poisson_mean_estimator_success():
    CWD = os.getcwd()
    filename = "o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
    filepath = os.path.join(CWD, "tests/gwkokab/poisson_mean", filename)
    if not os.path.exists(filepath):
        url = r"https://zenodo.org/records/7890398/files/" + filename
        command = f"wget -c {url} -O {filepath}"
        os.system(command)

    python_module_path = filepath = os.path.join(
        CWD, "tests/gwkokab/poisson_mean/test_custom_injection_based_pdet.py"
    )
    assert os.path.exists(python_module_path), (
        f"Custom python module not found at {python_module_path}"
    )

    _, poisson_mean_estimator, _ = get_selection_fn_and_poisson_mean_estimator(
        estimator_type="custom",
        key=None,
        parameters=[
            P.PRIMARY_MASS_SOURCE,
            P.SECONDARY_MASS_SOURCE,
            P.PRIMARY_SPIN_MAGNITUDE,
            P.SECONDARY_SPIN_MAGNITUDE,
            P.ECCENTRICITY,
            P.REDSHIFT,
        ],
        batch_size=32,
        filename=filepath,
        python_module_path=python_module_path,
    )

    model = PowerlawPeak(
        use_spin=True,
        use_redshift=True,
        use_eccentricity=True,
        use_tilt=True,
        validate_args=True,
        alpha=1.0,
        beta=1.0,
        chi_mean=0.2,
        chi_variance=0.025,
        cos_tilt_scale=3.0,
        cos_tilt_zeta=0.6,
        delta=5.0,
        eccentricity_scale=3.0,
        kappa=3.0,
        lambda_peak=0.4,
        loc=32.0,
        log_rate=4.0,
        mmax=200.0,
        mmin=2.0,
        scale=3.0,
        z_max=2.3,
    )

    value = poisson_mean_estimator(model)

    assert value.shape == (), "Poisson mean estimator did not return a scalar value."
    assert np.isnan(value), "Poisson mean estimator returned NaN value."
