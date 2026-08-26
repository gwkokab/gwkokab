# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json

import pytest
from jax import random as jrd
from pydantic import ValidationError

from gwkokab.analysis.core.inference_io import PoissonMeanEstimationLoader
from gwkokab.analysis.core.inference_io._poisson_mean import (
    CustomPoissonMeanEstimationLoader,
    GWTCInjectionLoader,
    NeuralVolumeProbabilityOfDetectionPoissonMeanLoader,
    NeuralVolumeTimeSensitivityPoissonMeanLoader,
)
from gwkokab.parameters import Parameters as P
from gwkokab.utils.exceptions import LoggedImportError, LoggedValueError


PARAMETERS = (P.REDSHIFT.value, P.PRIMARY_MASS_SOURCE.value)
NAMES = [P.PRIMARY_MASS_SOURCE.value, P.REDSHIFT.value]
INJECTION_PARAMETERS = (
    P.PRIMARY_MASS_SOURCE.value,
    P.SECONDARY_MASS_SOURCE.value,
    P.REDSHIFT.value,
)


def _write_config(tmp_path, payload) -> str:
    path = tmp_path / "pmean_cfg.json"
    path.write_text(json.dumps(payload))
    return str(path)


@pytest.mark.parametrize(
    "estimator_type, expected",
    [
        ("neural_vt", NeuralVolumeTimeSensitivityPoissonMeanLoader),
        ("neural_pdet", NeuralVolumeProbabilityOfDetectionPoissonMeanLoader),
        ("injection", GWTCInjectionLoader),
    ],
)
def test_discriminator_selects_the_loader(tmp_path, estimator_type, expected):
    config = _write_config(
        tmp_path, {"estimator_type": estimator_type, "filename": "unused.hdf5"}
    )
    loader = PoissonMeanEstimationLoader.read_from_json(config, jrd.key(0), PARAMETERS)
    assert isinstance(loader.loader, expected)
    assert loader.loader.parameters == PARAMETERS


def test_unknown_estimator_type(tmp_path):
    config = _write_config(
        tmp_path, {"estimator_type": "telepathy", "filename": "unused.hdf5"}
    )
    with pytest.raises(ValidationError):
        PoissonMeanEstimationLoader.read_from_json(config, jrd.key(0), PARAMETERS)


def test_unknown_field_is_rejected(tmp_path):
    """``extra="forbid"`` turns a typo into an error instead of a silent default."""
    config = _write_config(
        tmp_path,
        {"estimator_type": "neural_vt", "filename": "unused.hdf5", "num_sample": 10},
    )
    with pytest.raises(ValidationError, match="num_sample"):
        PoissonMeanEstimationLoader.read_from_json(config, jrd.key(0), PARAMETERS)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_samples": 0},
        {"num_samples": -5},
        {"time_scale": 0.0},
        {"batch_size": 0},
    ],
)
def test_non_positive_knobs_are_rejected(tmp_path, overrides):
    config = _write_config(
        tmp_path,
        {"estimator_type": "neural_vt", "filename": "unused.hdf5", **overrides},
    )
    with pytest.raises(ValidationError):
        PoissonMeanEstimationLoader.read_from_json(config, jrd.key(0), PARAMETERS)


def test_defaults(tmp_path):
    config = _write_config(
        tmp_path, {"estimator_type": "neural_vt", "filename": "unused.hdf5"}
    )
    loader = PoissonMeanEstimationLoader.read_from_json(
        config, jrd.key(0), PARAMETERS
    ).loader
    assert loader.batch_size is None
    assert loader.num_samples == 1_000
    assert loader.time_scale == 1.0


@pytest.mark.parametrize("estimator_type", ["neural_vt", "neural_pdet"])
def test_neural_loaders_build_estimators(tmp_path, linear_model_file, estimator_type):
    filename = linear_model_file(NAMES, [0.0, 0.0], 0.5)
    config = _write_config(
        tmp_path,
        {
            "estimator_type": estimator_type,
            "filename": filename,
            "num_samples": 16,
            "time_scale": 2.5,
        },
    )
    loader = PoissonMeanEstimationLoader.read_from_json(config, jrd.key(0), PARAMETERS)
    log_sensitivity, poisson_mean, kwargs = loader.get_estimators()

    assert callable(log_sensitivity)
    assert callable(poisson_mean)
    assert kwargs == {"T_obs": 2.5}


def test_injection_loader_builds_estimators(tmp_path, injections_file):
    config = _write_config(
        tmp_path,
        {
            "estimator_type": "injection",
            "filename": injections_file(total_generated=64),
            "far_cut": 1.0,
            "snr_cut": 10.0,
        },
    )
    loader = PoissonMeanEstimationLoader.read_from_json(
        config, jrd.key(0), INJECTION_PARAMETERS
    )
    log_sensitivity, poisson_mean, kwargs = loader.get_estimators()

    assert log_sensitivity is None
    assert callable(poisson_mean)
    assert kwargs["samples"].shape == (8, 3)


def test_custom_loader_forwards_arguments(tmp_path):
    module = tmp_path / "custom_estimator.py"
    module.write_text(
        "def custom_poisson_mean_estimator(key, parameters, filename, **kwargs):\n"
        "    return None, kwargs, {'parameters': tuple(parameters), "
        "'filename': filename}\n"
    )
    config = _write_config(
        tmp_path,
        {
            "estimator_type": "custom",
            "filename": "sensitivity.hdf5",
            "python_module_path": str(module),
            "kwargs": {"scale": 3.0},
        },
    )
    loader = PoissonMeanEstimationLoader.read_from_json(config, jrd.key(0), PARAMETERS)
    assert isinstance(loader.loader, CustomPoissonMeanEstimationLoader)

    _, forwarded_kwargs, info = loader.get_estimators()
    assert forwarded_kwargs == {"scale": 3.0}
    assert info == {"parameters": PARAMETERS, "filename": "sensitivity.hdf5"}


def test_custom_loader_without_the_expected_function(tmp_path):
    module = tmp_path / "custom_estimator.py"
    module.write_text("def something_else():\n    return None\n")
    config = _write_config(
        tmp_path,
        {
            "estimator_type": "custom",
            "filename": "sensitivity.hdf5",
            "python_module_path": str(module),
            "kwargs": {},
        },
    )
    loader = PoissonMeanEstimationLoader.read_from_json(config, jrd.key(0), PARAMETERS)
    with pytest.raises(LoggedValueError, match="custom_poisson_mean_estimator"):
        loader.get_estimators()


def test_custom_loader_with_an_unimportable_path(tmp_path):
    module = tmp_path / "custom_estimator.txt"
    module.write_text("not python\n")
    config = _write_config(
        tmp_path,
        {
            "estimator_type": "custom",
            "filename": "sensitivity.hdf5",
            "python_module_path": str(module),
            "kwargs": {},
        },
    )
    loader = PoissonMeanEstimationLoader.read_from_json(config, jrd.key(0), PARAMETERS)
    with pytest.raises(LoggedImportError, match="Could not load spec"):
        loader.get_estimators()
