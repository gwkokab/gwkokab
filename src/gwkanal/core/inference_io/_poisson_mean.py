# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from jaxtyping import Array, PRNGKeyArray
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from gwkanal.utils.common import read_json
from gwkokab.poisson_mean import (
    poisson_mean_from_neural_pdet,
    poisson_mean_from_neural_vt,
    poisson_mean_from_sensitivity_injections,
)
from gwkokab.utils.tools import error_if


class BaseLoader(BaseModel):
    """Base logic shared across all loaders."""

    filename: str
    key: Any
    parameters: Tuple[str, ...]

    def get_estimators(self):
        raise NotImplementedError("Subclasses must implement this method.")


class NeuralVolumeTimeSensitivityPoissonMeanLoader(BaseLoader):
    estimator_type: Literal["neural_vt"]
    batch_size: Optional[PositiveInt] = None
    num_samples: PositiveInt = 1_000
    time_scale: PositiveFloat = 1.0

    def get_estimators(self):
        return poisson_mean_from_neural_vt(
            key=self.key,
            parameters=self.parameters,
            filename=self.filename,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            time_scale=self.time_scale,
        )


class NeuralVolumeProbabilityOfDetectionPoissonMeanLoader(BaseLoader):
    estimator_type: Literal["neural_pdet"]
    batch_size: Optional[PositiveInt] = None
    num_samples: PositiveInt = 1_000
    time_scale: PositiveFloat = 1.0

    def get_estimators(self):
        return poisson_mean_from_neural_pdet(
            key=self.key,
            parameters=self.parameters,
            filename=self.filename,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            time_scale=self.time_scale,
        )


class GWTCInjectionLoader(BaseLoader):
    estimator_type: Literal["injection"]
    batch_size: Optional[PositiveInt] = None
    far_cut: PositiveFloat = 1.0
    snr_cut: PositiveFloat = 10.0

    def get_estimators(self):
        return poisson_mean_from_sensitivity_injections(
            key=self.key,
            parameters=self.parameters,
            filename=self.filename,
            batch_size=self.batch_size,
            far_cut=self.far_cut,
            snr_cut=self.snr_cut,
        )


class CustomPoissonMeanEstimationLoader(BaseLoader):
    estimator_type: Literal["custom"]
    python_module_path: str
    kwargs: Dict[str, Any]

    def get_estimators(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.python_module_path
        )
        error_if(
            spec is None or spec.loader is None,
            ImportError,
            f"Could not load spec for module at {self.python_module_path}",
        )

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.python_module_path
        )
        custom_module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(custom_module)  # type: ignore

        error_if(
            not hasattr(custom_module, "custom_poisson_mean_estimator"),
            msg="The custom module must have a 'custom_poisson_mean_estimator' function.",
        )

        return custom_module.custom_poisson_mean_estimator(
            self.key,
            self.parameters,
            self.filename,
            **self.kwargs,
        )


class PoissonMeanEstimationLoader(BaseModel):
    loader: Union[
        NeuralVolumeTimeSensitivityPoissonMeanLoader,
        NeuralVolumeProbabilityOfDetectionPoissonMeanLoader,
        GWTCInjectionLoader,
        CustomPoissonMeanEstimationLoader,
    ] = Field(discriminator="estimator_type")

    @classmethod
    def from_json(
        cls, config_path: str, key: PRNGKeyArray, parameters: Tuple[str, ...]
    ):
        raw_data = read_json(config_path)

        payload = {"loader": {**raw_data, "key": key, "parameters": parameters}}

        return cls(**payload)

    def get_estimators(
        self,
    ) -> Tuple[
        Optional[Callable[[Array], Array]],
        Callable[..., Array],
        Callable[..., Array],
        dict[str, Any],
    ]:
        return self.loader.get_estimators()
