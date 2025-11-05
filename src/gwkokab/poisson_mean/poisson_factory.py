# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import Optional, Tuple

from jaxtyping import Array, PRNGKeyArray

from ..models.utils import ScaledMixture
from ..utils.tools import error_if
from ._injection_based import (
    poisson_mean_from_sensitivity_injections as _poisson_mean_from_sensitivity_injections,
)
from ._neural_pdet import (
    poisson_mean_from_neural_pdet as _poisson_mean_from_neural_pdet,
)
from ._neural_vt import (
    poisson_mean_from_neural_vt as _poisson_mean_from_neural_vt,
)


def get_selection_fn_and_poisson_mean_estimator(
    estimator_type: str,
    key: PRNGKeyArray,
    parameters: Sequence[str],
    filename: str,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Tuple[
    Optional[Callable[[Array], Array]],
    Callable[[ScaledMixture], Array],
    float | Array,
    Callable[[ScaledMixture], Array],
]:
    valid_estimator_types = ("injection", "neural_vt", "neural_pdet", "custom")
    error_if(
        estimator_type not in valid_estimator_types,
        msg="estimator_type must be one of " + ", ".join(valid_estimator_types),
    )
    if estimator_type == "injection":
        ifar_pipelines = kwargs.pop("ifar_pipelines", None)
        return _poisson_mean_from_sensitivity_injections(
            key=key,
            parameters=parameters,
            filename=filename,
            batch_size=batch_size,
            far_cut=kwargs.pop("far_cut", 1.0),
            snr_cut=kwargs.pop("snr_cut", 10.0),
            ifar_pipelines=ifar_pipelines,
        )
    elif estimator_type == "neural_vt":
        return _poisson_mean_from_neural_vt(
            key=key,
            parameters=parameters,
            filename=filename,
            batch_size=batch_size,
            num_samples=kwargs.pop("num_samples", 1_000),
            time_scale=kwargs.pop("time_scale", 1.0),
        )
    elif estimator_type == "neural_pdet":
        return _poisson_mean_from_neural_pdet(
            key=key,
            parameters=parameters,
            filename=filename,
            batch_size=batch_size,
            num_samples=kwargs.pop("num_samples", 1_000),
            time_scale=kwargs.pop("time_scale", 1.0),
        )
    elif estimator_type == "custom":
        error_if(
            kwargs.get("python_module_path", None) is None,
            msg="For custom estimator_type, 'python_module_path' must be provided.",
        )

        python_module_path: str = kwargs.pop("python_module_path")
        error_if(
            not isinstance(python_module_path, str),
            msg="'python_module_path' must be a string, got "
            + str(type(python_module_path)),
        )
        error_if(
            not python_module_path.endswith(".py"),
            msg="'python_module_path' must point to a .py file, got '"
            + python_module_path
            + "'.",
        )

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "custom_module", python_module_path
        )  # type: ignore
        custom_module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(custom_module)  # type: ignore

        error_if(
            not hasattr(custom_module, "custom_poisson_mean_estimator"),
            msg="The custom module must have a 'custom_poisson_mean_estimator' function.",
        )

        return custom_module.custom_poisson_mean_estimator(
            key, parameters, filename, batch_size=batch_size, **kwargs
        )

    else:
        raise ValueError(f"Unknown estimator_type: {estimator_type}")
