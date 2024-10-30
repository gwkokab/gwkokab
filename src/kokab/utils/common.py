# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import json
from collections.abc import Sequence
from typing_extensions import Callable, List

import jax.numpy as jnp
import pandas as pd
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import Uniform
from pdet import pdet_O3

from gwkokab.vts import load_model

from .regex import match_all


def expand_arguments(arg: str, n: int) -> List[str]:
    r"""Extend the argument with a number of strings.

    .. code:: python

        >>> expand_arguments("physics", 3)
        ["physics_0", "physics_1", "physics_2"]

    :param arg: argument to extend
    :param n: number of strings to extend
    :return: list of extended arguments
    """
    return [f"{arg}_{i}" for i in range(n)]


def flowMC_json_read_and_process(json_file: str) -> dict:
    """Convert a json file to a dictionary."""
    with open(json_file, "r") as f:
        flowMC_json = json.load(f)

    key_key_value = [
        ("data_dump_kwargs", "out_dir", "sampler_data"),
        ("local_sampler_kwargs", "jit", True),
        ("local_sampler_kwargs", "sampler", "MALA"),
        ("nf_model_kwargs", "model", "MaskedCouplingRQSpline"),
        ("sampler_kwargs", "data", None),
        ("sampler_kwargs", "logging", True),
        ("sampler_kwargs", "outdir", "inf-plot"),
        ("sampler_kwargs", "precompile", False),
        ("sampler_kwargs", "use_global", True),
        ("sampler_kwargs", "verbose", False),
    ]

    for key1, key2, value in key_key_value:
        flowMC_json[key1][key2] = value

    return flowMC_json


def get_posterior_data(filenames: List[str], posterior_columns: List[str]) -> dict:
    r"""Get the posterior data from a list of files.

    :param filenames: list of filenames
    :param posterior_columns: list of posterior columns
    :return: dictionary of posterior data
    """
    if len(filenames) == 0:
        raise ValueError("No files found to read posterior data")
    data_list = []
    for event in filenames:
        df = pd.read_csv(event, delimiter=" ")
        missing_columns = set(posterior_columns) - set(df.columns)
        if missing_columns:
            raise KeyError(
                f"The file '{event}' is missing required columns: {missing_columns}"
            )
        data = df[posterior_columns].to_numpy()
        data_list.append(data)
    data_set = {
        "data": data_list,
        "N": len(filenames),
    }
    return data_set


def get_processed_priors(params: List[str], priors: dict) -> dict:
    r"""Get the processed priors from a list of parameters.

    :param params: list of parameters
    :param priors: dictionary of priors
    :raises ValueError: if the prior value is invalid
    :return: dictionary of processed priors
    """
    matched_prior_params = match_all(params, priors)
    for key, value in matched_prior_params.items():
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError(f"Invalid prior value for {key}: {value}")
            matched_prior_params[key] = Uniform(
                low=value[0], high=value[1], validate_args=True
            )
    for param in params:
        if param not in matched_prior_params:
            raise ValueError(f"Missing prior for {param}")
    return matched_prior_params


def check_vt_params(vt_params: List[str], parameters: List[str]) -> None:
    r"""Check if all the parameters in the VT are in the model.

    :param vt_params: list of VT parameters
    :param parameters: list of model parameters
    :raises ValueError: if the parameters in the VT do not match the parameters in
        the model
    """
    if set(vt_params) - set(parameters):
        raise ValueError(
            "The parameters in the VT do not match the parameters in the model. "
            f"VT_PARAMS: {vt_params}, parameters: {parameters}"
        )


def get_logVT(
    vt_path: str,
    vt_params: Sequence[str],
    model_params: Sequence[int],
    key: PRNGKeyArray,
    use_pdet: Bool[bool, "True", "False"] = False,
) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    """Get the logVT function.

    :param vt_path: path to the VT model
    :param vt_params: list of VT parameters
    :param model_params: list of model parameters
    :param key: random key
    :param use_pdet: whether to use injection based VT or not
    :return: logVT function
    """
    if use_pdet:
        pdet_vt = pdet_O3(parameters=vt_params)
        return lambda param: jnp.log(pdet_vt.predict(key=key, params=param))
    else:
        _, logVT = load_model(vt_path)
        selection_indexes = [model_params.index(name) for name in vt_params]

        def trimmed_logVT(x: Float[Array, "..."]) -> Float[Array, "..."]:
            m1m2 = x[..., selection_indexes]
            return vmap(logVT)(m1m2)

        return trimmed_logVT
