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


import json
import warnings
from collections.abc import Sequence
from typing import List, Union

import jax
import numpy as np
import pandas as pd
from numpyro import distributions as dist
from numpyro.distributions.distribution import DistributionLike

from gwkokab.vts import available as available_vts, VolumeTimeSensitivityInterface
from kokab.utils.priors import available as available_priors
from kokab.utils.regex import match_all


def read_json(json_file: str) -> dict:
    """Read json file and return.

    Parameters
    ----------
    json_file : str
        path of the json file

    Returns
    -------
    dict
        json file content as dict

    Raises
    ------
    ValueError
        If the file is not found or if the file is not a valid json file
    """
    try:
        with open(json_file, "r") as f:
            content = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading configuration: {e}")
    return content


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


def flowMC_default_parameters(**kwargs: dict) -> dict:
    """Convert a json file to a dictionary."""

    key_key_value = [
        ("data_dump_kwargs", "out_dir", "sampler_data"),
        ("local_sampler_kwargs", "jit", True),
        ("nf_model_kwargs", "model", "MaskedCouplingRQSpline"),
        ("sampler_kwargs", "data", None),
        ("sampler_kwargs", "logging", True),
        ("sampler_kwargs", "outdir", "inf-plot"),
        ("sampler_kwargs", "precompile", False),
        ("sampler_kwargs", "use_global", True),
        ("sampler_kwargs", "verbose", False),
    ]

    for key1, key2, value in key_key_value:
        if kwargs[key1].get(key2) is None:
            kwargs[key1][key2] = value

    local_sampler_name = kwargs["local_sampler_kwargs"].get("sampler")
    if local_sampler_name is None:
        raise ValueError("Local sampler name is not provided.")

    if local_sampler_name == "HMC":
        condition_matrix = kwargs["local_sampler_kwargs"].get("condition_matrix")
        if condition_matrix is None:
            warnings.warn(
                "HMC Sampler: `condition_matrix` is not provided. Using identity matrix."
            )
            condition_matrix = np.eye(kwargs["sampler_kwargs"]["n_dim"])
        kwargs["local_sampler_kwargs"]["condition_matrix"] = np.asarray(
            condition_matrix
        )

    return kwargs


def get_posterior_data(
    filenames: List[str], posterior_columns: List[str]
) -> List[np.ndarray]:
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
    data_list = jax.device_put(data_list, may_alias=True)
    return data_list


def get_vt_samples(filename: str, columns: List[str]) -> np.ndarray:
    r"""Get the VT samples from a list of files.

    :param filenames: list of filenames
    :param columns: list of columns
    :return: dictionary of VT samples
    """
    df = pd.read_csv(filename, delimiter=" ")
    missing_columns = set(columns) - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"The file '{filename}' is missing required columns: {missing_columns}"
        )
    data = df[columns].to_numpy()
    return data


def get_processed_priors(params: List[str], priors: dict) -> dict:
    r"""Get the processed priors from a list of parameters.

    :param params: list of parameters
    :param priors: dictionary of priors
    :raises ValueError: if the prior value is invalid
    :return: dictionary of processed priors
    """
    matched_prior_params = match_all(params, priors)
    for key, value in matched_prior_params.items():
        if isinstance(value, dict):
            value_cpy = value.copy()
            dist_type = value_cpy.pop("dist")
            matched_prior_params[key] = available_priors[dist_type](**value_cpy)
    for param in params:
        if param not in matched_prior_params:
            raise ValueError(f"Missing prior for {param}")
    return matched_prior_params


def check_vt_params(vt_params: List[str], parameters: List[str]) -> None:
    r"""Check if all the parameters in the VT are in the model.

    :param vt_params: list of VT parameters
    :param parameters: list of model parameters
    :raises ValueError: if the parameters in the VT do not match the parameters in the
        model
    """
    if set(vt_params) - set(parameters):
        raise ValueError(
            "The parameters in the VT do not match the parameters in the model. "
            f"VT_PARAMS: {vt_params}, parameters: {parameters}"
        )


def vt_json_read_and_process(
    parameters: Sequence[str], vt_path: str, settings_path: str
) -> VolumeTimeSensitivityInterface:
    r"""Read and process the VT JSON file.

    :param parameters: list of parameters
    :param vt_path: path to the VT
    :param settings_path: path to the VT settings
    :raises ValueError: if the VT is not found
    :return: VT object
    """
    with open(settings_path, "r") as f:
        vt_settings = json.load(f)

    vt_type = vt_settings["type"]
    vt_settings.pop("type")
    vt = available_vts[vt_type]
    return vt(parameters, vt_path, **vt_settings)


def get_dist(meta_dict: dict[str, Union[str, float]]) -> DistributionLike:
    """Get the distribution from the dictionary. It expects the dictionary to have the
    key 'dist' which is the name of the distribution and the rest of the keys to be the
    parameters of the distribution.

    Example
    -------
    >>> std_normal = get_dist({"dist": "Normal", "loc": 0.0, "scale": 1.0})
    >>> std_normal.loc
    0.0
    >>> std_normal.scale
    1.0

    Parameters
    ----------
    meta_dict : dict[str, Union[str, float]]
        Dictionary containing the distribution name and its parameters

    Returns
    -------
    DistributionLike
        The distribution object
    """
    dist_name = meta_dict.pop("dist")
    return getattr(dist, dist_name)(**meta_dict)
