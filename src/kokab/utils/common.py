# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
import os
import warnings
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple, Union

import arviz as az
import jax
import numpy as np
import numpyro
import pandas as pd
from numpyro import distributions as dist
from numpyro._typing import DistributionLike

from gwkokab.vts import available as available_vts, VolumeTimeSensitivityInterface
from kokab.utils.priors import available as available_priors
from kokab.utils.regex import match_all


LOG_REF_PRIOR_NAME: str = "log_prior"
"""Helper variable to store the column name of log reference prior."""


def read_json(json_file: str) -> Dict:
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


def write_json(json_file: str, content: Dict) -> None:
    """Write a dictionary to a json file.

    Parameters
    ----------
    json_file : str
        path of the json file
    content : dict
        content to write to the json file

    Raises
    ------
    ValueError
        If the file is not writable or if the content is not a valid json serializable object
    """
    try:
        with open(json_file, "w") as f:
            json.dump(content, f, indent=4)
    except (FileNotFoundError, TypeError) as e:
        raise ValueError(f"Error writing configuration: {e}")


def expand_arguments(arg: str, n: int) -> List[str]:
    """Extend the argument with a number of strings.

    .. code:: python

        >>> expand_arguments("physics", 3)
        ["physics_0", "physics_1", "physics_2"]

    Parameters
    ----------
    arg : str
        argument to extend
    n : int
        number of strings to extend

    Returns
    -------
    List[str]
        list of extended arguments
    """
    return [f"{arg}_{i}" for i in range(n)]


def flowMC_default_parameters(**kwargs) -> dict:
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
    gradient_checkpoint_policy_name: Optional[str] = kwargs.get(
        "gradient_checkpoint_policy"
    )
    if gradient_checkpoint_policy_name is not None:
        try:
            kwargs["gradient_checkpoint_policy"] = getattr(
                jax.checkpoint_policies, gradient_checkpoint_policy_name
            )
        except AttributeError:
            raise ValueError(
                f"Invalid gradient checkpoint policy: {gradient_checkpoint_policy_name}\n"
                "Please choose from `jax.checkpoint_policies` available at "
                "https://docs.jax.dev/en/latest/gradient-checkpointing.html#list-of-policies"
            )

    return kwargs


def get_posterior_data(
    filenames: List[str], posterior_columns: List[str]
) -> List[np.ndarray]:
    """Get the posterior data from a list of files.

    Parameters
    ----------
    filenames : List[str]
        list of filenames
    posterior_columns : List[str]
        list of posterior columns

    Returns
    -------
    List[np.ndarray]
        dictionary of posterior data

    Raises
    ------
    ValueError
        If no files are found to read posterior data
    KeyError
        If the file is missing required columns
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
    return data_list


def get_vt_samples(filename: str, columns: List[str]) -> np.ndarray:
    """Get the VT samples from a list of files.

    Parameters
    ----------
    filename : str
        list of filenames
    columns : List[str]
        list of columns

    Returns
    -------
    np.ndarray
        dictionary of VT samples

    Raises
    ------
    KeyError
        If the file is missing required columns
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
    """Get the processed priors from a list of parameters.

    Parameters
    ----------
    params : List[str]
        list of parameters
    priors : dict
        dictionary of priors

    Returns
    -------
    dict
        dictionary of processed priors

    Raises
    ------
    ValueError
        if the prior value is invalid
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
    """Check if all the parameters in the VT are in the model.

    Parameters
    ----------
    vt_params : List[str]
        list of VT parameters
    parameters : List[str]
        list of model parameters

    Raises
    ------
    ValueError
        if the parameters in the VT do not match the parameters in the model
    """
    if set(vt_params) - set(parameters):
        raise ValueError(
            "The parameters in the VT do not match the parameters in the model. "
            f"VT_PARAMS: {vt_params}, parameters: {parameters}"
        )


def vt_json_read_and_process(
    parameters: Sequence[str], settings_path: str
) -> VolumeTimeSensitivityInterface:
    """Read and process the VT JSON file.

    Parameters
    ----------
    parameters : Sequence[str]
        list of parameters
    settings_path : str
        path to the VT settings

    Returns
    -------
    VolumeTimeSensitivityInterface
        VT object
    """
    vt_settings = read_json(settings_path)
    vt_type = vt_settings.pop("type")
    vt = available_vts[vt_type]
    if vt_type == "pdet_O3":
        return vt(parameters=parameters, **vt_settings)
    else:
        vt_path = vt_settings.pop("filename")
        return vt(parameters=parameters, filename=vt_path, **vt_settings)


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


def ppd_ranges(
    parameters: List[str], ranges: List[Tuple[str, float, float, int]]
) -> List[Tuple[float, float, int]]:
    """Convert the PPD ranges to the format required by the PPD function.

    Parameters
    ----------
    parameters : List[str]
        list of parameters
    ranges : List[Tuple[str, float, float, int]]
        list of ranges

    Returns
    -------
    List[Tuple[float, float, int]]
        list of ranges

    Raises
    ------
    ValueError
        If the parameter is not found in the list of parameters
    """
    _ranges: List[Tuple[float, float, int]] = [None] * len(parameters)
    for name, *_range in ranges:
        if name not in parameters:
            raise ValueError(f"Parameter {name} not found in {parameters}.")
        _ranges[parameters.index(name)] = (
            float(_range[0]),
            float(_range[1]),
            int(_range[2]),
        )
    return _ranges


def save_inference_data(mcmc: numpyro.infer.MCMC) -> None:
    os.makedirs("numpyro_sampler_data", exist_ok=True)

    inference_data = az.from_numpyro(mcmc)

    header = list(inference_data.posterior.data_vars.keys())

    posterior_samples = mcmc.get_samples()
    np.savetxt(
        "numpyro_sampler_data/samples.dat",
        np.column_stack([posterior_samples[key] for key in header]),
        header=" ".join(header),
    )

    summary = {k: list(v) for k, v in inference_data.to_dict().items()}
    write_json("posterior_summary.json", summary)

    posterior_data = np.permute_dims(
        np.asarray(inference_data.posterior.to_dataarray()),
        (1, 2, 0),  # (variable, chain, draw) -> (chain, draw, variable)
    )

    n_chains = posterior_data.shape[0]

    for i in range(n_chains):
        np.savetxt(
            f"numpyro_sampler_data/chain_{i}.dat",
            posterior_data[i],
            header=" ".join(header),
            comments="#",
            delimiter=" ",
        )
