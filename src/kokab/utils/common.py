# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from typing import Dict, List, Tuple, Union

import jax
import numpy as np
import pandas as pd
from numpyro import distributions as dist
from numpyro._typing import DistributionT

from gwkokab.utils.tools import error_if
from kokab.utils.regex import match_all


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


def _available_prior(name: str) -> DistributionT:
    """Get the available prior from numpyro distributions.

    Parameters
    ----------
    name : str
        name of the prior

    Returns
    -------
    DistributionT
        prior distribution

    Raises
    ------
    AttributeError
        If the prior is not found in numpyro distributions
    """
    error_if(
        name not in dist.__all__,
        AttributeError,
        f"Prior {name} not found. Available priors are: " + ", ".join(dist.__all__),
    )
    return getattr(dist, name)


def _is_lazy_prior(prior_dict: dict[str, Union[str, float]]) -> bool:
    """Check if the prior is a lazy prior.

    Parameters
    ----------
    prior_dict : dict[str, Union[str, float]]
        dictionary of prior

    Returns
    -------
    bool
        True if the prior is a lazy prior, False otherwise
    """
    for value in prior_dict.values():
        if isinstance(value, str):
            return True
    return False


def get_processed_priors(params: List[str], priors: dict) -> dict:
    """Get the processed priors from a list of parameters. A processed prior is either
    an instantiated prior or a tuple of :code:`(jax.tree_util.Partial, lazy_vars)` where
    :code:`lazy_vars` is a dictionary of lazy variables.

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
    for param in params:
        error_if(param not in matched_prior_params, msg=f"Missing prior for {param}")

    for key, value in matched_prior_params.items():
        # if the value is not a dict, means its a constant/duplicate value
        if not isinstance(value, dict):
            continue
        value_cpy = value.copy()
        dist_type = value_cpy.pop("dist")
        prior = _available_prior(dist_type)

        # if there are no lazy variables, instantiate the prior
        if not _is_lazy_prior(value_cpy):
            matched_prior_params[key] = prior(**value_cpy, validate_args=True)
            continue

        # separate the lazy variables and the non-lazy variables
        lazy_vars = {k: v for k, v in value_cpy.items() if isinstance(v, str)}
        non_lazy_vars = {k: v for k, v in value_cpy.items() if not isinstance(v, str)}
        matched_prior_params[key] = (
            jax.tree_util.Partial(prior, **non_lazy_vars, validate_args=True),
            lazy_vars,
        )
    return matched_prior_params


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
