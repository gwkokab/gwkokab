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
from collections.abc import Sequence
from typing import List, Tuple

import numpy as np
import numpyro.distributions as dist
import pandas as pd
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import Uniform

from gwkokab.parameters import Parameter
from gwkokab.vts import NeuralVT

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


def log_weights_and_samples(
    key: PRNGKeyArray,
    parameters: Sequence[Parameter],
    vt_filename: str,
    num_samples: int,
) -> Tuple[Array, Array]:
    r"""Get the weights and samples from the VT.

    :param parameters: list of parameters
    :param vt_filename: VT filename
    :param num_samples: number of samples
    :return: tuple of weights and samples
    """
    nvt = NeuralVT([param.name for param in parameters], vt_filename)
    logVT_vmap = nvt.get_vmapped_logVT()
    hyper_uniform = dist.Uniform(
        low=jnp.asarray([param.prior.low for param in parameters]),
        high=jnp.asarray([param.prior.high for param in parameters]),
        validate_args=True,
    )

    uniform_key, proposal_key = jrd.split(key)
    uniform_samples = hyper_uniform.sample(uniform_key, (num_samples,))

    logVT_val = logVT_vmap(uniform_samples)

    mask = logVT_val > hyper_uniform.log_prob(uniform_samples).sum(-1)

    logVT_val = logVT_val[mask]
    uniform_samples = uniform_samples[mask]

    loc_vector_weights = jnn.softmax(logVT_val)
    loc_vector = jnp.average(uniform_samples, axis=0, weights=loc_vector_weights)

    covariance_matrix = jnp.cov(uniform_samples.T, aweights=loc_vector_weights)

    proposal_dist = dist.MultivariateNormal(
        loc=loc_vector, covariance_matrix=covariance_matrix, validate_args=True
    )

    proposal_samples = proposal_dist.sample(proposal_key, (num_samples,))

    mask = parameters[0].prior.support(proposal_samples[..., 0])
    for i in range(1, len(parameters)):
        mask &= parameters[i].prior.support(proposal_samples[..., i])

    proposal_samples = proposal_samples[mask]

    log_weights = logVT_vmap(proposal_samples) - proposal_dist.log_prob(
        proposal_samples
    )
    return log_weights, proposal_samples
