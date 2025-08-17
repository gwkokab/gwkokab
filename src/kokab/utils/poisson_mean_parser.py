# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from typing import Any, Dict, List, Literal, Union

from jaxtyping import PRNGKeyArray
from numpyro._typing import DistributionLike

from gwkokab.models.utils import JointDistribution
from gwkokab.poisson_mean import PoissonMean
from gwkokab.utils.tools import error_if

from .common import get_dist, vt_json_read_and_process


ProposalDistArgType = Dict[str, Union[Literal["self"], List[Dict[str, Any]]]]
PoissonMeanConfig = Dict[
    str,
    Union[
        int,  # sample size
        float,  # scale
        ProposalDistArgType,  # proposal_dists
    ],
]


def parse_distribution(
    per_component_dist: Union[str, List[Dict[str, Any]]],
) -> Union[Literal["self"], DistributionLike]:
    """Parse the proposal distribution.

    Parameters
    ----------
    per_component_dist : Union[str, List[Dict[str, Any]]]
        The proposal distribution

    Returns
    -------
    Union[Literal[&quot;self&quot;], DistributionLike]
        The proposal distribution

    Raises
    ------
    ValueError
        If the distribution format is invalid.
    """
    if isinstance(per_component_dist, str):
        error_if(
            per_component_dist.strip().lower() != "self",
            msg="The key in the proposal distribution must be 'self'.",
        )
        return "self"
    elif isinstance(per_component_dist, list):
        error_if(
            not all(isinstance(dist_dict, dict) for dist_dict in per_component_dist),
            msg="The proposal distribution must be a list of dictionaries.",
        )
        return JointDistribution(
            *[get_dist(dist_dict) for dist_dict in per_component_dist],
            validate_args=True,
        )
    raise ValueError("Invalid distribution format")


def poisson_mean_parser(filepath: str) -> PoissonMeanConfig:
    """Parse the JSON file containing the configuration of the Poisson mean.

    Parameters
    ----------
    filepath : str
        The path to the JSON file containing the configuration of the Poisson mean.

    Returns
    -------
    PoissonMeanConfig
        The configuration of the Poisson mean.
    """
    with open(filepath, "r") as file:
        pmean_json: PoissonMeanConfig = json.load(file)
    try:
        proposal_distribution_dict: ProposalDistArgType = pmean_json.pop(
            "proposal_dists"
        )
        proposal_distribution = [
            parse_distribution(dist) for dist in proposal_distribution_dict
        ]
        pmean_json["proposal_dists"] = proposal_distribution
    except KeyError:
        pass
    return pmean_json


def read_pmean(
    key: PRNGKeyArray, parameters: List[str], filename: str, selection_fn_filename: str
) -> PoissonMean:
    """Read the Poisson mean from a JSON file and create a PoissonMean instance.

    Parameters
    ----------
    key : PRNGKeyArray
        The random number generator key.
    parameters : List[str]
        List of parameters to be used in the Poisson mean.
    filename : str
        The path to the JSON file containing the configuration of the Poisson mean.
    selection_fn_filename : str
        The path to the JSON file containing the selection function.

    Returns
    -------
    PoissonMean
        An instance of the PoissonMean class initialized with the configuration from the JSON file.
    """
    selection_fn = vt_json_read_and_process(parameters, selection_fn_filename)
    kwargs = poisson_mean_parser(filename)
    erate_estimator = PoissonMean(
        selection_fn,
        key=key,
        **kwargs,  # type: ignore[arg-type]
    )
    return erate_estimator
