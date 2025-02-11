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
from typing import Any, Dict, List, Literal, Union

from numpyro.distributions.distribution import DistributionLike

from gwkokab.models.utils import JointDistribution
from gwkokab.utils.tools import error_if

from .common import get_dist


PoissonMeanConfig = Dict[
    str,
    Union[
        int,  # sample size
        float,  # scale
        Dict[str, Union[Literal["self"], List[Dict[str, Any]]]],  # proposal_dists
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


def poisson_mean_parser(
    filepath: str,
) -> List[Union[Literal["self"], DistributionLike]]:
    """Parse the JSON file containing the configuration of the Poisson mean.

    Parameters
    ----------
    filepath : str
        The path to the JSON file containing the configuration of the Poisson mean.

    Returns
    -------
    List[Union[Literal[ &quot;self&quot; ], DistributionLike]]
        The configuration of the Poisson mean.
    """
    with open(filepath, "r") as file:
        pmean_json: PoissonMeanConfig = json.load(file)
    proposal_distribution = []
    proposal_distribution_dict = pmean_json.pop("proposal_dists")
    proposal_distribution = [
        parse_distribution(dist) for dist in proposal_distribution_dict
    ]
    pmean_json["proposal_dists"] = proposal_distribution
    return pmean_json
