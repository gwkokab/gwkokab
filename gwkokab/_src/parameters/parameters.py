#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from functools import partial
from typing_extensions import Optional

from numpyro import distributions as dist

from ..priors.priors import UnnormalizedUniformOnRealLine


class Parameter(object):
    """Initializes a Parameter object.

    :param name: Name of the parameter.
    :param label: Label of the parameter, defaults to None
    :param prior: Distribution object representing the prior distribution of
        the parameter, defaults to None
    :param default_prior: Default prior distribution of the parameter, defaults
        to None
    :raises ValueError: If prior distribution is not provided
    """

    def __init__(
        self,
        *,
        name: str,
        prior: Optional[dist.Distribution] = None,
        default_prior: Optional[dist.Distribution] = None,
    ) -> None:
        if prior is None and default_prior is None:
            raise ValueError("Prior distribution must be provided.")
        if prior is None:
            prior = default_prior
        assert isinstance(
            prior, dist.Distribution
        ), "Prior must be a numpyro.distributions.Distribution object."
        self._prior = prior
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def prior(self) -> dist.Distribution:
        return self._prior

    def __repr__(self) -> str:
        string = (
            f"Parameter(name={self.name}, label={self.label}, "
            f"prior={self.prior.__class__.__name__}("
            + ", ".join(
                [
                    f"{v}={self.prior.__getattribute__(v)}"
                    for v in self.prior.arg_constraints.keys()
                ]
            )
            + "))"
        )
        return string

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameter):
            return False
        return self.name == other.name and self.prior == other.prior


def ncopy(
    n: int, params: list[str], priors: dict[dist.Distribution]
) -> dict[str, Parameter]:
    """Creates n copies of the parameters.

    :param n: Number of copies to create.
    :param params: List of parameters to create copies of.
    :param priors: Dictionary of priors for the parameters.
    :return: Dictionary of n copies of the parameters.
    """
    all_params = {}
    for param in params:
        param_arr = {
            f"{param}_{i}": Parameter(
                name=f"{param}_{i}",
                label="$" + param + "_{" + str(i) + "}$",
                prior=priors.get(f"{param}_{i}", priors.get(param)),
            )
            for i in range(n)
        }
        all_params.update(param_arr)
    return all_params


# TODO: Add more parameters as needed.


PRIMARY_MASS_SOURCE = partial(
    Parameter,
    name="mass_1_source",
    default_prior=UnnormalizedUniformOnRealLine(),
)
SECONDARY_MASS_SOURCE = partial(
    Parameter,
    name="mass_2_source",
    default_prior=UnnormalizedUniformOnRealLine(),
)
PRIMARY_MASS_DETECTED = partial(
    Parameter,
    name="mass_1",
    default_prior=UnnormalizedUniformOnRealLine(),
)
SECONDARY_MASS_DETECTED = partial(
    Parameter,
    name="mass_2",
    default_prior=UnnormalizedUniformOnRealLine(),
)
MASS_RATIO = partial(
    Parameter,
    name="mass_ratio",
    default_prior=UnnormalizedUniformOnRealLine(),
)
CHIRP_MASS = partial(
    Parameter,
    name="chirp_mass",
    default_prior=UnnormalizedUniformOnRealLine(),
)
SYMMETRIC_MASS_RATIO = partial(
    Parameter,
    name="symmetric_mass_ratio",
    default_prior=UnnormalizedUniformOnRealLine(),
)
REDUCED_MASS = partial(
    Parameter,
    name="reduced_mass",
    default_prior=UnnormalizedUniformOnRealLine(),
)
ECCENTRICITY = partial(
    Parameter,
    name="ecc",
    default_prior=UnnormalizedUniformOnRealLine(),
)
PRIMARY_SPIN_MAGNITUDE = partial(
    Parameter,
    name="a1",
    default_prior=UnnormalizedUniformOnRealLine(),
)
SECONDARY_SPIN_MAGNITUDE = partial(
    Parameter,
    name="a2",
    default_prior=UnnormalizedUniformOnRealLine(),
)
PRIMARY_SPIN_X = partial(
    Parameter,
    name="spin_1x",
    default_prior=UnnormalizedUniformOnRealLine(),
)
PRIMARY_SPIN_Y = partial(
    Parameter,
    name="spin_1y",
    default_prior=UnnormalizedUniformOnRealLine(),
)
PRIMARY_SPIN_Z = partial(
    Parameter,
    name="spin_1z",
    default_prior=UnnormalizedUniformOnRealLine(),
)
SECONDARY_SPIN_X = partial(
    Parameter,
    name="spin_2x",
    default_prior=UnnormalizedUniformOnRealLine(),
)
SECONDARY_SPIN_Y = partial(
    Parameter,
    name="spin_2y",
    default_prior=UnnormalizedUniformOnRealLine(),
)
SECONDARY_SPIN_Z = partial(
    Parameter,
    name="spin_2z",
    default_prior=UnnormalizedUniformOnRealLine(),
)
TILE_1 = partial(
    Parameter,
    name="tile_1",
    default_prior=UnnormalizedUniformOnRealLine(),
)
TILE_2 = partial(
    Parameter,
    name="tile_2",
    default_prior=UnnormalizedUniformOnRealLine(),
)
EFFECTIVE_SPIN_MAGNITUDE = partial(
    Parameter,
    name="chi_eff",
    default_prior=UnnormalizedUniformOnRealLine(),
)
COS_TILE_1 = partial(
    Parameter,
    name="tile_1",
    default_prior=UnnormalizedUniformOnRealLine(),
)
COS_TILE_2 = partial(
    Parameter,
    name="tile_2",
    default_prior=UnnormalizedUniformOnRealLine(),
)
REDSHIFT = partial(
    Parameter,
    name="redshift",
    default_prior=UnnormalizedUniformOnRealLine(),
)
