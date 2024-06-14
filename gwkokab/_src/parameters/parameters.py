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
        label: Optional[str] = None,
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
        if label is None:
            label = name
        self._label = label

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._label

    @property
    def prior(self) -> dist.Distribution:
        return self._prior

    def __repr__(self) -> str:
        return f"Parameter(name={self.name}, label={self.label})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Parameter):
            return False
        return self.name == value.name and self.label == value.label


# TODO: Add more parameters as needed.

uniform_01 = dist.Uniform(validate_args=True)
uniform_neg1_1 = dist.Uniform(-1.0, 1.0, validate_args=True)

PRIMARY_MASS_SOURCE = partial(
    Parameter,
    name="mass_1_source",
    label=r"$m_1^\text{source}$",
    default_prior=dist.Uniform(0.5, 200.0, validate_args=True),
)
SECONDARY_MASS_SOURCE = partial(
    Parameter,
    name="mass_2_source",
    label=r"$m_2^\text{source}$",
    default_prior=dist.Uniform(0.5, 200.0, validate_args=True),
)
PRIMARY_MASS_DETECTED = partial(
    Parameter,
    name="mass_1",
    label=r"$m_1$",
    default_prior=dist.Uniform(0.5, 200.0, validate_args=True),
)
SECONDARY_MASS_DETECTED = partial(
    Parameter,
    name="mass_2",
    label=r"$m_2$",
    default_prior=dist.Uniform(0.5, 200.0, validate_args=True),
)
MASS_RATIO = partial(
    Parameter, name="mass_ratio", label=r"$q$", default_prior=uniform_01
)
CHIRP_MASS = partial(
    Parameter,
    name="chirp_mass",
    label=r"$M_c$",
    default_prior=dist.Uniform(5.0, 50.0, validate_args=True),
)
SYMMETRIC_MASS_RATIO = partial(
    Parameter,
    name="symmetric_mass_ratio",
    label=r"$\eta$",
    default_prior=dist.Uniform(5.0, 50.0, validate_args=True),
)
REDUCED_MASS = partial(
    Parameter,
    name="reduced_mass",
    label=r"$M_r$",
    default_prior=dist.Uniform(5.0, 50.0, validate_args=True),
)
ECCENTRICITY = partial(
    Parameter, name="ecc", label=r"$\varepsilon$", default_prior=uniform_01
)
PRIMARY_SPIN_MAGNITUDE = partial(
    Parameter, name="a1", label=r"$a_1$", default_prior=uniform_01
)
SECONDARY_SPIN_MAGNITUDE = partial(
    Parameter, name="a2", label=r"$a_2$", default_prior=uniform_01
)
PRIMARY_SPIN_X = partial(
    Parameter, name="spin_1x", label=r"$a_1^x$", default_prior=uniform_neg1_1
)
PRIMARY_SPIN_Y = partial(
    Parameter, name="spin_1y", label=r"$a_1^y$", default_prior=uniform_neg1_1
)
PRIMARY_SPIN_Z = partial(
    Parameter, name="spin_1z", label=r"$a_1^z$", default_prior=uniform_neg1_1
)
SECONDARY_SPIN_X = partial(
    Parameter, name="spin_2x", label=r"$a_2^x$", default_prior=uniform_neg1_1
)
SECONDARY_SPIN_Y = partial(
    Parameter, name="spin_2y", label=r"$a_2^y$", default_prior=uniform_neg1_1
)
SECONDARY_SPIN_Z = partial(
    Parameter, name="spin_2z", label=r"$a_2^z$", default_prior=uniform_neg1_1
)
TILE_1 = partial(
    Parameter, name="tile_1", label=r"$\theta_1$", default_prior=uniform_neg1_1
)  # TODO: priors are incorrect
TILE_2 = partial(
    Parameter, name="tile_2", label=r"$\theta_2$", default_prior=uniform_neg1_1
)  # TODO: priors are incorrect
EFFECTIVE_SPIN_MAGNITUDE = partial(
    Parameter,
    name="chi_eff",
    label=r"$\chi_\text{eff}$",
    default_prior=uniform_neg1_1,
)
COS_TILE_1 = partial(
    Parameter,
    name="tile_1",
    label=r"$\cos(\theta_1)$",
    default_prior=uniform_neg1_1,
)
COS_TILE_2 = partial(
    Parameter,
    name="tile_2",
    label=r"$\cos(\theta_2)$",
    default_prior=uniform_neg1_1,
)
REDSHIFT = partial(
    Parameter,
    name="redshift",
    label=r"$z$",
    default_prior=dist.Uniform(0.0, 2.0, validate_args=True),
)
