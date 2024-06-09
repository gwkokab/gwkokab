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


from typing_extensions import Optional

from numpyro import distributions as dist


class Parameter(object):
    """Initializes a Parameter object.

    :param name: Name of the parameter.
    :param label: Label of the parameter, defaults to None
    :param prior: Distribution object representing the prior distribution of the parameter, defaults to None
    :raises ValueError: If prior distribution is not provided
    """

    def __init__(
        self,
        name: str,
        label: Optional[str] = None,
        prior: Optional[dist.Distribution] = None,
    ) -> None:
        if prior is None:
            raise ValueError("Prior distribution must be provided.")
        self._prior = prior
        self._name = name
        if label is None:
            self._label = name
        else:
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
        return f"Parameter(name={self.name})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Parameter):
            return False
        return self.name == value.name and self.label == value.label


# TODO: default priors are placeholders at the moment. Jeffery priors should be used.

uniform_01 = dist.Uniform()
uniform_neg1_1 = dist.Uniform(-1.0, 1.0)

PRIMARY_MASS_SOURCE = lambda prior=None: Parameter(
    name="m1_source", label=r"$m_1^\text{source}$", prior=dist.Uniform(50.0, 100.0) if prior is None else prior
)
SECONDARY_MASS_SOURCE = lambda prior=None: Parameter(
    name="m2_source", label=r"$m_2^\text{source}$", prior=dist.Uniform(5.0, 30.0) if prior is None else prior
)
MASS_RATIO = lambda prior=None: Parameter(name="q", label=r"$q$", prior=uniform_01 if prior is None else prior)
CHIRP_MASS = lambda prior=None: Parameter(
    name="M_c", label=r"$M_c$", prior=dist.Uniform(5.0, 50.0) if prior is None else prior
)
SYMMETRIC_MASS_RATIO = lambda prior=None: Parameter(
    name="eta", label=r"$\eta$", prior=dist.Uniform(5.0, 50.0) if prior is None else prior
)
REDUCED_MASS = lambda prior=None: Parameter(
    name="M_r", label=r"$M_r$", prior=dist.Uniform(5.0, 50.0) if prior is None else prior
)
ECCENTRICITY = lambda prior=None: Parameter(
    name="ecc", label=r"$\varepsilon$", prior=uniform_01 if prior is None else prior
)
PRIMARY_ALIGNED_SPIN = lambda prior=None: Parameter(
    name="a1", label=r"$a_1$", prior=uniform_01 if prior is None else prior
)
SECONDARY_ALIGNED_SPIN = lambda prior=None: Parameter(
    name="a2", label=r"$a_2$", prior=uniform_01 if prior is None else prior
)
PRIMARY_SPIN_X = lambda prior=None: Parameter(
    name="a1x", label=r"$a_1^x$", prior=uniform_neg1_1 if prior is None else prior
)
PRIMARY_SPIN_Y = lambda prior=None: Parameter(
    name="a1y", label=r"$a_1^y$", prior=uniform_neg1_1 if prior is None else prior
)
PRIMARY_SPIN_Z = lambda prior=None: Parameter(
    name="a1z", label=r"$a_1^z$", prior=uniform_neg1_1 if prior is None else prior
)
SECONDARY_SPIN_X = lambda prior=None: Parameter(
    name="a2x", label=r"$a_2^x$", prior=uniform_neg1_1 if prior is None else prior
)
SECONDARY_SPIN_Y = lambda prior=None: Parameter(
    name="a2y", label=r"$a_2^y$", prior=uniform_neg1_1 if prior is None else prior
)
SECONDARY_SPIN_Z = lambda prior=None: Parameter(
    name="a2z", label=r"$a_2^z$", prior=uniform_neg1_1 if prior is None else prior
)
