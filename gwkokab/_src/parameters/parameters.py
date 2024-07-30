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


from typing_extensions import Self

from jax.tree_util import register_pytree_node_class
from numpyro.distributions import Distribution, ImproperUniform
from numpyro.distributions.constraints import real


@register_pytree_node_class
class Parameter:
    r"""Initializes a Parameter object. Default prior is an
    :class:`~numpyro.distributions.distribution.ImproperUniform` distribution.

    Lets define chirp mass parameter with :class:`~numpyro.distributions.continuous.Uniform`
    prior from 1 to 10 as follows:

    .. code-block:: python

        >>> from numpyro.distributions import Uniform
        >>> chirp_mass = Parameter(name="chirp_mass")(Uniform(1, 10))

    :param name: Name of the parameter.
    """

    def __init__(self, *, name: str) -> None:
        self._prior = ImproperUniform(
            support=real, batch_shape=(), event_shape=(), validate_args=True
        )
        self._name = name

    def __call__(self, prior: Distribution) -> Self:
        prior._validate_args = True
        self._prior = prior
        return self

    @property
    def name(self) -> str:
        r"""Name of the parameter."""
        return self._name

    @property
    def prior(self) -> Distribution:
        r"""Prior distribution of the parameter."""
        return self._prior

    def __repr__(self) -> str:
        string = (
            f"Parameter(name={self.name}, "
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

    def tree_flatten(self):
        return (), {
            "name": self.name,
            "prior": self.prior,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(**aux_data)


def ncopy(
    n: int, params: list[str], priors: dict[str, Distribution]
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
                prior=priors.get(f"{param}_{i}", priors.get(param)),
            )
            for i in range(n)
        }
        all_params.update(param_arr)
    return all_params


# TODO: Add more parameters as needed.


PRIMARY_MASS_SOURCE = Parameter(name="mass_1_source")
SECONDARY_MASS_SOURCE = Parameter(name="mass_2_source")
PRIMARY_MASS_DETECTED = Parameter(name="mass_1")
SECONDARY_MASS_DETECTED = Parameter(name="mass_2")
MASS_RATIO = Parameter(name="mass_ratio")
CHIRP_MASS = Parameter(name="chirp_mass")
SYMMETRIC_MASS_RATIO = Parameter(name="symmetric_mass_ratio")
REDUCED_MASS = Parameter(name="reduced_mass")
ECCENTRICITY = Parameter(name="ecc")
PRIMARY_SPIN_MAGNITUDE = Parameter(name="a_1")
SECONDARY_SPIN_MAGNITUDE = Parameter(name="a_2")
PRIMARY_SPIN_X = Parameter(name="spin_1x")
PRIMARY_SPIN_Y = Parameter(name="spin_1y")
PRIMARY_SPIN_Z = Parameter(name="spin_1z")
SECONDARY_SPIN_X = Parameter(name="spin_2x")
SECONDARY_SPIN_Y = Parameter(name="spin_2y")
SECONDARY_SPIN_Z = Parameter(name="spin_2z")
TILT_1 = Parameter(name="tile_1")
TILT_2 = Parameter(name="tile_2")
EFFECTIVE_SPIN_MAGNITUDE = Parameter(name="chi_eff")
COS_TILT_1 = Parameter(name="tile_1")
COS_TILT_2 = Parameter(name="tile_2")
REDSHIFT = Parameter(name="redshift")
