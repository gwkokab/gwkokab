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


from functools import partial
from typing import Any, Callable, Dict, Tuple

from jax import lax, random as jrd
from jax.tree_util import register_pytree_node_class
from numpyro.distributions.distribution import Distribution


@register_pytree_node_class
class Bake(object):
    def __init__(self, dist: Distribution | Callable[..., Distribution]) -> None:
        """It is designed to be a simple and flexible way to define a distribution
        for the inference. It has a similar interface to the
        :class:`~numpyro.distributions.distribution.Distribution` class, but it
        allows for the parameters of the distribution to be fixed or variable.

        Parameters
        ----------
        dist : Distribution | Callable[..., Distribution]
            A distribution or a function that returns a distribution
        """
        self._dist = dist

    def __call__(self, **kwargs: Any) -> "Bake":
        """Set the parameters of the distribution.

        Returns
        -------
        Self
            The Bake object

        Raises
        ------
        ValueError
            If the type of a parameter is invalid
        """
        constants: Dict[str, int | float | None] = dict()
        variables: Dict[str, Distribution] = dict()
        duplicates: Dict[str, str] = dict()
        for key, value in kwargs.items():
            if value is None:
                constants[key] = None
            elif isinstance(value, Distribution):
                variables[key] = value
            elif isinstance(value, (int, float)):
                constants[key] = lax.stop_gradient(value)
            elif isinstance(value, str):
                continue
            else:
                raise ValueError(
                    f"Parameter {key} has an invalid type {type(value)}: {value}"
                )
        for key, value in kwargs.items():
            if isinstance(value, str):
                if value in constants:
                    constants[key] = constants[value]
                elif value in variables:
                    duplicates[key] = value
        self.constants = constants
        self.variables = variables
        self.duplicates = duplicates
        return self

    def get_dist(
        self,
    ) -> Tuple[Dict[str, Distribution], Dict[str, str], Callable[..., Distribution]]:
        """Return the distribution with the fixed parameters set.

        Returns
        -------
        Tuple[Dict[str, Distribution], Dict[str, str], Callable[..., Distribution]]
            A tuple containing the distribution with the fixed parameters set
            and a function that returns the distribution with the fixed parameters
            set.
        """
        return self.variables, self.duplicates, partial(self._dist, **self.constants)

    def get_dummy(self) -> Distribution:
        """Return a dummy distribution for debug and testing purposes.

        Returns
        -------
        Distribution
            A dummy distribution
        """
        key = jrd.PRNGKey(0)
        variables = {
            name: prior.sample(key=key, sample_shape=())
            for name, prior in self.variables.items()
        }
        duplicates = {name: variables[value] for name, value in self.duplicates.items()}
        return self._dist(**self.constants, **variables, **duplicates)

    def tree_flatten(self):
        return (self._dist, self.variables, self.constants, self.duplicates), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        obj = cls.__new__(cls)
        setattr(obj, "dist", children[0])
        setattr(obj, "variables", children[1])
        setattr(obj, "constants", children[2])
        setattr(obj, "duplicates", children[3])
        Bake.__init__(obj)
        return obj
