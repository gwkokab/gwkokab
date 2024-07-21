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


from __future__ import annotations

from functools import partial
from typing_extensions import Any, Callable, Dict, Self, Tuple

from jax import random as jrd
from jax.tree_util import register_pytree_node_class
from numpyro.distributions import Distribution


def get_numpyro_dist_repr(dist: Distribution) -> str:
    r"""Return a string representation of a numpyro distribution.

    :param dist: A numpyro distribution
    :type dist: Distribution
    :return: A string representation of the distribution
    :rtype: str
    """
    fmtstring = dist.__class__.__name__
    fmtstring += "("
    fmtstring += ", ".join(
        [f"{v}={dist.__getattribute__(v)}" for v in dist.arg_constraints.keys()]
    )
    fmtstring += ")"
    return fmtstring


@register_pytree_node_class
class Bake(object):
    r"""It is designed to be a simple and flexible way to define a
    distribution for the inference. It has a similar interface to the
    :class:`numpyro.distributions.distribution.Distribution` class, but it allows for
    the parameters of the distribution to be fixed or variable.

    We can define a :class:`numpyro.distributions.truncated.TruncatedNormal`
    distribution with variable `scale` that has uniform prior from :code:`0` to
    :code:`10`, and fixed :code:`loc`, :code:`low`, and :code:`high` parameters as
    follows:

    .. tab-set::

        .. tab-item:: Standard

            .. code-block:: python

                >>> from numpyro.distributions import TruncatedNormal, Uniform
                >>> standard_model = TruncatedNormal(
                ...    loc=0.0,
                ...    scale=0.05,
                ...    low=0.0,
                ...    high=10.0,
                ... )
                >>> print(standard_model)
                <numpyro.distributions.truncated.TwoSidedTruncatedDistribution object at 0x7fabaa6acb50>

        .. tab-item:: Bake

            .. code-block:: python

                >>> from numpyro.distributions import TruncatedNormal, Uniform
                >>> baked_model = Bake(TruncatedNormal)(
                ...    loc=0.0,
                ...    scale=Uniform(0.0, 10.0),
                ...    low=0.0,
                ...    high=10.0,
                ... )
                >>> print(baked_model)
                TruncatedNormal(loc=0.0, low=0.0, high=10.0, scale=Uniform(low=0.0, high=10.0))

    :param dist: A distribution or a function that returns a distribution
    :type dist: Distribution | Callable[[], Distribution]
    """

    def __init__(self, dist: Distribution | Callable[[], Distribution]) -> None:
        self.dist = dist

    def __call__(self, **kwargs: Any) -> Self:
        r"""Set the parameters of the distribution.

        :raises ValueError: If the type of a parameter is invalid
        :return: The Bake object
        :rtype: Self
        """
        constants: Dict[str, int | float] = dict()
        variables: Dict[str, Distribution] = dict()
        for key, value in kwargs.items():
            if isinstance(value, Distribution):
                variables[key] = value
            elif isinstance(value, (int, float)):
                constants[key] = value
            else:
                raise ValueError(
                    f"Parameter {key} has an invalid type {type(value)}: {value}"
                )
        self.constants = constants
        self.variables = variables
        return self

    def get_dist(self) -> Tuple[Dict[str, Distribution], Callable[[], Distribution]]:
        r"""Return the distribution with the fixed parameters set.

        :return: A tuple containing the distribution with the fixed parameters set and a
            function that returns the distribution with the fixed parameters set.
        :rtype: Tuple[Dict[str, Distribution], Callable[[], Distribution]]
        """
        return self.variables, partial(self.dist, **self.constants)

    def get_dummy(self) -> Distribution:
        r"""Return a dummy distribution for debug and testing purposes.

        :return: A dummy distribution
        :rtype: Distribution
        """
        key = jrd.PRNGKey(0)
        return self.dist(
            **self.constants,
            **{
                name: prior.sample(key=key, sample_shape=())
                for name, prior in self.variables.items()
            },
        )

    def __repr__(self) -> str:
        fmtstring = self.dist.__name__
        fmtstring += "("
        if self.constants:
            for key, value in self.constants.items():
                fmtstring += f"{key}={value}, "
        if self.variables:
            for key, value in self.variables.items():
                fmtstring += f"{key}={get_numpyro_dist_repr(value)}, "
        if fmtstring.endswith(", "):
            fmtstring = fmtstring[:-2]
        fmtstring += ")"
        return fmtstring

    def tree_flatten(self):
        return (self.dist, self.variables, self.constants), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        obj = cls.__new__(cls)
        setattr(obj, "dist", children[0])
        setattr(obj, "variables", children[1])
        setattr(obj, "constants", children[2])
        Bake.__init__(obj)
        return obj
