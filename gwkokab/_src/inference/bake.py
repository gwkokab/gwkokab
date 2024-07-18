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
    fmtstring = dist.__class__.__name__
    fmtstring += "("
    fmtstring += ", ".join(
        [f"{v}={dist.__getattribute__(v)}" for v in dist.arg_constraints.keys()]
    )
    fmtstring += ")"
    return fmtstring


@register_pytree_node_class
class Bake(object):
    def __init__(self, dist: Distribution | Callable[[], Distribution]) -> None:
        self.dist = dist

    def __call__(self, **kwargs: Any) -> Self:
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
        return self.variables, partial(self.dist, **self.constants)

    def get_dummy(self) -> Distribution:
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
        return (self.variables, self.constants), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
