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
from typing import Any
from typing_extensions import Self, Tuple

from numpyro.distributions import Distribution


def get_numpyro_dist_repr(dist: Distribution) -> str:
    fmtstring = dist.__class__.__name__
    fmtstring += "("
    fmtstring += ", ".join(
        [f"{v}={dist.__getattribute__(v)}" for v in dist.arg_constraints.keys()]
    )
    fmtstring += ")"
    return fmtstring


def classify_params(**kwds: Any) -> Tuple[dict, dict]:
    constants = dict()
    variables = dict()
    for key, value in kwds.items():
        if isinstance(value, Distribution):
            variables[key] = value
        elif isinstance(value, (int, float)):
            constants[key] = value
        else:
            raise ValueError(
                f"Parameter {key} has an invalid type {type(value)}: {value}"
            )
    return constants, variables


class Bake(object):
    def __init__(self, dist: Distribution) -> None:
        self.dist = dist

    def __call__(self, **kwds: Any) -> Self:
        constants, variables = classify_params(**kwds)
        if not variables:
            initialized_dist = self.dist(**constants)
        else:
            initialized_dist = partial(self.dist, **constants)
        self.dist = initialized_dist
        self.constants = constants
        self.variables = variables
        return self

    def __repr__(self) -> str:
        if isinstance(self.dist, partial):
            fmtstring = self.dist.func.__name__
        else:
            fmtstring = self.dist.__class__.__name__
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
