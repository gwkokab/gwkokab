#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

from jax import vmap
from jax.random import truncated_normal
from jaxtyping import Array
from numpyro.distributions import Distribution

from ..utils import get_key


class AbstractEccentricityModel(Distribution):
    def add_error(self, x: Array, scale: float = 0.5, size: int = 10) -> Array:
        return vmap(
            lambda x_: truncated_normal(
                key=get_key(),
                lower=0.0,
                upper=0.5,
                shape=(size,),
                dtype=x.dtype,
            )
            * scale
            + x_
        )(x)
