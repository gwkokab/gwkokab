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

from typing_extensions import Optional

from jax import numpy as jnp
from jax.random import truncated_normal
from jaxtyping import Array
from numpyro.distributions import constraints

from ..utils.misc import get_key
from .abstracteccentricitymodel import AbstractEccentricityModel


class EccentricityModel(AbstractEccentricityModel):
    arg_constraints = {"sigma_ecc": constraints.positive}

    def __init__(self, sigma_ecc: float, *, valid_args=None) -> None:
        self.sigma_ecc = sigma_ecc
        super(EccentricityModel, self).__init__(batch_shape=jnp.shape(sigma_ecc), validate_args=valid_args)

    def sample(self, key: Optional[Array | int], sample_shape: tuple = ()) -> Array:
        if key is None or isinstance(key, int):
            key = get_key(key)
        return (
            truncated_normal(
                key=key,
                lower=0.0,
                upper=1.0,
                shape=sample_shape + self.batch_shape,
            )
            * self.sigma_ecc
        )

    def __repr__(self) -> str:
        string = f"EccentricityModel(sigma_ecc={self._scale})"
        return string
