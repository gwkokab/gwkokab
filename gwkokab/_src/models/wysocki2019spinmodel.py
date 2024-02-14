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

from typing_extensions import Optional

from jax import lax, numpy as jnp
from jax.random import beta
from jaxtyping import Array
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes

from ..utils.misc import get_key
from .abstractspinmodel import AbstractSpinModel


class Wysocki2019SpinModel(AbstractSpinModel):
    """Beta distribution for the spin magnitude

    Wysocki2019SpinModel is a subclass of ContinuousRV and implements
    the beta distribution for the spin magnitude as described in
    https://arxiv.org/abs/1805.06442
    """

    arg_constraints = {
        "alpha": constraints.positive,
        "beta": constraints.positive,
        "chimax": constraints.interval(0.0, 1.0),
    }

    def __init__(self, alpha: float, beta: float, chimax: float = 1.0, *, error_scale: float, valid_args=None) -> None:
        self.alpha, self.beta, self.chimax, self.error_scale = promote_shapes(alpha, beta, chimax, error_scale)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(chimax),
            jnp.shape(error_scale),
        )
        super(Wysocki2019SpinModel, self).__init__(batch_shape=batch_shape, validate_args=valid_args)

    def sample(self, key: Optional[Array | int], sample_shape: tuple = ()) -> Array:
        if key is None or isinstance(key, int):
            key = get_key(key)
        return beta(
            key=key,
            a=self.alpha,
            b=self.beta,
            shape=sample_shape,
        )

    def __repr__(self) -> str:
        string = f"Wysocki2019SpinModel(alpha_chi={self.alpha}, "
        string += f"beta_chi={self.beta}, chimax={self.chimax})"
        return string
