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
from jax.random import uniform
from jaxtyping import Array
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes

from ..utils.misc import get_key


class Wysocki2019MassModel(Distribution):
    """Power law distribution with a lower and upper mass limit

    Wysocki2019MassModel is a subclass of jx.rvs.ContinuousRV and implements
    the power law distribution with a lower and upper mass limit as
    described in https://arxiv.org/abs/1805.06442
    """

    arg_constraints = {
        "alpha_m": constraints.real,
        "k": constraints.positive_integer,
        "mmin": constraints.positive,
        "mmax": constraints.positive,
        "Mmax": constraints.positive,
    }

    def __init__(
        self,
        alpha_m: float,
        k: int,
        mmin: float,
        mmax: float,
        Mmax: float,
        *,
        valid_args=None,
    ) -> None:
        self.alpha_m, self.k, self.mmin, self.mmax, self.Mmax = promote_shapes(
            alpha_m,
            k,
            mmin,
            mmax,
            Mmax,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha_m),
            jnp.shape(k),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(Mmax),
        )
        super(Wysocki2019MassModel, self).__init__(batch_shape=batch_shape, validate_args=valid_args)

    def sample(self, key: Optional[Array | int], sample_shape: tuple = ()) -> Array:
        if key is None or isinstance(key, int):
            key = get_key(key)
        m2 = uniform(key=key, minval=self.mmin, maxval=self.mmax, shape=sample_shape + self.batch_shape)
        U = uniform(key=get_key(key), minval=0.0, maxval=1.0, shape=sample_shape + self.batch_shape)
        beta = 1 - (self.k + self.alpha_m)
        conditions = [beta == 0.0, beta != 0.0]
        choices = [
            jnp.exp(U * jnp.log(self.mmax) + (1.0 - U) * jnp.log(m2)),
            jnp.exp(jnp.power(beta, -1.0) * jnp.log(U * jnp.power(self.mmax, beta) + (1.0 - U) * jnp.power(m2, beta))),
        ]
        m1 = jnp.select(conditions, choices)
        return jnp.stack([m1, m2], axis=1)

    def __repr__(self) -> str:
        string = f"Wysocki2019MassModel(alpha_m={self.alpha_m}, k={self.k}, "
        string += f"mmin={self.mmin}, mmax={self.mmax}, Mmax={self.Mmax})"
        return string
