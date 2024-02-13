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

from typing import Optional

from jax import numpy as jnp
from jax.random import uniform
from jaxtyping import Array

from ..utils.misc import get_key
from .abstractmassmodel import AbstractMassModel


class Wysocki2019MassModel(AbstractMassModel):
    """Power law distribution with a lower and upper mass limit

    Wysocki2019MassModel is a subclass of jx.rvs.ContinuousRV and implements
    the power law distribution with a lower and upper mass limit as
    described in https://arxiv.org/abs/1805.06442
    """

    def __init__(
        self,
        alpha_m: float,
        k: int,
        mmin: float,
        mmax: float,
        Mmax: float,
        name: Optional[str] = None,
    ) -> None:
        self._alpha_m = alpha_m
        self._k = k
        self._mmin = mmin
        self._mmax = mmax
        self._Mmax = Mmax
        self.check_params()

    def check_params(self) -> None:
        assert jnp.all(self._k % 1 == 0), f"k must be a positive integer, got {self._k}"
        assert jnp.all(0.0 <= self._mmin), f"mmin must be positive, got {self._mmin}"
        assert jnp.all(self._mmin <= self._mmax), f"mmin must be less than mmax, got {self._mmin} and {self._mmax}"
        assert jnp.all(self._mmax <= self._Mmax), f"mmax must be less than Mmax, got {self._mmax} and {self._Mmax}"

    def samples(self, num_of_samples: int) -> Array:
        m2 = uniform(key=get_key(), minval=self._mmin, maxval=self._mmax, shape=(num_of_samples,))
        U = uniform(key=get_key(), minval=0.0, maxval=1.0, shape=(num_of_samples,))
        beta = 1 - (self._k + self._alpha_m)
        conditions = [beta == 0.0, beta != 0.0]
        choices = [
            jnp.exp(U * jnp.log(self._mmax) + (1.0 - U) * jnp.log(m2)),
            jnp.exp(jnp.power(beta, -1) * jnp.log(U * jnp.power(self._mmax, beta) + (1.0 - U) * jnp.power(m2, beta))),
        ]
        m1 = jnp.select(conditions, choices)
        return jnp.stack([m1, m2], axis=1)

    def __repr__(self) -> str:
        string = f"Wysocki2019MassModel(alpha_m={self._alpha_m}, k={self._k}, "
        string += f"mmin={self._mmin}, mmax={self._mmax}, Mmax={self._Mmax}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
