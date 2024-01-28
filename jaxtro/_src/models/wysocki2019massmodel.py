#  Copyright 2023 The Jaxtro Authors
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
from typing import Optional

from jax import jit, numpy as jnp
from jaxampler.rvs import RandomVariable, TruncPowerLaw, Uniform
from jaxampler.typing import Numeric
from jaxampler.utils import jxam_array_cast


class Wysocki2019MassModel(RandomVariable):
    """Power law distribution with a lower and upper mass limit

    Wysocki2019MassModel is a subclass of jx.rvs.ContinuousRV and implements
    the power law distribution with a lower and upper mass limit as
    described in https://arxiv.org/abs/1805.06442
    """

    def __init__(
        self,
        alpha_m: Numeric,
        k: Numeric,
        mmin: Numeric,
        mmax: Numeric,
        Mmax: Numeric,
        name: Optional[str] = None,
    ) -> None:
        (
            shape,
            self._alpha_m,
            self._k,
            self._mmin,
            self._mmax,
            self._Mmax,
        ) = jxam_array_cast(alpha_m, k, mmin, mmax, Mmax)
        self.check_params()
        self._Z = self.Z()
        super().__init__(shape=shape, name=name)

    def check_params(self) -> None:
        assert jnp.all(self._k % 1 == 0), f"k must be a positive integer, got {self._k}"
        assert jnp.all(0.0 <= self._mmin), f"mmin must be positive, got {self._mmin}"
        assert jnp.all(self._mmin <= self._mmax), f"mmin must be less than mmax, got {self._mmin} and {self._mmax}"
        assert jnp.all(self._mmax <= self._Mmax), f"mmax must be less than Mmax, got {self._mmax} and {self._Mmax}"

    def Z(self) -> Numeric:
        q_m = self._mmin / self._mmax
        beta = 1.0 - self._alpha_m
        factor = jnp.power(self._mmin, beta) / (self._k + 1)
        _Z = 0
        frac = lambda ii: (jnp.power(q_m, ii - beta) - 1) / (beta - ii)
        for i in range(0, self._k + 1):
            if i == beta:
                _Z -= jnp.log(q_m)
            else:
                _Z += frac(i)
        _Z *= factor
        return _Z

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, m1: Numeric, m2: Numeric) -> Numeric:
        logpdf_val = jnp.where(
            self.mask(m1, m2, self._mmin, self._mmax, self._Mmax),
            self._k * jnp.log(m2)
            - (self._k + self._alpha_m) * jnp.log(m1)
            - jnp.log(m1 - self._mmin)
            - jnp.log(self._Z),
            -jnp.inf,
        )
        return logpdf_val

    @staticmethod
    @jit
    def mask(m1: Numeric, m2: Numeric, mmin: Numeric, mmax: Numeric, Mmax: Numeric) -> Numeric:
        return (mmin <= m2) & (m2 <= m1) & (m1 <= mmax) & (m1 + m2 <= Mmax)

    def samples(self, num_of_samples: int) -> Numeric:
        m2 = Uniform(low=self._mmin, high=self._mmax).rvs(shape=(num_of_samples,))
        m1 = TruncPowerLaw(alpha=-(self._k + self._alpha_m), low=m2, high=self._mmax).rvs(shape=(1,)).flatten()
        return jnp.stack([m1, m2], axis=1)

    def __repr__(self) -> str:
        string = f"Wysocki2019MassModel(alpha_m={self._alpha_m}, k={self._k}, "
        string += f"mmin={self._mmin}, mmax={self._mmax}, Mmax={self._Mmax})"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
