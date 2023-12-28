# Copyright 2023 The Jaxtro Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import jaxampler as jx
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike


class Wysocki2019MassModel(jx.rvs.ContinuousRV):
    """Power law distribution with a lower and upper mass limit

    Wysocki2019MassModel is a subclass of jx.rvs.ContinuousRV and implements
    the power law distribution with a lower and upper mass limit as
    described in https://arxiv.org/abs/1805.06442
    """

    def __init__(self,
                 alpha: ArrayLike,
                 k: ArrayLike,
                 mmin: ArrayLike,
                 mmax: ArrayLike,
                 Mmax: ArrayLike,
                 name: str = None) -> None:
        """__init__ method for Wysocki2019MassModel

        Parameters
        ----------
        alpha : ArrayLike
            Alpha parameter of the power law
        k : ArrayLike
            k parameter of the power law
        mmin : ArrayLike
            minimum mass of the black hole
        mmax : ArrayLike
            maximum mass of the black hole
        Mmax : ArrayLike
            maximum total mass of the binary
        name : str, optional
            name of the object, by default None
        """
        self._alpha = alpha
        self._k = k
        self._mmin = mmin
        self._mmax = mmax
        self._Mmax = Mmax
        self.check_params()
        self._Z = self.Z()
        self._name = name

    def check_params(self) -> None:
        """Check if the parameters are valid"""
        assert jnp.all(self._k % 1 == 0), f"k must be a positive integer, got {self._k}"
        assert jnp.all(0.0 <= self._mmin), f"mmin must be positive, got {self._mmin}"
        assert jnp.all(self._mmin <= self._mmax), f"mmin must be less than mmax, got {self._mmin} and {self._mmax}"
        assert jnp.all(self._mmax <= self._Mmax), f"mmax must be less than Mmax, got {self._mmax} and {self._Mmax}"

    def Z(self) -> float:
        """Normalization constant

        Returns
        -------
        float
            Normalization constant
        """
        q_m = self._mmin / self._mmax
        beta = 1.0 - self._alpha
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
    def logpdf(self, m1: ArrayLike, m2: ArrayLike) -> Array:
        """Log of the probability density function

        Parameters
        ----------
        m1 : ArrayLike
            mass of the primary black hole
        m2 : ArrayLike
            mass of the secondary black hole

        Returns
        -------
        Array
            Log of the probability density function
        """
        logpdf_val = jnp.where(
            self.mask(m1, m2, self._mmin, self._mmax, self._Mmax),
            self._k * jnp.log(m2) - (self._k + self._alpha) * jnp.log(m1) - jnp.log(m1 - self._mmin) - jnp.log(self._Z),
            -jnp.inf,
        )
        return logpdf_val

    @staticmethod
    def mask(m1: ArrayLike, m2: ArrayLike, mmin: ArrayLike, mmax: ArrayLike, Mmax: ArrayLike) -> Array:
        """Conditions for mass to be in the distribution

        Parameters
        ----------
        m1 : ArrayLike
            mass of the primary black hole
        m2 : ArrayLike
            mass of the secondary black hole
        mmin : ArrayLike
            minimum mass of the black hole
        mmax : ArrayLike
            maximum mass of the black hole
        Mmax : ArrayLike
            maximum total mass of the binary

        Returns
        -------
        Array
            Conditions for mass to be in the distribution
        """
        return (mmin <= m2) & (m2 <= m1) & (m1 <= mmax) & (m1 + m2 <= Mmax)

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        """Random variates from the distribution

        Parameters
        ----------
        N : int, optional
            number of random variates, by default 1

        Returns
        -------
        Array
            Random variates from the distribution
        """
        m2 = jx.rvs.Uniform(low=self._mmin, high=self._mmax).rvs(N, key).flatten()
        key = jx.utils.new_prn_key(key)
        m1 = jx.rvs.TruncPowerLaw(alpha=-(self._k + self._alpha), low=m2, high=self._mmax).rvs(1, key).flatten()
        return jnp.column_stack([m1, m2])

    def __repr__(self) -> str:
        """string representation of the object

        Returns
        -------
        str
            string representation of the object
        """
        string = f"Wysocki2019MassModel(alpha={self._alpha}, k={self._k}, "
        string += f"mmin={self._mmin}, mmax={self._mmax}, Mmax={self._Mmax})"
        if self._name is not None:
            string += f", {self._name}"
        string += ")"
        return string
