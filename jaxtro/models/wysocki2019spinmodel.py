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
from jax.scipy.special import betaln
from jax.typing import ArrayLike


class Wysocki2019SpinModel(jx.rvs.ContinuousRV):
    """Beta distribution for the spin magnitude

    Wysocki2019SpinModel is a subclass of ContinuousRV and implements
    the beta distribution for the spin magnitude as described in
    https://arxiv.org/abs/1805.06442
    """

    def __init__(
        self,
        alpha_1: ArrayLike,
        beta_1: ArrayLike,
        alpha_2: ArrayLike,
        beta_2: ArrayLike,
        chimax: ArrayLike,
        name: str = None,
    ) -> None:
        """__init__ method for Wysocki2019SpinModel

        Parameters
        ----------
        alpha : ArrayLike
            Shape parameter of the beta distribution
        beta : ArrayLike
            Shape parameter of the beta distribution
        chimax : ArrayLike
            Maximum spin magnitude
        name : str, optional
            Name of the object, by default None
        """
        self._alpha, self._beta = jx.utils.jx_cast([alpha_1, alpha_2], [beta_1, beta_2])
        self._chimax = chimax
        self.check_params()
        self._name = name

    def check_params(self) -> None:
        """Check if the parameters are valid"""
        assert jnp.all(self._alpha > 0), "alpha must be greater than 0"
        assert jnp.all(self._beta > 0), "beta must be greater than 0"
        assert jnp.all(self._chimax > 0), "chimax must be greater than 0"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, chi: ArrayLike) -> Array:
        """Log of the probability density function

        Parameters
        ----------
        chi : ArrayLike
            Spin magnitude

        Returns
        -------
        Array
            Log of the probability density function
        """
        return jnp.where(
            (chi >= 0) & (chi <= self._chimax),
            (self._alpha - 1) * jnp.log(chi) + (self._beta - 1) * jnp.log(self._chimax - chi) +
            (1 - self._alpha - self._beta) * jnp.log(self._chimax) - betaln(self._alpha, self._beta),
            -jnp.inf,
        )

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
        return jx.rvs.Beta(alpha=self._alpha, beta=self._beta).rvs(N=N, key=key) * self._chimax

    def __repr__(self) -> str:
        """string representation of the object

        Returns
        -------
        str
            string representation of the object
        """
        string = f"Wysocki2019SpinModel(alpha={self._alpha}, "
        string += f"beta={self._beta}, chimax={self._chimax}"
        if self._name is not None:
            string += f", {self._name}"
        string += ")"
        return string
