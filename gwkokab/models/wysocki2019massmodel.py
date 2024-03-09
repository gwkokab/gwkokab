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
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils.misc import get_key


class Wysocki2019MassModel(Distribution):
    r"""It is a double side truncated power law distribution, as
    described in equation 7 of the `paper <https://arxiv.org/abs/1805.06442>`__.

    .. math::
        p(m_1,m_2\mid\alpha,k,m_{\text{min}},m_{\text{max}},M_{\text{max}})\propto\frac{m_1^{-\alpha-k}m_2^k}{m_1-m_{\text{min}}}
    """

    arg_constraints = {
        "alpha_m": constraints.real,
        "k": constraints.nonnegative_integer,
        "mmin": constraints.positive,
        "mmax": constraints.positive,
    }

    support = constraints.real_vector
    reparametrized_params = ["m1", "m2"]

    def __init__(self, alpha_m: float, k: int, mmin: float, mmax: float, *, valid_args=None) -> None:
        r"""Initialize the power law distribution with a lower and upper mass limit.

        :param alpha_m: index of the power law distribution
        :param k: mass ratio power law index
        :param mmin: lower mass limit
        :param mmax: upper mass limit
        :param valid_args: If `True`, validate the input arguments.
        """
        self.alpha_m, self.k, self.mmin, self.mmax = promote_shapes(alpha_m, k, mmin, mmax)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha_m),
            jnp.shape(k),
            jnp.shape(mmin),
            jnp.shape(mmax),
        )
        super(Wysocki2019MassModel, self).__init__(
            batch_shape=batch_shape,
            validate_args=valid_args,
            event_shape=(2,),
        )
        q = self.mmin / self.mmax
        K = jnp.arange(self.k + 1)
        Z = jnp.asarray([jnp.power(q, i + self.alpha_m - 1.0) for i in K])
        d = 1 - self.alpha_m - K

        Z = jnp.where(
            (d == 0) & (1.0 - self.k <= self.alpha_m) & (self.alpha_m <= 1.0),
            -jnp.log(q),
            Z / d,
        )
        Z = jnp.sum(Z)
        Z *= jnp.power(self.mmin, 1.0 - self.alpha_m) / (self.k + 1.0)
        self.logZ = jnp.log(Z)

    @validate_sample
    def log_prob(self, value):
        return (
            -(self.alpha_m + self.k) * jnp.log(value[..., 0])
            + self.k * jnp.log(value[..., 1])
            - jnp.log(value[..., 0] - self.mmin)
            - self.logZ
        )

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
        return jnp.stack([m1, m2], axis=-1)

    def __repr__(self) -> str:
        string = f"Wysocki2019MassModel(alpha_m={self.alpha_m}, k={self.k}, "
        string += f"mmin={self.mmin}, mmax={self.mmax})"
        return string
