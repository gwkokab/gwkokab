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
from jaxtyping import Array
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils.misc import get_key
from .truncpowerlaw import TruncatedPowerLaw
from .utils.constraints import mass_sandwich


class SimpleWysocki2019MassModel(dist.Distribution):
    r"""It is a double side truncated power law distribution, as
    described in equation 7 of the `paper <https://arxiv.org/abs/1805.06442>`__.

    .. math::
        p(m_1,m_2\mid\alpha,m_{\text{min}},m_{\text{max}},M_{\text{max}})\propto
        \frac{m_1^{-\alpha}}{m_1-m_{\text{min}}}
    """

    arg_constraints = {
        "alpha_m": dist.constraints.real,
        "mmin": dist.constraints.positive,
        "mmax": dist.constraints.positive,
    }

    def __init__(self, alpha_m: float, mmin: float, mmax: float) -> None:
        r"""Initialize the power law distribution with a lower and upper mass limit.

        :param alpha_m: index of the power law distribution
        :param mmin: lower mass limit
        :param mmax: upper mass limit
        :param valid_args: validate the input arguments or not, defaults to `None`
        """
        self.alpha_m, self.mmin, self.mmax = promote_shapes(alpha_m, mmin, mmax)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha_m),
            jnp.shape(mmin),
            jnp.shape(mmax),
        )
        self.support = mass_sandwich(self.mmin, self.mmax)
        super(SimpleWysocki2019MassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        log_prob_m1 = TruncatedPowerLaw(alpha=-self.alpha_m, xmin=self.mmin, xmax=self.mmax).log_prob(m1)
        log_prob_m2 = -jnp.log(m1 - self.mmin)
        return log_prob_m1 + log_prob_m2

    def sample(self, key: Optional[Array | int], sample_shape: tuple = ()) -> Array:
        if key is None or isinstance(key, int):
            key = get_key(key)
        m1 = TruncatedPowerLaw(
            alpha=-self.alpha_m,
            xmin=self.mmin,
            xmax=self.mmax,
        ).sample(key=key, sample_shape=sample_shape + self.batch_shape)
        key = get_key(key)
        m2 = dist.Uniform(
            low=self.mmin,
            high=m1,
        ).sample(key=key, sample_shape=())
        return jnp.column_stack((m1, m2))

    def __repr__(self) -> str:
        string = f"SimpleWysocki2019MassModel(alpha_m={self.alpha_m}, mmin={self.mmin}, mmax={self.mmax})"
        return string
