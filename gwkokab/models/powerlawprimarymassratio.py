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

from jax import lax, numpy as jnp
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils.misc import get_key
from .truncpowerlaw import TruncatedPowerLaw
from .utils.constraints import mass_ratio_mass_sandwich


class PowerLawPrimaryMassRatio(dist.Distribution):
    r"""Power law model for two-dimensional mass distribution,
    modelling primary mass and conditional mass ratio
    distribution.

    .. math::
    
        p(m_1,q\mid\alpha,\beta) = p(m_1\mid\alpha)p(q \mid m_1, \beta)
    
    .. math::
    
        \begin{align*}
            p(m_1\mid\alpha)   & \propto m_1^{\alpha}, \qquad m_{\text{min}}             \leq m_1 \leq m_{\max} \\
            p(q\mid m_1,\beta) & \propto q^{\beta},    \qquad \frac{m_{\text{min}}}{m_1} \leq q   \leq 1
        \end{align*}
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "beta": dist.constraints.real,
        "mmin": dist.constraints.positive,
        "mmax": dist.constraints.positive,
    }

    def __init__(self, alpha: float, beta: float, mmin: float, mmax: float) -> None:
        self.alpha, self.beta, self.mmin, self.mmax = promote_shapes(alpha, beta, mmin, mmax)
        batch_shape = lax.broadcast_shapes(jnp.shape(alpha), jnp.shape(beta), jnp.shape(mmin), jnp.shape(mmax))
        self.support = mass_ratio_mass_sandwich(self.mmin, self.mmax)
        super(PowerLawPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = TruncatedPowerLaw(
            alpha=self.alpha,
            xmin=self.mmin,
            xmax=self.mmax,
        ).log_prob(m1)
        log_prob_q = TruncatedPowerLaw(
            alpha=self.beta,
            xmin=self.mmin / m1,
            xmax=1.0,
        ).log_prob(q)
        return log_prob_m1 + log_prob_q

    def sample(self, key, sample_shape: tuple = ()):
        if key is None or isinstance(key, int):
            key = get_key(key)

        m1 = TruncatedPowerLaw(
            alpha=self.alpha,
            xmin=self.mmin,
            xmax=self.mmax,
        ).sample(key=key, sample_shape=sample_shape + self.batch_shape)
        key = get_key(key)
        q = TruncatedPowerLaw(
            alpha=self.beta,
            xmin=self.mmin / m1,
            xmax=1.0,
        ).sample(key=key, sample_shape=())

        return jnp.column_stack((m1, q))
