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

from typing_extensions import Self

from jax import lax, numpy as jnp, random as jrd
from jaxtyping import Float, PRNGKeyArray
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils.mass_relations import mass_ratio
from .truncpowerlaw import TruncatedPowerLaw
from .utils.constraints import mass_ratio_mass_sandwich, mass_sandwich


class PowerLawPrimaryMassRatio(dist.Distribution):
    r"""Power law model for two-dimensional mass distribution,
    modelling primary mass and conditional mass ratio
    distribution.

    $$p(m_1,q\mid\alpha,\beta) = p(m_1\mid\alpha)p(q \mid m_1, \beta)$$
    
    $$
        \begin{align*}
            p(m_1\mid\alpha)   & \propto m_1^{\alpha}, \qquad m_{\text{min}}             \leq m_1 \leq m_{\max} \\
            p(q\mid m_1,\beta) & \propto q^{\beta},    \qquad \frac{m_{\text{min}}}{m_1} \leq q   \leq 1
        \end{align*}
    $$
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "beta": dist.constraints.real,
        "mmin": dist.constraints.dependent,
        "mmax": dist.constraints.dependent,
    }
    reparametrized_params = ["alpha", "beta", "mmin", "mmax"]
    pytree_aux_fields = ("_support",)

    def __init__(self: Self, alpha: Float, beta: Float, mmin: Float, mmax: Float, **kwargs) -> None:
        """
        :param alpha: Power law index for primary mass
        :param beta: Power law index for mass ratio
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param default_params: If `True`, the model will use the default parameters
            i.e. primary mass and secondary mass. If `False`, the model will use
            primary mass and mass ratio.
        """
        self.alpha, self.beta, self.mmin, self.mmax = promote_shapes(alpha, beta, mmin, mmax)
        batch_shape = lax.broadcast_shapes(jnp.shape(alpha), jnp.shape(beta), jnp.shape(mmin), jnp.shape(mmax))
        self._default_params = kwargs.get("default_params", True)
        if self._default_params:
            self._support = mass_sandwich(mmin, mmax)
        else:
            self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(PowerLawPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

    @dist.constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self: Self, value):
        m1 = value[..., 0]
        if self._default_params:
            m2 = value[..., 1]
            q = mass_ratio(m1, m2)
        else:
            q = value[..., 1]
        log_prob_m1 = TruncatedPowerLaw(
            alpha=self.alpha,
            xmin=self.mmin,
            xmax=self.mmax,
        ).log_prob(m1)
        log_prob_q = TruncatedPowerLaw(
            alpha=self.beta,
            xmin=lax.div(self.mmin, m1),
            xmax=1.0,
        ).log_prob(q)
        return log_prob_m1 + log_prob_q

    def sample(self: Self, key: PRNGKeyArray, sample_shape: tuple = ()):
        m1 = TruncatedPowerLaw(
            alpha=self.alpha,
            xmin=self.mmin,
            xmax=self.mmax,
        ).sample(key=key, sample_shape=sample_shape + self.batch_shape)
        key = jrd.split(key)[1]
        q = TruncatedPowerLaw(
            alpha=self.beta,
            xmin=lax.div(self.mmin, m1),
            xmax=1.0,
        ).sample(key=key, sample_shape=())

        if self._default_params:
            return jnp.column_stack((m1, m1 * q))
        return jnp.column_stack((m1, q))
