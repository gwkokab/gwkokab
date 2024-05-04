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

from functools import partial

from jax import jit, lax
from jax import numpy as jnp
from jax import random as jrd
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample

from ..typing import Numeric
from ..utils import get_key
from .truncpowerlaw import TruncatedPowerLaw
from .utils.constraints import mass_ratio_mass_sandwich
from .utils.smoothing import smoothing_kernel


class BrokenPowerLawMassModel(dist.Distribution):
    r"""See equation (B7) and (B6) in `Population Properties of Compact Objects
    from the Second LIGO-Virgo Gravitational-Wave Transient Catalog
    <https://arxiv.org/abs/2010.14533>`__.

    
    $$
        \begin{align*}
            p(m_1) &\propto \begin{cases}
                m_1^{-\alpha_1}S(m_1\mid m_{\text{min}},\delta_m)
                & \text{if } m_{\min} \leq m_1 < m_{\text{break}} \\
                m_1^{-\alpha_2}S(m_1\mid m_{\text{min}},\delta_m)
                & \text{if } m_{\text{break}} < m_1 \leq m_{\max} \\
                0 & \text{otherwise}
            \end{cases} \\
            p(q\mid m_1) &\propto q^{\beta_q}S(m_1q\mid m_{\text{min}},\delta_m)
        \end{align*}
    $$
        
    Where :math:`S(m\mid m_{\text{min}},\delta_m)` is the smoothing kernel,
    defined in :func:`gwkokab.models.utils.smoothing.smoothing_kernel`.
    """

    arg_constraints = {
        "alpha1": dist.constraints.real,
        "alpha2": dist.constraints.real,
        "beta_q": dist.constraints.real,
        "mmin": dist.constraints.positive,
        "mmax": dist.constraints.positive,
        "mbreak": dist.constraints.positive,
        "delta": dist.constraints.positive,
    }

    def __init__(
        self, alpha1: float, alpha2: float, beta_q: float, mmin: float, mmax: float, mbreak: float, delta: float
    ):
        self.alpha1, self.alpha2, self.beta_q, self.mmin, self.mmax, self.mbreak, self.delta = promote_shapes(
            alpha1, alpha2, beta_q, mmin, mmax, mbreak, delta
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha1),
            jnp.shape(alpha2),
            jnp.shape(beta_q),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mbreak),
            jnp.shape(delta),
        )
        self.support = mass_ratio_mass_sandwich(self.mmin, self.mmax)
        super(BrokenPowerLawMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(),
            validate_args=True,
        )
        self.alpha1_powerlaw = TruncatedPowerLaw(
            alpha=-self.alpha1,
            xmin=self.mmin,
            xmax=self.mbreak,
        )
        self.alpha2_powerlaw = TruncatedPowerLaw(
            alpha=-self.alpha2,
            xmin=self.mbreak,
            xmax=self.mmax,
        )
        self._normalization()

    def _normalization(self):
        """Precomputes the normalization constant for the primary mass model
        and mass ratio model using Monte Carlo integration.
        """
        num_samples = 20000

        mm1 = jrd.uniform(  # mmin <= xx < mmax
            get_key(),
            shape=(num_samples,),
            minval=self.mmin,
            maxval=self.mmax,
        )
        qq = jrd.uniform(  # mmin / mm1 <= qq < 1
            get_key(),
            shape=(num_samples,),
            minval=self.mmin / mm1,
            maxval=1,
        )

        value = jnp.column_stack([mm1, qq])

        self._logZ = jnp.zeros_like(self.mmin)
        log_prob = self.log_prob(value)
        prob = jnp.exp(log_prob)

        self._logZ = jnp.log(jnp.sum(prob))

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(self, m1: Numeric) -> Numeric:
        r"""Log probability of primary mass model.
        
        .. math::

            \log p(m_1) = \begin{cases}
                \log S(m_1\mid m_{\text{min}},\delta_m) - \log Z_{\text{primary}}
                - \alpha_1 \log m_1 & \text{if } m_{\min} \leq m_1 < m_{\text{break}} \\
                \log S(m_1\mid m_{\text{min}},\delta_m) - \log Z_{\text{primary}}
                - \alpha_2 \log m_1 & \text{if } m_{\text{break}} < m_1 \leq m_{\max}
            \end{cases}
            
        :param m1: primary mass 
        :return: log probability of primary mass
        """
        conditions = [
            (self.mmin <= m1) & (m1 < self.mbreak),
            (self.mbreak <= m1) & (m1 <= self.mmax),
        ]
        log_smoothing_val = jnp.log(smoothing_kernel(m1, self.mmin, self.delta))
        log_probs = [
            log_smoothing_val + self.alpha1_powerlaw.log_prob(m1),
            log_smoothing_val + self.alpha2_powerlaw.log_prob(m1),
        ]
        return jnp.select(conditions, log_probs, default=jnp.full_like(m1, -jnp.inf))

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(self, m1: Numeric, q: Numeric) -> Numeric:
        r"""Log probability of mass ratio model

        .. math::

            \log p(q\mid m_1) = \beta_q \log q + \log S(m_1q\mid m_{\text{min}},\delta_m)

        :param m1: primary mass
        :param q: mass ratio
        :return: log probability of mass ratio model
        """
        log_smoothing_val = jnp.log(smoothing_kernel(m1 * q, self.mmin, self.delta))
        return self.beta_q * jnp.log(q) + log_smoothing_val

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        return log_prob_m1 + log_prob_q - self._logZ
