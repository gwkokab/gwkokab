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

from jax import jit, lax, numpy as jnp, random as jrd
from jax.scipy.stats import norm
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample

from ..typing import Numeric
from ..utils import get_key
from .utils.constraints import mass_ratio_mass_sandwich
from .utils.smoothing import smoothing_kernel


class MultiPeakMassModel(dist.Distribution):
    r"""See equation (B9) and (B6) in [Population Properties of Compact
    Objects from the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog](https://arxiv.org/abs/2010.14533).
    
    $$
        p(m_1\mid\lambda,\lambda_1,\alpha,\delta,m_{\text{min}},m_{\text{max}},\mu,\sigma)\propto
        \left[(1-\lambda)m_1^{-\alpha}\Theta(m_\text{max}-m_1)
        +\lambda\lambda_1\varphi\left(\frac{m_1-\mu_1}{\sigma_1}\right)
        +\lambda(1-\lambda_1)\varphi\left(\frac{m_1-\mu_2}{\sigma_2}\right)
        \right]S(m_1\mid m_{\text{min}},\delta)
    $$
      
    $$
        \begin{align*}
            p(q\mid \beta, m_1,m_{\text{min}},\delta)&\propto q^{\beta}S(m_1q\mid m_{\text{min}},\delta)\\
        \end{align*}
    $$
      
    Where,
    
    $$\varphi(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp{\left(\displaystyle-\frac{x^{2}}{2}\right)}$$
    
    
    $S(m\mid m_{\text{min}},\delta_m)$ is the smoothing kernel,
    defined in [`gwkokab.models.utils.smoothing
    .smoothing_kernel`](utils.html#gwkokab.models.utils.smoothing.smoothing_kernel),
    and $\Theta$ is the Heaviside step function.
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "beta": dist.constraints.real,
        "lam": dist.constraints.interval(0, 1),
        "lam1": dist.constraints.interval(0, 1),
        "delta": dist.constraints.positive,
        "mmin": dist.constraints.positive,
        "mmax": dist.constraints.positive,
        "mu1": dist.constraints.positive,
        "sigma1": dist.constraints.positive,
        "mu2": dist.constraints.positive,
        "sigma2": dist.constraints.positive,
    }

    def __init__(
        self,
        alpha: float,
        beta: float,
        lam: float,
        lam1: float,
        delta: float,
        mmin: float,
        mmax: float,
        mu1: float,
        sigma1: float,
        mu2: float,
        sigma2: float,
    ):
        r"""
        :param alpha: Power-law index for primary mass model
        :param beta: Power-law index for mass ratio model
        :param lam: weight for power-law component
        :param lam1: weight for first Gaussian component
        :param delta: Smoothing parameter
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mu1: Mean of first Gaussian component
        :param sigma1: Standard deviation of first Gaussian component
        :param mu2: Mean of second Gaussian component
        :param sigma2: Standard deviation of second Gaussian component
        """
        (
            self.alpha,
            self.beta,
            self.lam,
            self.lam1,
            self.delta,
            self.mmin,
            self.mmax,
            self.mu1,
            self.sigma1,
            self.mu2,
            self.sigma2,
        ) = promote_shapes(alpha, beta, lam, lam1, delta, mmin, mmax, mu1, sigma1, mu2, sigma2)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(lam),
            jnp.shape(lam1),
            jnp.shape(delta),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mu1),
            jnp.shape(sigma1),
            jnp.shape(mu2),
            jnp.shape(sigma2),
        )
        self.support = mass_ratio_mass_sandwich(self.mmin, self.mmax)
        super(MultiPeakMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

        self._normalization()

    def _normalization(self):
        """Precomputes the normalization constant for the primary mass model
        and mass ratio model using Monte Carlo integration.
        """
        num_samples = 20_000

        mm1 = jrd.uniform(  # mmin <= xx < mmax
            get_key(),
            shape=(num_samples,),
            minval=self.mmin,
            maxval=self.mmax,
        )
        qq = jrd.uniform(  # 0 <= qq < 1
            get_key(),
            shape=(num_samples,),
            minval=0,
            maxval=1,
        )

        value = jnp.column_stack([mm1, qq])
        support = self.support(value)
        self._logZ = jnp.zeros_like(self.mmin)
        log_prob = self.log_prob(value)
        prob = jnp.exp(log_prob)
        self._logZ = jnp.log(jnp.mean(prob, axis=-1, where=support))

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(self, m1: Numeric) -> Numeric:
        r"""Log probability of primary mass model.

        $$
            \begin{multline*}
                p(m_1\mid\lambda,\lambda_1,\alpha,\delta,m_{\text{min}},m_{\text{max}},\mu,\sigma)\propto\\
                \left[(1-\lambda)m_1^{-\alpha}\Theta(m_\text{max}-m_1)
                +\lambda\lambda_1\varphi\left(\frac{m_1-\mu_1}{\sigma_1}\right)
                +\lambda(1-\lambda_1)\varphi\left(\frac{m_1-\mu_2}{\sigma_2}\right)
                \right]S(m_1\mid m_{\text{min}},\delta)
            \end{multline*}
        $$
        
        :param m1: primary mass
        :return: log probability of primary mass
        """
        gaussian_term1 = jnp.exp(jnp.log(self.lam) + jnp.log(self.lam1) + norm.logpdf(m1, self.mu1, self.sigma1))
        gaussian_term2 = jnp.exp(jnp.log(self.lam) + jnp.log(1 - self.lam1) + norm.logpdf(m1, self.mu2, self.sigma2))
        powerlaw_term = jnp.where(
            m1 < self.mmax, jnp.exp(jnp.log(1 - self.lam) - self.alpha * jnp.log(m1)), jnp.zeros_like(m1)
        )
        log_prob_val = jnp.log(powerlaw_term + gaussian_term1 + gaussian_term2)
        log_prob_val += jnp.log(smoothing_kernel(m1, self.mmin, self.delta))
        return log_prob_val

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(self, m1: Numeric, q: Numeric) -> Numeric:
        r"""Log probability of mass ratio model

        $$\log p(q\mid m_1) = \beta \log q + \log S(m_1q\mid m_{\text{min}},\delta_m)$$

        :param m1: primary mass
        :param q: mass ratio
        :return: log probability of mass ratio model
        """
        log_smoothing_val = jnp.log(smoothing_kernel(m1 * q, self.mmin, self.delta))
        return self.beta * jnp.log(q) + log_smoothing_val

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        return log_prob_m1 + log_prob_q - self._logZ