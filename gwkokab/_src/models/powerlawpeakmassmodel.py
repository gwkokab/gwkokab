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
from typing_extensions import Self

import numpy as np
from jax import jit, lax, numpy as jnp, random as jrd, tree as jtr, vmap
from jax.scipy.stats import norm
from jaxtyping import Array, Float, PRNGKeyArray, Real
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils.transformations import m1_q_to_m2, mass_ratio
from .utils import numerical_inverse_transform_sampling
from .utils.constraints import mass_ratio_mass_sandwich, mass_sandwich
from .utils.smoothing import smoothing_kernel


class PowerLawPeakMassModel(dist.Distribution):
    r"""See equation (B3) and (B6) in [Population Properties of Compact
    Objects from the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog](https://arxiv.org/abs/2010.14533).

    $$
        \begin{align*}
            p(m_1\mid\lambda,\alpha,\delta,m_{\text{min}},m_{\text{max}},
            \mu,\sigma)
            &\propto \left[(1-\lambda)m_1^{-\alpha}\Theta(m_\text{max}-m_1)
            +\frac{\lambda}{\sigma\sqrt{2\pi}}
            e^{-\frac{1}{2}\left(\frac{m_1-\mu}{\sigma}\right)^{2}}\right]
            S(m_1\mid m_{\text{min}},\delta)
            \\
            p(q\mid \beta, m_1,m_{\text{min}},\delta)
            &\propto q^{\beta}S(m_1q\mid m_{\text{min}},\delta)
        \end{align*}
    $$
        
    Where $S(m\mid m_{\text{min}},\delta_m)$ is the smoothing kernel, defined
    in [`gwkokab.models.utils.smoothing
    .smoothing_kernel`](utils.html#gwkokab.models.utils.smoothing.smoothing_kernel),
    and $\Theta$ is the Heaviside step function.
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "beta": dist.constraints.real,
        "lam": dist.constraints.interval(0, 1),
        "delta": dist.constraints.real,
        "mmin": dist.constraints.dependent,
        "mmax": dist.constraints.dependent,
        "mu": dist.constraints.real,
        "sigma": dist.constraints.positive,
    }
    reparametrized_params = [
        "alpha",
        "beta",
        "lam",
        "delta",
        "mmin",
        "mmax",
        "mu",
        "sigma",
    ]
    pytree_aux_fields = ("_logZ", "_support")

    def __init__(
        self: Self,
        alpha: Float,
        beta: Float,
        lam: Float,
        delta: Float,
        mmin: Float,
        mmax: Float,
        mu: Float,
        sigma: Float,
        **kwargs,
    ) -> None:
        r"""
        :param alpha: Power-law index for primary mass model
        :param beta: Power-law index for mass ratio model
        :param lam: Fraction of Gaussian component
        :param delta: Smoothing parameter
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mu: Mean of Gaussian component
        :param sigma: Standard deviation of Gaussian component
        :param default_params: If `True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If `False`, the
            model will use primary mass and mass ratio.
        """
        (
            self.alpha,
            self.beta,
            self.lam,
            self.delta,
            self.mmin,
            self.mmax,
            self.mu,
            self.sigma,
        ) = promote_shapes(alpha, beta, lam, delta, mmin, mmax, mu, sigma)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(lam),
            jnp.shape(delta),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mu),
            jnp.shape(sigma),
        )
        self._default_params = kwargs.get("default_params", True)
        if self._default_params:
            self._support = mass_sandwich(mmin, mmax)
        else:
            self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(PowerLawPeakMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

        self._normalization()

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def _normalization(self):
        """Precomputes the normalization constant for the primary mass model
        and mass ratio model using Monte Carlo integration.
        """
        num_samples = 20_000
        self._logZ = jnp.zeros_like(self.mmin)
        samples = self.sample(
            jrd.PRNGKey(np.random.randint(0, 2**32 - 1)), (num_samples,)
        )
        log_prob = self.log_prob(samples)
        prob = jnp.exp(log_prob)
        volume = jnp.prod(self.mmax - self.mmin)
        self._logZ = jnp.log(jnp.mean(prob, axis=-1)) + jnp.log(volume)

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(
        self: Self, m1: Array | Real
    ) -> Array | Real:
        r"""Log probability of primary mass model.

        .. math::

            p(m_1\mid\lambda,\alpha,\delta,m_{\text{min}},m_{\text{max}},
            \mu,\sigma) \propto \left[(1-\lambda)m_1^{-\alpha}
            \Theta(m_\text{max}-m_1)
            +\frac{\lambda}{\sigma\sqrt{2\pi}}
            e^{-\frac{1}{2}\left(\frac{m_1-\mu}{\sigma}\right)^{2}}\right]
            S(m_1\mid m_{\text{min}},\delta)

        :param m1: primary mass
        :return: log probability of primary mass
        """
        gaussian_term = jnp.exp(
            jnp.log(self.lam) + norm.logpdf(m1, self.mu, self.sigma)
        )
        powerlaw_term = jnp.where(
            m1 < self.mmax,
            jnp.exp(jnp.log(1 - self.lam) - self.alpha * jnp.log(m1)),
            jnp.zeros_like(m1),
        )
        log_prob_val = jnp.log(powerlaw_term + gaussian_term) + jnp.log(
            smoothing_kernel(m1, self.mmin, self.delta)
        )
        return log_prob_val

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(
        self: Self, m1: Array | Real, q: Array | Real
    ) -> Array | Real:
        r"""Log probability of mass ratio model

        .. math::

            \log p(q\mid m_1) = \beta \log q +
            \log S(m_1q\mid m_{\text{min}},\delta_m)

        :param m1: primary mass
        :param q: mass ratio
        :return: log probability of mass ratio model
        """
        log_smoothing_val = jnp.log(
            smoothing_kernel(m1_q_to_m2(m1=m1, q=q), self.mmin, self.delta)
        )
        return self.beta * jnp.log(q) + log_smoothing_val

    @validate_sample
    def log_prob(self: Self, value):
        m1 = value[..., 0]
        if self._default_params:
            m2 = value[..., 1]
            q = mass_ratio(m1=m1, m2=m2)
        else:
            q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        return log_prob_m1 + log_prob_q - self._logZ

    def sample(self, key: PRNGKeyArray, sample_shape: tuple = ()):
        flattened_sample_shape = jtr.reduce(lambda x, y: x * y, sample_shape, 1)

        m1 = numerical_inverse_transform_sampling(
            logpdf=self._log_prob_primary_mass_model,
            limits=(self.mmin, self.mmax),
            sample_shape=(flattened_sample_shape,),
            key=key,
            batch_shape=self.batch_shape,
            n_grid_points=1000,
        )

        keys = jrd.split(key, m1.shape)

        q = vmap(
            lambda _m1, _k: numerical_inverse_transform_sampling(
                logpdf=partial(self._log_prob_mass_ratio_model, _m1),
                limits=(self.mmin / _m1, 1.0),
                sample_shape=(),
                key=_k,
                batch_shape=self.batch_shape,
                n_grid_points=1000,
            )
        )(m1, keys)

        if self._default_params:
            return jnp.column_stack([m1, m1_q_to_m2(m1=m1, q=q)]).reshape(
                sample_shape + self.event_shape
            )
        return jnp.column_stack([m1, q]).reshape(
            sample_shape + self.event_shape
        )