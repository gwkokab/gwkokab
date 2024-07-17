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

import numpy as np
from jax import jit, lax, numpy as jnp, random as jrd, tree as jtr, vmap
from jax.scipy.stats import norm, uniform
from jaxtyping import Array, Int, Real
from numpyro.distributions import (
    Beta,
    Categorical,
    constraints,
    Distribution,
    MixtureGeneral,
    MultivariateNormal,
    Normal,
    TruncatedNormal,
    Uniform,
)
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key

from ..utils.math import beta_dist_mean_variance_to_concentrations
from ..utils.transformations import m1_q_to_m2
from .constraints import mass_ratio_mass_sandwich, mass_sandwich
from .utils import (
    JointDistribution,
    numerical_inverse_transform_sampling,
    smoothing_kernel,
)


__all__ = [
    "BrokenPowerLawMassModel",
    "GaussianSpinModel",
    "IndependentSpinOrientationGaussianIsotropic",
    "MultiPeakMassModel",
    "MultiSpinModel",
    "NDistribution",
    "PowerLawPeakMassModel",
    "PowerLawPrimaryMassRatio",
    "TruncatedPowerLaw",
    "Wysocki2019MassModel",
]


class BrokenPowerLawMassModel(Distribution):
    r"""See equation (B7) and (B6) in `Population Properties of Compact Objects
    from the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog <https://arxiv.org/abs/2010.14533>`_.

    .. math::
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

    Where :math:`S(m\mid m_{\text{min}},\delta_m)` is the smoothing kernel.
    """

    arg_constraints = {
        "alpha1": constraints.real,
        "alpha2": constraints.real,
        "beta_q": constraints.real,
        "mmin": constraints.dependent,
        "mmax": constraints.dependent,
        "mbreak": constraints.positive,
        "delta": constraints.positive,
    }
    reparametrized_params = [
        "alpha1",
        "alpha2",
        "beta_q",
        "mmin",
        "mmax",
        "mbreak",
        "delta",
    ]
    pytree_aux_fields = ("_support",)
    pytree_data_fields = (
        "alpha1",
        "alpha2",
        "beta_q",
        "mmin",
        "mmax",
        "mbreak",
        "delta",
        "_logZ",
    )

    def __init__(self, alpha1, alpha2, beta_q, mmin, mmax, mbreak, delta):
        r"""
        :param alpha1: Power-law index for first component of primary mass model
        :param alpha2: Power-law index for second component of primary mass
            model
        :param beta_q: Power-law index for mass ratio model
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mbreak: Break mass
        :param delta: Smoothing parameter
        :param default_params: If :code:`True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If :code:`False`, the
            model will use primary mass and mass ratio.
        """
        (
            self.alpha1,
            self.alpha2,
            self.beta_q,
            self.mmin,
            self.mmax,
            self.mbreak,
            self.delta,
        ) = promote_shapes(alpha1, alpha2, beta_q, mmin, mmax, mbreak, delta)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha1),
            jnp.shape(alpha2),
            jnp.shape(beta_q),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mbreak),
            jnp.shape(delta),
        )
        self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(BrokenPowerLawMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )
        self.alpha1_powerlaw = TruncatedPowerLaw(
            alpha=jnp.negative(self.alpha1),
            low=self.mmin,
            high=self.mbreak,
            validate_args=True,
        )
        self.alpha2_powerlaw = TruncatedPowerLaw(
            alpha=jnp.negative(self.alpha2),
            low=self.mbreak,
            high=self.mmax,
            validate_args=True,
        )
        self._normalization()

    @constraints.dependent_property(is_discrete=False, event_dim=0)
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
        volume = jnp.prod(jnp.subtract(self.mmax, self.mmin))
        self._logZ = jnp.add(jnp.log(jnp.mean(prob, axis=-1)), jnp.log(volume))

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(self, m1: Array | Real) -> Array | Real:
        r"""Log probability of primary mass model.

        .. math::
            \log p(m_1) = \begin{cases}
                \log S(m_1\mid m_{\text{min}},\delta_m)
                - \log Z_{\text{primary}}
                - \alpha_1 \log m_1
                & \text{if } m_{\min} \leq m_1 < m_{\text{break}} \\
                \log S(m_1\mid m_{\text{min}},\delta_m)
                - \log Z_{\text{primary}}
                - \alpha_2 \log m_1
                & \text{if } m_{\text{break}} < m_1 \leq m_{\max}
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
            jnp.add(log_smoothing_val, self.alpha1_powerlaw.log_prob(m1)),
            jnp.add(log_smoothing_val, self.alpha2_powerlaw.log_prob(m1)),
        ]
        return jnp.select(conditions, log_probs, default=jnp.full_like(m1, -jnp.inf))

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(
        self, m1: Array | Real, q: Array | Real
    ) -> Array | Real:
        r"""Log probability of mass ratio model

        .. math::

            \log p(q\mid m_1) = \beta_q \log q +
            \log S(m_1q\mid m_{\text{min}},\delta_m)

        :param m1: primary mass
        :param q: mass ratio
        :return: log probability of mass ratio model
        """
        log_smoothing_val = jnp.log(
            smoothing_kernel(m1_q_to_m2(m1=m1, q=q), self.mmin, self.delta)
        )
        log_val = jnp.log(q)
        log_val = jnp.multiply(self.beta_q, log_val)
        return jnp.add(log_val, log_smoothing_val)

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        log_val = jnp.add(log_prob_m1, log_prob_q)
        return jnp.subtract(log_val, self._logZ)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
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
                limits=(jnp.divide(self.mmin, _m1), 1.0),
                sample_shape=(),
                key=_k,
                batch_shape=self.batch_shape,
                n_grid_points=1000,
            )
        )(m1, keys)

        return jnp.column_stack([m1, q]).reshape(sample_shape + self.event_shape)


def GaussianSpinModel(mu_eff, sigma_eff, mu_p, sigma_p, rho) -> MultivariateNormal:
    r"""Bivariate normal distribution for the effective and precessing spins.
    See Eq. (D3) and (D4) in `Population Properties of Compact Objects from
    the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog <https://arxiv.org/abs/2010.14533>`_.

    .. math::
        \left(\chi_{\text{eff}}, \chi_{p}\right) \sim \mathcal{N}\left(
            \begin{bmatrix}
                \mu_{\text{eff}} \\ \mu_{p}
            \end{bmatrix},
            \begin{bmatrix}
                \sigma_{\text{eff}}^2 & \rho \sigma_{\text{eff}} \sigma_{p} \\
                \rho \sigma_{\text{eff}} \sigma_{p} & \sigma_{p}^2
            \end{bmatrix}
        \right)

    where :math:`\chi_{\text{eff}}` is the effective spin and
    :math:`\chi_{\text{eff}}\in[-1,1]` and :math:`\chi_{p}` is the precessing spin and
    :math:`\chi_{p}\in[0,1]`.

    :param mu_eff: mean of the effective spin
    :param sigma_eff: standard deviation of the effective spin
    :param mu_p: mean of the precessing spin
    :param sigma_p: standard deviation of the precessing spin
    :param rho: correlation coefficient between the effective and precessing
        spins
    :return: Multivariate normal distribution for the effective and precessing
        spins
    """
    return MultivariateNormal(
        loc=jnp.array([mu_eff, mu_p]),
        covariance_matrix=jnp.array(
            [
                [
                    jnp.square(sigma_eff),
                    jnp.multiply(rho, jnp.multiply(sigma_eff, sigma_p)),
                ],
                [
                    jnp.multiply(rho, jnp.multiply(sigma_eff, sigma_p)),
                    jnp.square(sigma_p),
                ],
            ]
        ),
        validate_args=True,
    )


def IndependentSpinOrientationGaussianIsotropic(zeta, sigma1, sigma2) -> MixtureGeneral:
    r"""A mixture model of spin orientations with isotropic and normally
    distributed components. See Eq. (4) of `Determining the population
    properties of spinning black holes <https://arxiv.org/abs/1704.08370>`_.

    .. math::
        p(z_1,z_2\mid\zeta,\sigma_1,\sigma_2) = \frac{1-\zeta}{4} +
        \zeta\mathbb{I}_{[-1,1]}(z_1)\mathbb{I}_{[-1,1]}(z_2)
        \mathcal{N}(z_1\mid 1,\sigma_1)\mathcal{N}(z_2\mid 1,\sigma_2)

    where :math:`\mathbb{I}(\cdot)` is the indicator function.

    :param zeta: The mixing probability of the second component.
    :param sigma1: The standard deviation of the first component.
    :param sigma2: The standard deviation of the second component.
    :return: Mixture model of spin orientations.
    """
    mixing_probs = jnp.array([1 - zeta, zeta])
    component_0_dist = Uniform(low=-1, high=1, validate_args=True)
    component_1_dist = TruncatedNormal(
        loc=1.0,
        scale=jnp.array([sigma1, sigma2]),
        low=-1,
        high=1,
        validate_args=True,
    )
    return MixtureGeneral(
        mixing_distribution=Categorical(probs=mixing_probs),
        component_distributions=[component_0_dist, component_1_dist],
        support=constraints.real,
        validate_args=True,
    )


class MultiPeakMassModel(Distribution):
    r"""See equation (B9) and (B6) in `Population Properties of Compact
    Objects from the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog <https://arxiv.org/abs/2010.14533>`_.

    .. math::
        p(m_1\mid\lambda,\lambda_1,\alpha,\delta,m_{\text{min}},m_{\text{max}},
        \mu,\sigma)\propto \left[(1-\lambda)m_1^{-\alpha}
        \Theta(m_\text{max}-m_1)
        +\lambda\lambda_1\varphi\left(\frac{m_1-\mu_1}{\sigma_1}\right)
        +\lambda(1-\lambda_1)\varphi\left(\frac{m_1-\mu_2}{\sigma_2}\right)
        \right]S(m_1\mid m_{\text{min}},\delta)

    .. math::
        p(q\mid \beta, m_1,m_{\text{min}},\delta)\propto
        q^{\beta}S(m_1q\mid m_{\text{min}},\delta)

    Where,

    .. math::
        \varphi(x)=\frac{1}{\sigma\sqrt{2\pi}}
        \exp{\left(\displaystyle-\frac{x^{2}}{2}\right)}

    :math:`S(m\mid m_{\text{min}},\delta_m)` is the smoothing kernel,
    and :math:`\Theta` is the Heaviside step function.
    """

    arg_constraints = {
        "alpha": constraints.real,
        "beta": constraints.real,
        "lam": constraints.interval(0, 1),
        "lam1": constraints.interval(0, 1),
        "delta": constraints.positive,
        "mmin": constraints.dependent,
        "mmax": constraints.dependent,
        "mu1": constraints.positive,
        "sigma1": constraints.positive,
        "mu2": constraints.positive,
        "sigma2": constraints.positive,
    }
    reparametrized_params = [
        "alpha",
        "beta",
        "lam",
        "lam1",
        "delta",
        "mmin",
        "mmax",
        "mu1",
        "sigma1",
        "mu2",
        "sigma2",
    ]
    pytree_aux_fields = ("_support",)
    pytree_data_fields = (
        "alpha",
        "beta",
        "lam",
        "lam1",
        "delta",
        "mmin",
        "mmax",
        "mu1",
        "sigma1",
        "mu2",
        "sigma2",
        "_logZ",
    )

    def __init__(
        self, alpha, beta, lam, lam1, delta, mmin, mmax, mu1, sigma1, mu2, sigma2
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
        :param default_params: If :code:`True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If :code:`False`, the
            model will use primary mass and mass ratio.
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
        ) = promote_shapes(
            alpha, beta, lam, lam1, delta, mmin, mmax, mu1, sigma1, mu2, sigma2
        )
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
        self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(MultiPeakMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

        self._normalization()

    @constraints.dependent_property(is_discrete=False, event_dim=0)
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
        volume = jnp.prod(jnp.subtract(self.mmax, self.mmin))
        self._logZ = jnp.add(jnp.log(jnp.mean(prob, axis=-1)), jnp.log(volume))

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(self, m1: Array | Real) -> Array | Real:
        r"""Log probability of primary mass model.

        .. math::
            \begin{multline*}
                p(m_1\mid\lambda,\lambda_1,\alpha,\delta,m_{\text{min}},
                m_{\text{max}},\mu,\sigma)\propto\\
                \left[(1-\lambda)m_1^{-\alpha}\Theta(m_\text{max}-m_1)
                +\lambda\lambda_1\varphi\left(\frac{m_1-\mu_1}{\sigma_1}\right)
                +\lambda(1-\lambda_1)\varphi\left(\frac{m_1-\mu_2}{\sigma_2}
                \right)\right]S(m_1\mid m_{\text{min}},\delta)
            \end{multline*}

        :param m1: primary mass
        :return: log probability of primary mass
        """
        gaussian_term1 = jnp.add(jnp.log(self.lam), jnp.log(self.lam1))
        gaussian_term1 = jnp.add(
            gaussian_term1,
            norm.logpdf(m1, self.mu1, self.sigma1),
        )
        gaussian_term1 = jnp.exp(gaussian_term1)

        gaussian_term2 = jnp.add(jnp.log(self.lam), jnp.log(jnp.subtract(1, self.lam1)))
        gaussian_term2 = jnp.add(
            gaussian_term2,
            norm.logpdf(m1, self.mu2, self.sigma2),
        )
        gaussian_term2 = jnp.exp(gaussian_term2)

        powerlaw_term = jnp.where(
            jnp.less(m1, self.mmax),
            jnp.exp(
                jnp.subtract(
                    jnp.log(jnp.subtract(1, self.lam)),
                    jnp.multiply(self.alpha, jnp.log(m1)),
                )
            ),
            jnp.zeros_like(m1),
        )
        log_prob_val = jnp.add(powerlaw_term, gaussian_term1)
        log_prob_val = jnp.add(log_prob_val, gaussian_term2)
        log_prob_val = jnp.log(log_prob_val)
        log_prob_val = jnp.add(
            log_prob_val, jnp.log(smoothing_kernel(m1, self.mmin, self.delta))
        )
        return log_prob_val

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(
        self, m1: Array | Real, q: Array | Real
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
        log_prob_val = jnp.multiply(self.beta, jnp.log(q))
        return jnp.add(log_prob_val, log_smoothing_val)

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        return jnp.subtract(jnp.add(log_prob_m1, log_prob_q), self._logZ)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        flattened_sample_shape = jtr.reduce(lambda x, y: x * y, sample_shape, 1)

        m1 = numerical_inverse_transform_sampling(
            logpdf=self._log_prob_primary_mass_model,
            limits=(self.mmin, self.mmax),
            sample_shape=(flattened_sample_shape,),
            key=key,
            batch_shape=self.batch_shape,
            n_grid_points=1000,
        )

        key = jrd.split(key, m1.shape)

        q = vmap(
            lambda _m1, _k: numerical_inverse_transform_sampling(
                logpdf=partial(self._log_prob_mass_ratio_model, _m1),
                limits=(jnp.divide(self.mmin, _m1), 1.0),
                sample_shape=(),
                key=_k,
                batch_shape=self.batch_shape,
                n_grid_points=1000,
            )
        )(m1, key)

        return jnp.column_stack([m1, q]).reshape(sample_shape + self.event_shape)


def NDistribution(distribution: Distribution, n: Int, **params) -> MixtureGeneral:
    """Mixture of any :math:`n` distributions.

    :param distribution: distribution to mix
    :param n: number of components
    :return: Mixture of :math:`n` distributions
    """
    arg_names = distribution.arg_constraints.keys()
    mixing_dist = Categorical(probs=jnp.divide(jnp.ones(n), n), validate_args=True)
    args_per_component = [
        {arg: params.get(f"{arg}_{i}") for arg in arg_names} for i in range(n)
    ]
    component_dists = jtr.map(
        lambda x: distribution(**x),
        args_per_component,
        is_leaf=lambda x: isinstance(x, dict),
    )
    return MixtureGeneral(
        mixing_dist,
        component_dists,
        support=distribution.support,
        validate_args=True,
    )


class PowerLawPeakMassModel(Distribution):
    r"""See equation (B3) and (B6) in `Population Properties of Compact
    Objects from the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog <https://arxiv.org/abs/2010.14533>`_.

    .. math::
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

    Where :math:`S(m\mid m_{\text{min}},\delta_m)` is the smoothing kernel,
    and :math:`\Theta` is the Heaviside step function.
    """

    arg_constraints = {
        "alpha": constraints.real,
        "beta": constraints.real,
        "lam": constraints.interval(0, 1),
        "delta": constraints.real,
        "mmin": constraints.dependent,
        "mmax": constraints.dependent,
        "mu": constraints.real,
        "sigma": constraints.positive,
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
    pytree_aux_fields = ("_support",)
    pytree_data_fields = (
        "alpha",
        "beta",
        "lam",
        "delta",
        "mmin",
        "mmax",
        "mu",
        "sigma",
        "_logZ",
    )

    def __init__(self, alpha, beta, lam, delta, mmin, mmax, mu, sigma) -> None:
        r"""
        :param alpha: Power-law index for primary mass model
        :param beta: Power-law index for mass ratio model
        :param lam: Fraction of Gaussian component
        :param delta: Smoothing parameter
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mu: Mean of Gaussian component
        :param sigma: Standard deviation of Gaussian component
        :param default_params: If :code:`True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If :code:`False`, the
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
        self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(PowerLawPeakMassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

        self._normalization()

    @constraints.dependent_property(is_discrete=False, event_dim=0)
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
        volume = jnp.prod(jnp.subtract(self.mmax, self.mmin))
        self._logZ = jnp.add(jnp.log(jnp.mean(prob, axis=-1)), jnp.log(volume))

    @partial(jit, static_argnums=(0,))
    def _log_prob_primary_mass_model(self, m1: Array | Real) -> Array | Real:
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
            jnp.add(jnp.log(self.lam), norm.logpdf(m1, self.mu, self.sigma))
        )
        powerlaw_term = jnp.where(
            jnp.less(m1, self.mmax),
            jnp.exp(
                jnp.subtract(
                    jnp.log(jnp.subtract(1, self.lam)),
                    jnp.multiply(self.alpha, jnp.log(m1)),
                )
            ),
            jnp.zeros_like(m1),
        )
        log_prob_val = jnp.add(powerlaw_term, gaussian_term)
        log_prob_val = jnp.log(log_prob_val)
        log_prob_val = jnp.add(
            log_prob_val, jnp.log(smoothing_kernel(m1, self.mmin, self.delta))
        )
        return log_prob_val

    @partial(jit, static_argnums=(0,))
    def _log_prob_mass_ratio_model(
        self, m1: Array | Real, q: Array | Real
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
        log_prob_val = jnp.multiply(self.beta, jnp.log(q))
        return jnp.add(log_prob_val, log_smoothing_val)

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = self._log_prob_primary_mass_model(m1)
        log_prob_q = self._log_prob_mass_ratio_model(m1, q)
        return jnp.subtract(jnp.add(log_prob_m1, log_prob_q), self._logZ)

    def sample(self, key, sample_shape=()):
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
                limits=(jnp.divide(self.mmin, _m1), 1.0),
                sample_shape=(),
                key=_k,
                batch_shape=self.batch_shape,
                n_grid_points=1000,
            )
        )(m1, keys)

        return jnp.column_stack([m1, q]).reshape(sample_shape + self.event_shape)


class PowerLawPrimaryMassRatio(Distribution):
    r"""Power law model for two-dimensional mass distribution,
    modelling primary mass and conditional mass ratio
    distribution.

    .. math::
        p(m_1,q\mid\alpha,\beta) = p(m_1\mid\alpha)p(q \mid m_1, \beta)

    .. math::
        \begin{align*}
            p(m_1\mid\alpha)&
            \propto m_1^{\alpha},\qquad m_{\text{min}}\leq m_1\leq m_{\max}\\
            p(q\mid m_1,\beta)&
            \propto q^{\beta},\qquad \frac{m_{\text{min}}}{m_1}\leq q\leq 1
        \end{align*}
    """

    arg_constraints = {
        "alpha": constraints.real,
        "beta": constraints.real,
        "mmin": constraints.dependent,
        "mmax": constraints.dependent,
    }
    reparametrized_params = ["alpha", "beta", "mmin", "mmax"]
    pytree_aux_fields = ("_support",)

    def __init__(self, alpha, beta, mmin, mmax) -> None:
        """
        :param alpha: Power law index for primary mass
        :param beta: Power law index for mass ratio
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param default_params: If :code:`True`, the model will use the default
            parameters i.e. primary mass and secondary mass. If :code:`False`, the
            model will use primary mass and mass ratio.
        """
        self.alpha, self.beta, self.mmin, self.mmax = promote_shapes(
            alpha, beta, mmin, mmax
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha), jnp.shape(beta), jnp.shape(mmin), jnp.shape(mmax)
        )
        self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(PowerLawPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = TruncatedPowerLaw(
            alpha=self.alpha,
            low=self.mmin,
            high=self.mmax,
            validate_args=True,
        ).log_prob(m1)
        log_prob_q = TruncatedPowerLaw(
            alpha=self.beta,
            low=jnp.divide(self.mmin, m1),
            high=1.0,
            validate_args=True,
        ).log_prob(q)
        return jnp.add(log_prob_m1, log_prob_q)

    def sample(self, key, sample_shape=()):
        m1 = TruncatedPowerLaw(
            alpha=self.alpha,
            low=self.mmin,
            high=self.mmax,
            validate_args=True,
        ).sample(key=key, sample_shape=sample_shape + self.batch_shape)
        key = jrd.split(key)[1]
        q = TruncatedPowerLaw(
            alpha=self.beta,
            low=jnp.divide(self.mmin, m1),
            high=1.0,
            validate_args=True,
        ).sample(key=key, sample_shape=())

        return jnp.column_stack((m1, q))


class TruncatedPowerLaw(Distribution):
    r"""A generic double side truncated power law distribution.

    .. note::
        There are many different definition of Power Law that include
        exponential cut-offs and interval cut-offs.  They are just
        interchangeably. This class is the implementation of power law that has
        been restricted over a closed interval.

    .. math::
        p(x\mid\alpha, x_{\text{min}}, x_{\text{max}}):=
        \begin{cases}
            \displaystyle\frac{x^{\alpha}}{\mathcal{Z}}
            & 0<x_{\text{min}}\leq x\leq x_{\text{max}}\\
            0 & \text{otherwise}
        \end{cases}

    where :math:`\mathcal{Z}` is the normalization constant and :math:`\alpha` is the power
    law index. :math:`x_{\text{min}}` and :math:`x_{\text{max}}` are the lower and upper
    truncation limits, respectively. The normalization constant is given by,

    .. math::
        \mathcal{Z}:=\begin{cases}
            \log{x_{\text{max}}}-\log{x_{\text{min}}} & \alpha = -1 \\
            \displaystyle
            \frac{x_{\text{max}}^{1+\alpha}-x_{\text{min}}^{1+\alpha}}{1+\alpha}
            & \text{otherwise}
        \end{cases}
    """

    arg_constraints = {
        "alpha": constraints.real,
        "low": constraints.positive,
        "high": constraints.positive,
    }
    reparametrized_params = ["low", "high", "alpha"]

    def __init__(self, alpha, low=0.0, high=1.0, validate_args=None):
        self.low, self.high, self.alpha = promote_shapes(low, high, alpha)
        self._support = constraints.interval(low, high)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(low),
            jnp.shape(high),
            jnp.shape(alpha),
        )
        super(TruncatedPowerLaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return self.icdf(jrd.uniform(key, shape=sample_shape + self.batch_shape))

    @validate_sample
    def log_prob(self, value):
        def logp_neg1(value):
            logp_neg1_val = jnp.add(jnp.log(value), jnp.log(self.high))
            logp_neg1_val = jnp.subtract(jnp.log(self.low), logp_neg1_val)
            return logp_neg1_val

        def logp(value):
            log_value = jnp.log(value)
            logp = jnp.multiply(self.alpha, log_value)
            beta = jnp.add(1.0, self.alpha)
            logp = jnp.add(
                logp,
                jnp.log(
                    jnp.divide(
                        beta,
                        jnp.subtract(
                            jnp.power(self.high, beta), jnp.power(self.low, beta)
                        ),
                    )
                ),
            )
            return logp

        return jnp.where(jnp.equal(self.alpha, -1.0), logp_neg1(value), logp(value))

    def cdf(self, value):
        beta = jnp.add(1.0, self.alpha)
        cdf = jnp.atleast_1d(value**beta - self.low**beta) / (
            self.high**beta - self.low**beta
        )
        cdf_neg1 = (jnp.log(value) - jnp.log(self.low)) / (
            jnp.log(self.high) - jnp.log(self.low)
        )
        cdf = jnp.where(jnp.equal(self.alpha, -1.0), cdf_neg1, cdf)
        cdf = jnp.minimum(cdf, 1.0)
        cdf = jnp.maximum(cdf, 0.0)
        return cdf

    def icdf(self, q):
        beta = jnp.add(1.0, self.alpha)
        low_pow_beta = jnp.power(self.low, beta)
        high_pow_beta = jnp.power(self.high, beta)
        icdf = jnp.multiply(q, jnp.subtract(high_pow_beta, low_pow_beta))
        icdf = jnp.add(low_pow_beta, icdf)
        icdf = jnp.power(icdf, jnp.reciprocal(beta))

        icdf_neg1 = jnp.divide(self.high, self.low)
        icdf_neg1 = jnp.log(icdf_neg1)
        icdf_neg1 = jnp.multiply(q, icdf_neg1)
        icdf_neg1 = jnp.exp(icdf_neg1)
        icdf_neg1 = jnp.multiply(self.low, icdf_neg1)
        return jnp.where(jnp.equal(self.alpha, -1.0), icdf_neg1, icdf)


class Wysocki2019MassModel(Distribution):
    r"""It is a double side truncated power law distribution, as described in
    equation 7 of the `Reconstructing phenomenological distributions of compact
    binaries via gravitational wave observations <https://arxiv.org/abs/1805.06442>`_.

    .. math::
        p(m_1,m_2\mid\alpha,m_{\text{min}},m_{\text{max}},M_{\text{max}})\propto
        \frac{m_1^{-\alpha}}{m_1-m_{\text{min}}}
    """

    arg_constraints = {
        "alpha_m": constraints.real,
        "mmin": constraints.dependent,
        "mmax": constraints.dependent,
    }
    reparametrized_params = ["alpha_m", "mmin", "mmax"]
    pytree_aux_fields = ("_support",)

    def __init__(self, alpha_m, mmin, mmax) -> None:
        r"""
        :param alpha_m: index of the power law distribution
        :param mmin: lower mass limit
        :param mmax: upper mass limit
        """
        self.alpha_m, self.mmin, self.mmax = promote_shapes(alpha_m, mmin, mmax)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha_m),
            jnp.shape(mmin),
            jnp.shape(mmax),
        )
        self._support = mass_sandwich(mmin, mmax)
        super(Wysocki2019MassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        m2 = value[..., 1]
        log_prob_m1 = TruncatedPowerLaw(
            alpha=jnp.negative(self.alpha_m),
            low=self.mmin,
            high=self.mmax,
            validate_args=True,
        ).log_prob(m1)
        log_prob_m2_given_m1 = uniform.logpdf(
            m2, loc=self.mmin, scale=jnp.subtract(m1, self.mmin)
        )
        return jnp.add(log_prob_m1, log_prob_m2_given_m1)

    def sample(self, key, sample_shape=()) -> Array:
        m1 = TruncatedPowerLaw(
            alpha=jnp.negative(self.alpha_m),
            low=self.mmin,
            high=self.mmax,
            validate_args=True,
        ).sample(key=key, sample_shape=sample_shape + self.batch_shape)
        key = jrd.split(key)[1]
        m2 = jrd.uniform(
            key, shape=sample_shape + self.batch_shape, minval=self.mmin, maxval=m1
        )
        return jnp.column_stack((m1, m2))


def MultiSpinModel(
    alpha_m,
    beta_m,
    mmin,
    mmax,
    mean_chi_1_pl,
    var_chi_1_pl,
    mean_chi_2_pl,
    var_chi_2_pl,
    std_dev_title_1_pl,
    std_dev_title_2_pl,
    mean_m1,
    std_dev_m1,
    mean_m2,
    std_dev_m2,
    mean_chi_1_g,
    var_chi_1_g,
    mean_chi_2_g,
    var_chi_2_g,
    std_dev_title_1_g,
    std_dev_title_2_g,
) -> MixtureGeneral:
    r"""See details Appendix D3 of `Population Properties of Compact Objects from
    the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog <https://iopscience.iop.org/article/10.3847/2041-8213/abe949>`_.

    .. note::
        In original formulation, each component is scaled up by their respective rate
        :math:`\mathcal{R}_{pl}` and :math:`\mathcal{R}_{g}`. However to streamline
        the implementation with :mod:`numpyro`, we are not scaling each component with
        their respective rate. Instead, we will deal this in Log-likelihood function.


    :param alpha_m: Power-law slope of the primary mass distribution for the low-mass
        subpopulation
    :param beta_m: Power-law slope of the mass ratio distribution for the low-mass
        subpopulation
    :param mmin: Minimum mass of the primary mass distribution for the low-mass
        subpopulation
    :param mmax: Maximum mass of the primary mass distribution for the low-mass
        subpopulation
    :param mean_chi_1_pl: Mean of the beta distribution of primary spin magnitudes for
        the low-mass subpopulation
    :param var_chi_1_pl: Variance of the beta distribution of primary spin magnitudes
        for the low-mass subpopulation
    :param mean_chi_2_pl: Mean of the beta distribution of secondary spin magnitudes
        for the low-mass subpopulation
    :param var_chi_2_pl: Variance of the beta distribution of secondary spin
        magnitudes for the low-mass subpopulation
    :param std_dev_title_1_pl: Width of the Truncated Gaussian distribution of the
        cosine of the primary spin-tilt angle for the low-mass subpopulation
    :param std_dev_title_2_pl: Width of the Truncated Gaussian distribution of cos(
        secondary spin-tilt angle) for the low-mass subpopulation
    :param mean_m1: Centroid of the primary mass distribution for the high-mass
        subpopulation
    :param std_dev_m1: Width of the primary mass distribution for the high-mass
        subpopulation
    :param mean_m2: Centroid of the secondary mass distribution for the high-mass
        subpopulation
    :param std_dev_m2: Width of the secondary mass distribution for the high-mass
        subpopulation
    :param mean_chi_1_g: Mean of the beta distribution of primary spin magnitudes for
        the high-mass subpopulation
    :param var_chi_1_g: Variance of the beta distribution of primary spin magnitudes
        for the high-mass subpopulation
    :param mean_chi_2_g: Width of the Truncated Gaussian distribution of cos(primary
        spin-tilt angle) for the high-mass subpopulation
    :param var_chi_2_g: Mean of the beta distribution of secondary spin magnitudes for
        the high-mass subpopulation
    :param std_dev_title_1_g: Variance of the beta distribution of secondary spin
        magnitudes for the high-mass subpopulation
    :param std_dev_title_2_g: Width of the Truncated Gaussian distribution of cos(
        secondary spin-tilt angle) for the high-mass subpopulation
    """
    alpha_chi_1_pl, beta_chi_1_pl = beta_dist_mean_variance_to_concentrations(
        mean_chi_1_pl, var_chi_1_pl
    )
    alpha_chi_2_pl, beta_chi_2_pl = beta_dist_mean_variance_to_concentrations(
        mean_chi_2_pl, var_chi_2_pl
    )
    alpha_chi_1_g, beta_chi_1_g = beta_dist_mean_variance_to_concentrations(
        mean_chi_1_g, var_chi_1_g
    )
    alpha_chi_2_g, beta_chi_2_g = beta_dist_mean_variance_to_concentrations(
        mean_chi_2_g, var_chi_2_g
    )

    ######################
    # POWERLAW COMPONENT #
    ######################

    powerlaw = PowerLawPrimaryMassRatio(
        alpha=alpha_m, beta=beta_m, mmin=mmin, mmax=mmax
    )
    chi1_dist_pl = Beta(
        concentration1=alpha_chi_1_pl,
        concentration0=beta_chi_1_pl,
        validate_args=True,
    )
    chi2_dist_pl = Beta(
        concentration1=alpha_chi_2_pl,
        concentration0=beta_chi_2_pl,
        validate_args=True,
    )
    cos_tilt1_dist_pl = TruncatedNormal(
        loc=1,
        scale=std_dev_title_1_pl,
        low=-1,
        high=1,
        validate_args=True,
    )
    cos_tilt2_dist_pl = TruncatedNormal(
        loc=1,
        scale=std_dev_title_2_pl,
        low=-1,
        high=1,
        validate_args=True,
    )

    powerlaw_component = JointDistribution(
        powerlaw, chi1_dist_pl, chi2_dist_pl, cos_tilt1_dist_pl, cos_tilt2_dist_pl
    )

    ######################
    # GAUSSIAN COMPONENT #
    ######################

    m1_dist_g = Normal(loc=mean_m1, scale=std_dev_m1, validate_args=True)
    m2_dist_g = Normal(loc=mean_m2, scale=std_dev_m2, validate_args=True)
    chi1_dist_g = Beta(
        concentration1=alpha_chi_1_g,
        concentration0=beta_chi_1_g,
        validate_args=True,
    )
    chi2_dist_g = Beta(
        concentration1=alpha_chi_2_g,
        concentration0=beta_chi_2_g,
        validate_args=True,
    )
    cos_tilt1_dist_g = TruncatedNormal(
        loc=1,
        scale=std_dev_title_1_g,
        low=-1,
        high=1,
        validate_args=True,
    )
    cos_tilt2_dist_g = TruncatedNormal(
        loc=1,
        scale=std_dev_title_2_g,
        low=-1,
        high=1,
        validate_args=True,
    )

    gaussian_component = JointDistribution(
        m1_dist_g,
        m2_dist_g,
        chi1_dist_g,
        chi2_dist_g,
        cos_tilt1_dist_g,
        cos_tilt2_dist_g,
    )

    return MixtureGeneral(
        mixing_distribution=Categorical(probs=jnp.array([0.5, 0.5])),
        component_distributions=[powerlaw_component, gaussian_component],
        support=constraints.real,
        validate_args=True,
    )
