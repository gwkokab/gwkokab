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
from typing_extensions import Optional

import chex
import jax
from jax import lax, numpy as jnp, random as jrd, tree as jtr, vmap
from jax.nn import softplus
from jax.scipy.special import expit, logsumexp
from jax.scipy.stats import truncnorm, uniform
from jaxtyping import Array, Int, Real
from numpyro.distributions import (
    CategoricalProbs,
    constraints,
    Distribution,
    MixtureGeneral,
    MultivariateNormal,
    Normal,
    TransformedDistribution,
    TruncatedNormal,
    Uniform,
)
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key

from ..utils.transformations import m1_q_to_m2
from .constraints import mass_ratio_mass_sandwich, mass_sandwich, unique_intervals
from .transformations import PrimaryMassAndMassRatioToComponentMassesTransform
from .utils import (
    doubly_truncated_powerlaw_icdf,
    doubly_truncated_powerlaw_log_prob,
    get_default_spin_magnitude_dists,
    get_spin_magnitude_and_misalignment_dist,
    get_spin_misalignment_dist,
    JointDistribution,
    numerical_inverse_transform_sampling,
)


__all__ = [
    "BrokenPowerLawMassModel",
    "FlexibleMixtureModel",
    "GaussianSpinModel",
    "IndependentSpinOrientationGaussianIsotropic",
    "MassGapModel",
    "MultiPeakMassModel",
    "MultiSourceModel",
    "MultiSpinModel",
    "NDistribution",
    "NPowerLawMGaussian",
    "NPowerLawMGaussianWithDefaultSpinMagnitude",
    "NPowerLawMGaussianWithDefaultSpinMagnitudeAndSpinMisalignment",
    "NPowerLawMGaussianWithSpinMisalignment",
    "PowerLawPeakMassModel",
    "PowerLawPrimaryMassRatio",
    "TruncatedPowerLaw",
    "Wysocki2019MassModel",
]


class _BaseSmoothedMassDistribution(Distribution):
    def __init__(self, *, batch_shape, event_shape):
        super(_BaseSmoothedMassDistribution, self).__init__(
            batch_shape, event_shape, validate_args=True
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        return mass_ratio_mass_sandwich(self.mmin, self.mmax)

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1_val = self.log_prob_m1(m1)
        log_prob_q_val = self.log_prob_q(m1, q)
        return jnp.add(log_prob_m1_val, log_prob_q_val)

    def log_prob_m1(self, m1):
        log_prob_m_val = self.__class__.log_primary_model(self, m1)
        log_prob_m_val = jnp.add(log_prob_m_val, jnp.log(self.smoothing_kernel(m1)))
        log_norm = self.log_norm_p_m1()
        log_prob_m_val = jnp.subtract(log_prob_m_val, log_norm)
        return log_prob_m_val

    def log_prob_q(self, m1, q):
        log_prob_q_val = doubly_truncated_powerlaw_log_prob(
            q,
            self.beta_q,
            jnp.divide(self.mmin, m1),
            1.0,
        )
        log_prob_q_val = jnp.add(
            log_prob_q_val,
            jnp.log(self.smoothing_kernel(m1_q_to_m2(m1=m1, q=q))),
        )
        log_norm = self.log_norm_p_q()
        log_prob_q_val = jnp.subtract(log_prob_q_val, log_norm)
        return log_prob_q_val

    def log_norm_p_m1(self):
        m1s = jrd.uniform(
            jrd.PRNGKey(0), shape=(1000,), minval=self.mmin, maxval=self.mmax
        )
        log_prob_m_val = self.__class__.log_primary_model(self, m1s)
        log_prob_m_val = jnp.add(
            log_prob_m_val,
            jnp.log(self.smoothing_kernel(m1s)),
        )
        log_norm = (
            logsumexp(log_prob_m_val)
            - jnp.log(m1s.shape[0])
            + jnp.log(jnp.prod(self.mmax - self.mmin))
        )
        return log_norm

    def log_norm_p_q(self):
        m1s = jrd.uniform(
            jrd.PRNGKey(0),
            shape=(1000,) + self.batch_shape,
            minval=self.mmin,
            maxval=self.mmax,
        )
        qs = jrd.uniform(
            jrd.PRNGKey(1), shape=(1000,) + self.batch_shape, minval=0.001, maxval=1
        )
        log_prob_q_val = doubly_truncated_powerlaw_log_prob(
            qs, self.beta_q, jnp.divide(self.mmin, m1s), 1.0
        )
        log_prob_q_val = jnp.nan_to_num(log_prob_q_val, nan=-jnp.inf)
        log_prob_q_val = jnp.add(
            log_prob_q_val,
            jnp.log(self.smoothing_kernel(m1_q_to_m2(m1=m1s, q=qs))),
        )
        log_norm = (
            logsumexp(log_prob_q_val) - jnp.log(m1s.shape[0]) + jnp.log(1.0 - 0.001)
        )
        return log_norm

    def smoothing_kernel(self, mass: Array | Real) -> Array | Real:
        r"""See equation B4 in `Population Properties of Compact Objects from the
        Second LIGO-Virgo Gravitational-Wave Transient Catalog 
        <https://arxiv.org/abs/2010.14533>`_.
        
        .. math::
            S(m\mid m_{\min}, \delta) = \begin{cases}
                0 & \text{if } m < m_{\min}, \\
                \left[\displaystyle 1 + \exp\left(\frac{\delta}{m}
                +\frac{\delta}{m-\delta}\right)\right]^{-1}
                & \text{if } m_{\min} \leq m < m_{\min} + \delta, \\
                1 & \text{if } m \geq m_{\min} + \delta
            \end{cases}
    
        :param mass: mass of the primary black hole
        :param mass_min: minimum mass of the primary black hole
        :param delta: small mass difference
        :return: smoothing kernel value
        """
        mass_min_shifted = jnp.add(self.mmin, self.delta_m)

        shifted_mass = jnp.nan_to_num(
            jnp.divide(jnp.subtract(mass, self.mmin), self.delta_m), nan=0.0
        )
        shifted_mass = jnp.clip(shifted_mass, 1e-6, 1.0 - 1e-6)
        neg_exponent = jnp.subtract(
            jnp.reciprocal(jnp.subtract(1.0, shifted_mass)),
            jnp.reciprocal(shifted_mass),
        )
        window = expit(neg_exponent)
        conditions = [
            jnp.less(mass, self.mmin),
            jnp.logical_and(
                jnp.less_equal(self.mmin, mass),
                jnp.less_equal(mass, mass_min_shifted),
            ),
            jnp.greater(mass, mass_min_shifted),
        ]
        choices = [jnp.zeros(mass.shape), window, jnp.ones(mass.shape)]
        return jnp.select(conditions, choices, default=jnp.zeros(mass.shape))

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        flattened_sample_shape = jtr.reduce(lambda x, y: x * y, sample_shape, 1)

        m1 = numerical_inverse_transform_sampling(
            logpdf=self.log_prob_m1,
            limits=(self.mmin, self.mmax),
            sample_shape=(flattened_sample_shape,),
            key=key,
            batch_shape=self.batch_shape,
            n_grid_points=1000,
        )

        key = jrd.split(key, m1.shape)

        q = vmap(
            lambda _m1, _k: numerical_inverse_transform_sampling(
                logpdf=partial(self.log_prob_q, _m1),
                limits=(jnp.divide(self.mmin, _m1), 1.0),
                sample_shape=(),
                key=_k,
                batch_shape=self.batch_shape,
                n_grid_points=1000,
            )
        )(m1, key)

        return jnp.column_stack([m1, q]).reshape(sample_shape + self.event_shape)


class BrokenPowerLawMassModel(_BaseSmoothedMassDistribution):
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
        "break_fraction": constraints.unit_interval,
        "delta_m": constraints.positive,
    }
    reparametrized_params = [
        "alpha1",
        "alpha2",
        "beta_q",
        "mmin",
        "mmax",
        "break_fraction",
        "delta_m",
    ]
    pytree_data_fields = (
        "alpha1",
        "alpha2",
        "beta_q",
        "mmin",
        "mmax",
        "break_fraction",
        "delta_m",
    )

    def __init__(self, alpha1, alpha2, beta_q, mmin, mmax, break_fraction, delta_m):
        r"""
        :param alpha1: Power-law index for first component of primary mass model
        :param alpha2: Power-law index for second component of primary mass
            model
        :param beta_q: Power-law index for mass ratio model
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mbreak: Break mass
        :param delta: Smoothing parameter
        """
        (
            self.alpha1,
            self.alpha2,
            self.beta_q,
            self.mmin,
            self.mmax,
            self.break_fraction,
            self.delta_m,
        ) = promote_shapes(alpha1, alpha2, beta_q, mmin, mmax, break_fraction, delta_m)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha1),
            jnp.shape(alpha2),
            jnp.shape(beta_q),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(break_fraction),
            jnp.shape(delta_m),
        )
        super(BrokenPowerLawMassModel, self).__init__(
            batch_shape=batch_shape, event_shape=(2,)
        )

    def log_primary_model(self, mass):
        mbreak = jnp.add(
            self.mmin,
            jnp.multiply(self.break_fraction, jnp.subtract(self.mmax, self.mmin)),
        )

        log_correction = doubly_truncated_powerlaw_log_prob(
            value=mbreak, alpha=-self.alpha2, low=mbreak, high=self.mmax
        ) - doubly_truncated_powerlaw_log_prob(
            value=mbreak, alpha=-self.alpha1, low=self.mmin, high=mbreak
        )

        log_low_part = doubly_truncated_powerlaw_log_prob(
            value=mass, alpha=-self.alpha1, low=self.mmin, high=mbreak
        )

        log_high_part = doubly_truncated_powerlaw_log_prob(
            value=mass, alpha=-self.alpha2, low=mbreak, high=self.mmax
        )

        log_prob_val = (log_low_part + log_correction) * jnp.less(
            mass, mbreak
        ) + log_high_part * jnp.greater_equal(mass, mbreak)

        return log_prob_val - jax.nn.softplus(log_correction)


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
        mixing_distribution=CategoricalProbs(probs=mixing_probs, validate_args=True),
        component_distributions=[component_0_dist, component_1_dist],
        support=constraints.real,
        validate_args=True,
    )


class MultiPeakMassModel(_BaseSmoothedMassDistribution):
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
        "beta_q": constraints.real,
        "lam": constraints.unit_interval,
        "lam1": constraints.unit_interval,
        "mmin": constraints.dependent,
        "mmax": constraints.dependent,
        "delta_m": constraints.positive,
        "mu1": constraints.positive,
        "sigma1": constraints.positive,
        "mu2": constraints.positive,
        "sigma2": constraints.positive,
    }
    reparametrized_params = [
        "alpha",
        "beta_q",
        "lam",
        "lam1",
        "delta_m",
        "mmin",
        "mmax",
        "mu1",
        "sigma1",
        "mu2",
        "sigma2",
    ]
    pytree_data_fields = (
        "alpha",
        "beta_q",
        "lam",
        "lam1",
        "delta_m",
        "mmin",
        "mmax",
        "mu1",
        "sigma1",
        "mu2",
        "sigma2",
    )

    def __init__(
        self, alpha, beta_q, lam, lam1, delta_m, mmin, mmax, mu1, sigma1, mu2, sigma2
    ):
        r"""
        :param alpha: Power-law index for primary mass model
        :param beta_q: Power-law index for mass ratio model
        :param lam: weight for power-law component
        :param lam1: weight for first Gaussian component
        :param delta_m: Smoothing parameter
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mu1: Mean of first Gaussian component
        :param sigma1: Standard deviation of first Gaussian component
        :param mu2: Mean of second Gaussian component
        :param sigma2: Standard deviation of second Gaussian component
        """
        (
            self.alpha,
            self.beta_q,
            self.lam,
            self.lam1,
            self.delta_m,
            self.mmin,
            self.mmax,
            self.mu1,
            self.sigma1,
            self.mu2,
            self.sigma2,
        ) = promote_shapes(
            alpha, beta_q, lam, lam1, delta_m, mmin, mmax, mu1, sigma1, mu2, sigma2
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta_q),
            jnp.shape(lam),
            jnp.shape(lam1),
            jnp.shape(delta_m),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mu1),
            jnp.shape(sigma1),
            jnp.shape(mu2),
            jnp.shape(sigma2),
        )
        super(MultiPeakMassModel, self).__init__(
            batch_shape=batch_shape, event_shape=(2,)
        )

    def log_primary_model(self, m1):
        gaussian_term_1 = jnp.add(jnp.log(self.lam), jnp.log(self.lam1))
        gaussian_term_1 = jnp.add(
            gaussian_term_1,
            truncnorm.logpdf(
                m1,
                a=(self.mmin - self.mu1) / self.sigma1,
                b=(self.mmax - self.mu1) / self.sigma1,
                loc=self.mu1,
                scale=self.sigma1,
            ),
        )

        gaussian_term_2 = jnp.add(jnp.log(self.lam), jnp.log1p(-self.lam1))
        gaussian_term_2 = jnp.add(
            gaussian_term_2,
            truncnorm.logpdf(
                m1,
                a=(self.mmin - self.mu2) / self.sigma2,
                b=(self.mmax - self.mu2) / self.sigma2,
                loc=self.mu2,
                scale=self.sigma2,
            ),
        )

        powerlaw_term = jnp.add(
            jnp.log1p(-self.lam),
            doubly_truncated_powerlaw_log_prob(
                value=m1, alpha=-self.alpha, low=self.mmin, high=self.mmax
            ),
        )
        log_prob_val = jnp.logaddexp(powerlaw_term, gaussian_term_1)
        log_prob_val = jnp.add(log_prob_val, jnp.exp(gaussian_term_2))
        return log_prob_val


def NDistribution(distribution: Distribution, n: Int, **params) -> MixtureGeneral:
    """Mixture of any :math:`n` distributions.

    :param distribution: distribution to mix
    :param n: number of components
    :return: Mixture of :math:`n` distributions
    """
    arg_names = distribution.arg_constraints.keys()
    mixing_dist = CategoricalProbs(probs=jnp.divide(jnp.ones(n), n), validate_args=True)
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


class PowerLawPeakMassModel(_BaseSmoothedMassDistribution):
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
        "beta_q": constraints.real,
        "lam": constraints.unit_interval,
        "delta_m": constraints.real,
        "mmin": constraints.dependent,
        "mmax": constraints.dependent,
        "mu": constraints.real,
        "sigma": constraints.positive,
    }
    reparametrized_params = [
        "alpha",
        "beta_q",
        "lam",
        "delta_m",
        "mmin",
        "mmax",
        "mu",
        "sigma",
    ]
    pytree_data_fields = (
        "alpha",
        "beta_q",
        "lam",
        "delta_m",
        "mmin",
        "mmax",
        "mu",
        "sigma",
    )

    def __init__(self, alpha, beta_q, lam, delta_m, mmin, mmax, mu, sigma) -> None:
        r"""
        :param alpha: Power-law index for primary mass model
        :param beta_q: Power-law index for mass ratio model
        :param lam: Fraction of Gaussian component
        :param delta_m: Smoothing parameter
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param mu: Mean of Gaussian component
        :param sigma: Standard deviation of Gaussian component
        """
        (
            self.alpha,
            self.beta_q,
            self.lam,
            self.delta_m,
            self.mmin,
            self.mmax,
            self.mu,
            self.sigma,
        ) = promote_shapes(alpha, beta_q, lam, delta_m, mmin, mmax, mu, sigma)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta_q),
            jnp.shape(lam),
            jnp.shape(delta_m),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(mu),
            jnp.shape(sigma),
        )
        super(PowerLawPeakMassModel, self).__init__(
            batch_shape=batch_shape, event_shape=(2,)
        )

    def log_primary_model(self, m1):
        gaussian_term = jnp.add(
            jnp.log(self.lam),
            truncnorm.logpdf(
                m1,
                a=(self.mmin - self.mu) / self.sigma,
                b=(self.mmax - self.mu) / self.sigma,
                loc=self.mu,
                scale=self.sigma,
            ),
        )
        powerlaw_term = jnp.add(
            jnp.log1p(-self.lam),
            doubly_truncated_powerlaw_log_prob(
                value=m1, alpha=-self.alpha, low=self.mmin, high=self.mmax
            ),
        )
        log_prob_val = jnp.logaddexp(powerlaw_term, gaussian_term)
        return log_prob_val


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
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = doubly_truncated_powerlaw_log_prob(
            value=m1,
            alpha=self.alpha,
            low=self.mmin,
            high=self.mmax,
        )
        log_prob_q = doubly_truncated_powerlaw_log_prob(
            value=q,
            alpha=self.beta,
            low=jnp.divide(self.mmin, m1),
            high=1.0,
        )
        return jnp.add(log_prob_m1, log_prob_q)

    def sample(self, key, sample_shape=()):
        u = jrd.uniform(key, shape=(2,) + sample_shape + self.batch_shape)
        u1 = u[0]
        u2 = u[1]
        m1 = doubly_truncated_powerlaw_icdf(
            q=u1,
            alpha=self.alpha,
            low=self.mmin,
            high=self.mmax,
        )
        q = doubly_truncated_powerlaw_icdf(
            q=u2,
            alpha=self.beta,
            low=jnp.divide(self.mmin, m1),
            high=1.0,
        )
        return jnp.column_stack((m1, q))


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
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        m2 = value[..., 1]
        log_prob_m1 = doubly_truncated_powerlaw_log_prob(
            value=m1, alpha=jnp.negative(self.alpha_m), low=self.mmin, high=self.mmax
        )
        log_prob_m2_given_m1 = uniform.logpdf(
            m2, loc=self.mmin, scale=jnp.subtract(m1, self.mmin)
        )
        return jnp.add(log_prob_m1, log_prob_m2_given_m1)

    def sample(self, key, sample_shape=()) -> Array:
        u = jrd.uniform(key, shape=sample_shape + self.batch_shape)
        m1 = doubly_truncated_powerlaw_icdf(
            q=u, alpha=jnp.negative(self.alpha_m), low=self.mmin, high=self.mmax
        )
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
    ######################
    # POWERLAW COMPONENT #
    ######################

    powerlaw = TransformedDistribution(
        base_distribution=PowerLawPrimaryMassRatio(
            alpha=alpha_m, beta=beta_m, mmin=mmin, mmax=mmax
        ),
        transforms=[PrimaryMassAndMassRatioToComponentMassesTransform()],
        validate_args=True,
    )
    powerlaw_component = JointDistribution(
        powerlaw,
        *get_spin_magnitude_and_misalignment_dist(
            mean_chi1=mean_chi_1_pl,
            variance_chi1=var_chi_1_pl,
            mean_chi2=mean_chi_2_pl,
            variance_chi2=var_chi_2_pl,
            mean_tilt_1=1.0,
            std_dev_tilt_1=std_dev_title_1_pl,
            mean_tilt_2=1.0,
            std_dev_tilt_2=std_dev_title_2_pl,
        ),
    )

    ######################
    # GAUSSIAN COMPONENT #
    ######################

    m1_dist_g = Normal(loc=mean_m1, scale=std_dev_m1, validate_args=True)
    m2_dist_g = Normal(loc=mean_m2, scale=std_dev_m2, validate_args=True)
    gaussian_component = JointDistribution(
        m1_dist_g,
        m2_dist_g,
        *get_spin_magnitude_and_misalignment_dist(
            mean_chi1=mean_chi_1_g,
            variance_chi1=var_chi_1_g,
            mean_chi2=mean_chi_2_g,
            variance_chi2=var_chi_2_g,
            mean_tilt_1=1.0,
            std_dev_tilt_1=std_dev_title_1_g,
            mean_tilt_2=1.0,
            std_dev_tilt_2=std_dev_title_2_g,
        ),
    )

    return MixtureGeneral(
        mixing_distribution=CategoricalProbs(
            probs=jnp.ones(2) / 2.0, validate_args=True
        ),
        component_distributions=[powerlaw_component, gaussian_component],
        support=constraints.real_vector,
        validate_args=True,
    )


def MultiSourceModel(
    # BBH Powerlaw params
    alpha_BBH,
    beta_BBH,
    mmin_BBH,
    mmax_BBH,
    mean_chi_1_pl_BBH,
    var_chi_1_pl_BBH,
    mean_chi_2_pl_BBH,
    var_chi_2_pl_BBH,
    std_dev_title_1_pl_BBH,
    std_dev_title_2_pl_BBH,
    # BBH Gaussian params
    mean_m1_BBH,
    std_dev_m1_BBH,
    mean_m2_BBH,
    std_dev_m2_BBH,
    mean_chi_1_g_BBH,
    var_chi_1_g_BBH,
    mean_chi_2_g_BBH,
    var_chi_2_g_BBH,
    std_dev_title_1_g_BBH,
    std_dev_title_2_g_BBH,
    # BHNS params
    mean_m_BHNS,
    std_dev_m_BHNS,
    mean_chi_BH_BHNS,
    var_chi_BH_BHNS,
    mean_chi_NS_BHNS,
    var_chi_NS_BHNS,
    std_dev_title_BH_BHNS,
    std_dev_title_NS_BHNS,
    # BNS params
    mean_m_NS,
    std_dev_m_NS,
    mmax_NS,
    mean_chi_1_BNS,
    var_chi_1_BNS,
    mean_chi_2_BNS,
    var_chi_2_BNS,
    std_dev_title_1_BNS,
    std_dev_title_2_BNS,
) -> MixtureGeneral:
    r"""See details in Appendix D3 of `Population of Merging Compact Binaries Inferred
    Using Gravitational Waves through
    GWTC-3 <https://doi.org/10.1103/PhysRevX.13.011048>`_.

    .. note::
        In original formulation, each component is scaled up by their respective rate
        :math:`\mathcal{R}_{\mathrm{BBH},pl}`, :math:`\mathcal{R}_{\mathrm{BBH},g}`,
        :math:`\mathcal{R}_{\mathrm{NSBH}}` and :math:`\mathcal{R}_{\mathrm{BNS}}`.
        However to streamline the implementation with :mod:`NumPyro`, we are not
        scaling each component with their respective rate. Instead, we will deal this
        in Log-likelihood function.


    :param alpha_BBH: Primary mass spectral index for the BBH power-law subpopulation
    :param beta_BBH: Mass ratio spectral index for the BBH power-law subpopulation
    :param mmin_BBH: Minimum mass of the BBH power-law subpopulation
    :param mmax_BBH: Maximum mass of the BBH power-law subpopulation
    :param mean_chi_1_pl_BBH: Mean of the beta distribution of primary spin
        magnitudes for the BBH Gaussian subpopulation
    :param var_chi_1_pl_BBH: Variance of the beta distribution of primary spin
        magnitudes for the BBH Gaussian subpopulation
    :param mean_chi_2_pl_BBH: Mean of the beta distribution of secondary spin
        magnitudes for the BBH Gaussian subpopulation
    :param var_chi_2_pl_BBH: Variance of the beta distribution of secondary spin
        magnitudes for the BBH Gaussian subpopulation
    :param std_dev_title_1_pl_BBH: Width of truncated Gaussian, determining typical
        primary spin misalignment for the BBH Gaussian subpopulation
    :param std_dev_title_2_pl_BBH: Width of truncated Gaussian, determining typical
        secondary spin misalignment for the BBH Gaussian subpopulation
    :param mean_m1_BBH: Centroid of the primary mass distribution for the BBH
        Gaussian subpopulation
    :param std_dev_m1_BBH: Width of the primary mass distribution for the BBH
        Gaussian subpopulation
    :param mean_m2_BBH: Centroid of the secondary mass distribution for the BBH
        Gaussian subpopulation
    :param std_dev_m2_BBH: Width of the secondary mass distribution for the BBH
        Gaussian subpopulation
    :param mean_chi_1_g_BBH: Mean of the beta distribution of primary spin magnitudes
        for the BBH Gaussian subpopulation
    :param var_chi_1_g_BBH: Variance of the beta distribution of primary spin
        magnitudes for the BBH Gaussian subpopulation
    :param mean_chi_2_g_BBH: Mean of the beta distribution of secondary spin
        magnitudes for the BBH Gaussian subpopulation
    :param var_chi_2_g_BBH: Variance of the beta distribution of secondary spin
        magnitudes for the BBH Gaussian subpopulation
    :param std_dev_title_1_g_BBH: Width of truncated Gaussian determining typical
        primary spin misalignment for the BBH Gaussian subpopulation
    :param std_dev_title_2_g_BBH: Width of truncated Gaussian determining typical
        secondary spin misalignment for the BBH Gaussian subpopulation
    :param mean_m_BHNS: Centroid of the BH mass distribution for the NSBH
    :param std_dev_m_BHNS: Width of the BH mass distribution for the NSBH
    :param mean_chi_BH_BHNS: Mean of the beta distribution of spin magnitudes for the
        BH in the NSBH subpopulation
    :param var_chi_BH_BHNS: Variance of the beta distribution of spin magnitudes for
        the BH in the NSBH subpopulation
    :param mean_chi_NS_BHNS: Mean of the beta distribution of spin magnitudes for the
        NS in the NSBH subpopulation
    :param var_chi_NS_BHNS: Variance of the beta distribution of spin magnitudes for
        the NS in the NSBH subpopulation
    :param std_dev_title_BH_BHNS: Width of truncated Gaussian determining typical
        primary (secondary) spin misalignment for the BH in the NSBH subpopulation
    :param std_dev_title_NS_BHNS: Width of truncated Gaussian determining typical
        primary (secondary) spin misalignment for the NS in the NSBH subpopulation
    :param mean_m_NS: Centroid of the NS mass distribution
    :param std_dev_m_NS: Width of the NS mass distribution
    :param mmax_NS: Maximum mass of all NSs
    :param mean_chi_1_BNS: Mean of the beta distribution of primary spin magnitudes in
        the BNS subpopulation
    :param var_chi_1_BNS: Variance of the beta distribution of primary spin magnitudes
        in the BNS subpopulation
    :param mean_chi_2_BNS: Mean of the beta distribution of secondary spin magnitudes
        in the BNS subpopulation
    :param var_chi_2_BNS: Variance of the beta distribution of secondary spin magnitudes
        in the BNS subpopulation
    :param std_dev_title_1_BNS: Width of truncated Gaussian determining typical primary
        spin misalignment in the BNS subpopulation
    :param std_dev_title_2_BNS: Width of truncated Gaussian determining typical
        secondary spin misalignment in the BNS subpopulation
    """
    ##########################
    # BBH POWERLAW COMPONENT #
    ##########################

    powerlaw_BBH = TransformedDistribution(
        base_distribution=PowerLawPrimaryMassRatio(
            alpha=alpha_BBH, beta=beta_BBH, mmin=mmin_BBH, mmax=mmax_BBH
        ),
        transforms=[PrimaryMassAndMassRatioToComponentMassesTransform()],
        validate_args=True,
    )
    powerlaw_BBH_component = JointDistribution(
        powerlaw_BBH,
        *get_spin_magnitude_and_misalignment_dist(
            mean_chi1=mean_chi_1_pl_BBH,
            variance_chi1=var_chi_1_pl_BBH,
            mean_chi2=mean_chi_2_pl_BBH,
            variance_chi2=var_chi_2_pl_BBH,
            mean_tilt_1=1.0,
            std_dev_tilt_1=std_dev_title_1_pl_BBH,
            mean_tilt_2=1.0,
            std_dev_tilt_2=std_dev_title_2_pl_BBH,
        ),
    )

    ##########################
    # BBH GAUSSIAN COMPONENT #
    ##########################

    m1_dist_g_BBH = Normal(loc=mean_m1_BBH, scale=std_dev_m1_BBH, validate_args=True)
    m2_dist_g_BBH = Normal(loc=mean_m2_BBH, scale=std_dev_m2_BBH, validate_args=True)
    gaussian_BBH_component = JointDistribution(
        m1_dist_g_BBH,
        m2_dist_g_BBH,
        *get_spin_magnitude_and_misalignment_dist(
            mean_chi1=mean_chi_1_g_BBH,
            variance_chi1=var_chi_1_g_BBH,
            mean_chi2=mean_chi_2_g_BBH,
            variance_chi2=var_chi_2_g_BBH,
            mean_tilt_1=1.0,
            std_dev_tilt_1=std_dev_title_1_g_BBH,
            mean_tilt_2=1.0,
            std_dev_tilt_2=std_dev_title_2_g_BBH,
        ),
    )

    ###########################
    # BHNS GAUSSIAN COMPONENT #
    ###########################

    m_dist_BH_BHNS = Normal(loc=mean_m_BHNS, scale=std_dev_m_BHNS, validate_args=True)
    m_dist_NS_BHNS = TruncatedNormal(
        loc=mean_m_NS, scale=std_dev_m_NS, high=mmax_NS, validate_args=True
    )
    BHNS_component = JointDistribution(
        m_dist_BH_BHNS,
        m_dist_NS_BHNS,
        *get_spin_magnitude_and_misalignment_dist(
            mean_chi1=mean_chi_BH_BHNS,
            variance_chi1=var_chi_BH_BHNS,
            mean_chi2=mean_chi_NS_BHNS,
            variance_chi2=var_chi_NS_BHNS,
            mean_tilt_1=1.0,
            std_dev_tilt_1=std_dev_title_BH_BHNS,
            mean_tilt_2=1.0,
            std_dev_tilt_2=std_dev_title_NS_BHNS,
        ),
    )

    #########################
    # NS GAUSSIAN COMPONENT #
    #########################

    BNS_component = JointDistribution(
        m_dist_NS_BHNS,
        m_dist_NS_BHNS,
        *get_spin_magnitude_and_misalignment_dist(
            mean_chi1=mean_chi_1_BNS,
            variance_chi1=var_chi_1_BNS,
            mean_chi2=mean_chi_2_BNS,
            variance_chi2=var_chi_2_BNS,
            mean_tilt_1=1.0,
            std_dev_tilt_1=std_dev_title_1_BNS,
            mean_tilt_2=1.0,
            std_dev_tilt_2=std_dev_title_2_BNS,
        ),
    )

    return MixtureGeneral(
        mixing_distribution=CategoricalProbs(
            probs=jnp.ones(4) / 4.0, validate_args=True
        ),
        component_distributions=[
            powerlaw_BBH_component,
            gaussian_BBH_component,
            BHNS_component,
            BNS_component,
        ],
        support=constraints.real_vector,
        validate_args=True,
    )


def NPowerLawMGaussianWithDefaultSpinMagnitudeAndSpinMisalignment(
    N_pl, N_g, **params
) -> MixtureGeneral:
    r"""Mixture of N power-law and M Gaussians.

    :param N_pl: number of power-law components
    :param N_g: number of Gaussian components
    :param alpha_i: Power law index for primary mass for ith component, where
        :math:`0\leq i < N_{pl}`
    :param beta_i: Power law index for mass ratio for ith component, where
        :math:`0\leq i < N_{pl}`
    :param mmin_i: Minimum mass for ith component, where :math:`0\leq i < N_{pl}`
    :param mmax_i: Maximum mass for ith component, where :math:`0\leq i < N_{pl}`
    :param mean_chi1_pl_i: Mean of the beta distribution of primary spin magnitudes for
        ith component, where :math:`0\leq i < N_{pl}`
    :param variance_chi1_pl_i: Variance of the beta distribution of primary spin
        magnitudes for ith component, where :math:`0\leq i < N_{pl}`
    :param mean_chi2_pl_i: Mean of the beta distribution of secondary spin magnitudes
        for ith component, where :math:`0\leq i < N_{pl}`
    :param variance_chi2_pl_i: Variance of the beta distribution of secondary spin
        magnitudes for ith component, where :math:`0\leq i < N_{pl}`
    :param std_dev_tilt1_pl_i: Standard deviation of the tilt distribution of primary
        tilt for ith component, where :math:`0\leq i < N_{pl}`
    :param std_dev_tilt2_pl_i: Standard deviation of the tilt distribution of secondary
        tilt for ith component, where :math:`0\leq i < N_{pl}`
    :param loc_m1_i: Mean of the primary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param scale_m1_i: Width of the primary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param loc_m2_i: Mean of the secondary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param scale_m2_i: Width of the secondary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param mean_chi1_g_i: Mean of the beta distribution of primary spin magnitudes for
        ith component, where :math:`0\leq i < N_{g}`
    :param variance_chi1_g_i: Variance of the beta distribution of primary spin
        magnitudes for ith component, where :math:`0\leq i < N_{g}`
    :param mean_chi2_g_i: Mean of the beta distribution of secondary spin magnitudes for
        ith component, where :math:`0\leq i < N_{g}`
    :param variance_chi2_g_i: Variance of the beta distribution of secondary spin
        magnitudes for ith component, where :math:`0\leq i < N_{g}`
    :param std_dev_tilt1_g_i: Variance of the beta distribution of primary spin
        magnitudes for ith component, where :math:`0\leq i < N_{g}`
    :param std_dev_tilt2_g_i: Variance of the beta distribution of secondary spin
        magnitudes for ith component, where :math:`0\leq i < N_{g}`
    :param alpha: default value for :code:`alpha`
    :param beta: default value for :code:`beta`
    :param mmin: default value for :code:`mmin`
    :param mmax: default value for :code:`mmax`
    :param mean_chi1_pl: default value for :code:`mean_chi1_pl`
    :param variance_chi1_pl: default value for :code:`variance_chi1_pl`
    :param mean_chi2_pl: default value for :code:`mean_chi2_pl`
    :param variance_chi2_pl: default value for :code:`variance_chi2_pl`
    :param std_dev_tilt1_pl: default value for :code:`std_dev_tilt1_pl`
    :param std_dev_tilt2_pl: default value for :code:`std_dev_tilt2_pl`
    :param loc_m1: default value for :code:`loc_m1`
    :param scale_m1: default value for :code:`scale_m1`
    :param loc_m2: default value for :code:`loc_m2`
    :param scale_m2: default value for :code:`scale_m2`
    :param mean_chi1_g: default value for :code:`mean_chi1_g`
    :param variance_chi1_g: default value for :code:`variance_chi1_g`
    :param mean_chi2_g: default value for :code:`mean_chi2_g`
    :param variance_chi2_g: default value for :code:`variance_chi2_g`
    :param std_dev_tilt1_g: default value for :code:`std_dev_tilt1_g`
    :param std_dev_tilt2_g: default value for :code:`std_dev_tilt2_g`
    :return: Mixture of N power-law and M Gaussians
    """
    if N_pl > 0:
        pl_arg_names = PowerLawPrimaryMassRatio.arg_constraints.keys()
        pl_args_per_component = [
            {arg: params.get(f"{arg}_{i}", params.get(arg)) for arg in pl_arg_names}
            for i in range(N_pl)
        ]
        powerlaws = jtr.map(
            lambda x: TransformedDistribution(
                base_distribution=PowerLawPrimaryMassRatio(**x),
                transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            ),
            pl_args_per_component,
            is_leaf=lambda x: isinstance(x, dict),
        )
        chis_pl = [
            get_default_spin_magnitude_dists(
                mean_chi1=params.get(f"mean_chi1_pl_{i}", params.get("mean_chi1_pl")),
                variance_chi1=params.get(
                    f"variance_chi1_pl_{i}", params.get("variance_chi1_pl")
                ),
                mean_chi2=params.get(f"mean_chi2_pl_{i}", params.get("mean_chi2_pl")),
                variance_chi2=params.get(
                    f"variance_chi2_pl_{i}", params.get("variance_chi2_pl")
                ),
            )
            for i in range(N_pl)
        ]
        tilts_pl = [
            get_spin_misalignment_dist(
                mean_tilt_1=1.0,
                std_dev_tilt_1=params.get(
                    f"std_dev_tilt1_pl_{i}", params.get("std_dev_tilt1_pl")
                ),
                mean_tilt_2=1.0,
                std_dev_tilt_2=params.get(
                    f"std_dev_tilt2_pl_{i}", params.get("std_dev_tilt2_pl")
                ),
            )
            for i in range(N_pl)
        ]
        pl_component_dist = jtr.map(
            lambda pl, chis, tilts: JointDistribution(pl, *chis, *tilts),
            powerlaws,
            chis_pl,
            tilts_pl,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

    if N_g > 0:
        g_args_per_component = [
            {
                "loc": jnp.array(
                    [
                        params.get(f"loc_m1_{i}", params.get("loc_m1")),
                        params.get(f"loc_m2_{i}", params.get("loc_m2")),
                    ]
                ),
                "covariance_matrix": jnp.array(
                    [
                        [
                            jnp.square(
                                params.get(f"scale_m1_{i}", params.get("scale_m1"))
                            ),
                            0.0,
                        ],
                        [
                            0.0,
                            jnp.square(
                                params.get(f"scale_m2_{i}", params.get("scale_m2"))
                            ),
                        ],
                    ]
                ),
            }
            for i in range(N_g)
        ]
        gaussians = jtr.map(
            lambda x: MultivariateNormal(
                loc=x["loc"],
                covariance_matrix=x["covariance_matrix"],
                validate_args=True,
            ),
            g_args_per_component,
            is_leaf=lambda x: isinstance(x, dict),
        )
        chis_g = [
            get_default_spin_magnitude_dists(
                mean_chi1=params.get(f"mean_chi1_g_{i}", params.get("mean_chi1_g")),
                variance_chi1=params.get(
                    f"variance_chi1_g_{i}", params.get("variance_chi1_g")
                ),
                mean_chi2=params.get(f"mean_chi2_g_{i}", params.get("mean_chi2_g")),
                variance_chi2=params.get(
                    f"variance_chi2_g_{i}", params.get("variance_chi2_g")
                ),
            )
            for i in range(N_g)
        ]
        tilts_g = [
            get_spin_misalignment_dist(
                mean_tilt_1=1.0,
                std_dev_tilt_1=params.get(
                    f"std_dev_tilt1_g_{i}", params.get("std_dev_tilt1_g")
                ),
                mean_tilt_2=1.0,
                std_dev_tilt_2=params.get(
                    f"std_dev_tilt2_g_{i}", params.get("std_dev_tilt2_g")
                ),
            )
            for i in range(N_g)
        ]
        g_component_dist = jtr.map(
            lambda g, chis, tilts: JointDistribution(g, *chis, *tilts),
            gaussians,
            chis_g,
            tilts_g,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

    if N_pl == 0 and N_g != 0:
        component_dists = g_component_dist
    elif N_g == 0 and N_pl != 0:
        component_dists = pl_component_dist
    else:
        component_dists = pl_component_dist + g_component_dist

    N = N_pl + N_g
    mixing_dist = CategoricalProbs(probs=jnp.divide(jnp.ones(N), N), validate_args=True)

    return MixtureGeneral(
        mixing_dist,
        component_dists,
        support=constraints.real_vector,
        validate_args=True,
    )


def NPowerLawMGaussianWithDefaultSpinMagnitude(N_pl, N_g, **params) -> MixtureGeneral:
    r"""Mixture of N power-law and M Gaussians.

    :param N_pl: number of power-law components
    :param N_g: number of Gaussian components
    :param alpha_i: Power law index for primary mass for ith component, where
        :math:`0\leq i < N_{pl}`
    :param beta_i: Power law index for mass ratio for ith component, where
        :math:`0\leq i < N_{pl}`
    :param mmin_i: Minimum mass for ith component, where :math:`0\leq i < N_{pl}`
    :param mmax_i: Maximum mass for ith component, where :math:`0\leq i < N_{pl}`
    :param mean_chi1_pl_i: Mean of the beta distribution of primary spin magnitudes for
        ith component, where :math:`0\leq i < N_{pl}`
    :param variance_chi1_pl_i: Variance of the beta distribution of primary spin
        magnitudes for ith component, where :math:`0\leq i < N_{pl}`
    :param mean_chi2_pl_i: Mean of the beta distribution of secondary spin magnitudes
        for ith component, where :math:`0\leq i < N_{pl}`
    :param variance_chi2_pl_i: Variance of the beta distribution of secondary spin
        magnitudes for ith component, where :math:`0\leq i < N_{pl}`
    :param loc_m1_i: Mean of the primary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param scale_m1_i: Width of the primary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param loc_m2_i: Mean of the secondary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param scale_m2_i: Width of the secondary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param mean_chi1_g_i: Mean of the beta distribution of primary spin magnitudes for
        ith component, where :math:`0\leq i < N_{g}`
    :param variance_chi1_g_i: Variance of the beta distribution of primary spin
        magnitudes for ith component, where :math:`0\leq i < N_{g}`
    :param mean_chi2_g_i: Mean of the beta distribution of secondary spin magnitudes for
        ith component, where :math:`0\leq i < N_{g}`
    :param variance_chi2_g_i: Variance of the beta distribution of secondary spin
        magnitudes for ith component, where :math:`0\leq i < N_{g}`
    :param alpha: default value for :code:`alpha`
    :param beta: default value for :code:`beta`
    :param mmin: default value for :code:`mmin`
    :param mmax: default value for :code:`mmax`
    :param mean_chi1_pl: default value for :code:`mean_chi1_pl`
    :param variance_chi1_pl: default value for :code:`variance_chi1_pl`
    :param mean_chi2_pl: default value for :code:`mean_chi2_pl`
    :param variance_chi2_pl: default value for :code:`variance_chi2_pl`
    :param loc_m1: default value for :code:`loc_m1`
    :param scale_m1: default value for :code:`scale_m1`
    :param loc_m2: default value for :code:`loc_m2`
    :param scale_m2: default value for :code:`scale_m2`
    :param mean_chi1_g: default value for :code:`mean_chi1_g`
    :param variance_chi1_g: default value for :code:`variance_chi1_g`
    :param mean_chi2_g: default value for :code:`mean_chi2_g`
    :param variance_chi2_g: default value for :code:`variance_chi2_g`
    :return: Mixture of N power-law and M Gaussians
    """
    if N_pl > 0:
        pl_arg_names = PowerLawPrimaryMassRatio.arg_constraints.keys()
        pl_args_per_component = [
            {arg: params.get(f"{arg}_{i}", params.get(arg)) for arg in pl_arg_names}
            for i in range(N_pl)
        ]
        powerlaws = jtr.map(
            lambda x: TransformedDistribution(
                base_distribution=PowerLawPrimaryMassRatio(**x),
                transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            ),
            pl_args_per_component,
            is_leaf=lambda x: isinstance(x, dict),
        )

        chis_pl = [
            get_default_spin_magnitude_dists(
                mean_chi1=params.get(f"mean_chi1_pl_{i}", params.get("mean_chi1_pl")),
                variance_chi1=params.get(
                    f"variance_chi1_pl_{i}", params.get("variance_chi1_pl")
                ),
                mean_chi2=params.get(f"mean_chi2_pl_{i}", params.get("mean_chi2_pl")),
                variance_chi2=params.get(
                    f"variance_chi2_pl_{i}", params.get("variance_chi2_pl")
                ),
            )
            for i in range(N_pl)
        ]

        pl_component_dist = jtr.map(
            lambda pl, chis: JointDistribution(pl, *chis),
            powerlaws,
            chis_pl,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

    if N_g > 0:
        g_args_per_component = [
            {
                "loc": jnp.array(
                    [
                        params.get(f"loc_m1_{i}", params.get("loc_m1")),
                        params.get(f"loc_m2_{i}", params.get("loc_m2")),
                    ]
                ),
                "covariance_matrix": jnp.array(
                    [
                        [
                            jnp.square(
                                params.get(f"scale_m1_{i}", params.get("scale_m1"))
                            ),
                            0.0,
                        ],
                        [
                            0.0,
                            jnp.square(
                                params.get(f"scale_m2_{i}", params.get("scale_m2"))
                            ),
                        ],
                    ]
                ),
            }
            for i in range(N_g)
        ]
        gaussians = jtr.map(
            lambda x: MultivariateNormal(
                loc=x["loc"],
                covariance_matrix=x["covariance_matrix"],
                validate_args=True,
            ),
            g_args_per_component,
            is_leaf=lambda x: isinstance(x, dict),
        )

        chis_g = [
            get_default_spin_magnitude_dists(
                mean_chi1=params.get(f"mean_chi1_g_{i}", params.get("mean_chi1_g")),
                variance_chi1=params.get(
                    f"variance_chi1_g_{i}", params.get("variance_chi1_g")
                ),
                mean_chi2=params.get(f"mean_chi2_g_{i}", params.get("mean_chi2_g")),
                variance_chi2=params.get(
                    f"variance_chi2_g_{i}", params.get("variance_chi2_g")
                ),
            )
            for i in range(N_g)
        ]

        g_component_dist = jtr.map(
            lambda g, chis: JointDistribution(g, *chis),
            gaussians,
            chis_g,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

    if N_pl == 0 and N_g != 0:
        component_dists = g_component_dist
    elif N_g == 0 and N_pl != 0:
        component_dists = pl_component_dist
    else:
        component_dists = pl_component_dist + g_component_dist

    N = N_pl + N_g
    mixing_dist = CategoricalProbs(probs=jnp.divide(jnp.ones(N), N), validate_args=True)

    return MixtureGeneral(
        mixing_dist,
        component_dists,
        support=constraints.real_vector,
        validate_args=True,
    )


def NPowerLawMGaussianWithSpinMisalignment(N_pl, N_g, **params) -> MixtureGeneral:
    r"""Mixture of N power-law and M Gaussians.

    :param N_pl: number of power-law components
    :param N_g: number of Gaussian components
    :param alpha_i: Power law index for primary mass for ith component, where
        :math:`0\leq i < N_{pl}`
    :param beta_i: Power law index for mass ratio for ith component, where
        :math:`0\leq i < N_{pl}`
    :param mmin_i: Minimum mass for ith component, where :math:`0\leq i < N_{pl}`
    :param mmax_i: Maximum mass for ith component, where :math:`0\leq i < N_{pl}`
    :param std_dev_tilt1_pl_i: Standard deviation of the tilt distribution of primary
        tilt for ith component, where :math:`0\leq i < N_{pl}`
    :param std_dev_tilt2_pl_i: Standard deviation of the tilt distribution of secondary
        tilt for ith component, where :math:`0\leq i < N_{pl}`
    :param loc_m1_i: Mean of the primary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param scale_m1_i: Width of the primary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param loc_m2_i: Mean of the secondary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param scale_m2_i: Width of the secondary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param std_dev_tilt1_g_i: Variance of the beta distribution of primary spin
        magnitudes for ith component, where :math:`0\leq i < N_{g}`
    :param std_dev_tilt2_g_i: Variance of the beta distribution of secondary spin
        magnitudes for ith component, where :math:`0\leq i < N_{g}`
    :param alpha: default value for :code:`alpha`
    :param beta: default value for :code:`beta`
    :param mmin: default value for :code:`mmin`
    :param mmax: default value for :code:`mmax`
    :param std_dev_tilt1_pl: default value for :code:`std_dev_tilt1_pl`
    :param std_dev_tilt2_pl: default value for :code:`std_dev_tilt2_pl`
    :param loc_m1: default value for :code:`loc_m1`
    :param scale_m1: default value for :code:`scale_m1`
    :param loc_m2: default value for :code:`loc_m2`
    :param scale_m2: default value for :code:`scale_m2`
    :param std_dev_tilt1_g: default value for :code:`std_dev_tilt1_g`
    :param std_dev_tilt2_g: default value for :code:`std_dev_tilt2_g`
    :return: Mixture of N power-law and M Gaussians
    """
    if N_pl > 0:
        pl_arg_names = PowerLawPrimaryMassRatio.arg_constraints.keys()
        pl_args_per_component = [
            {arg: params.get(f"{arg}_{i}", params.get(arg)) for arg in pl_arg_names}
            for i in range(N_pl)
        ]
        powerlaws = jtr.map(
            lambda x: TransformedDistribution(
                base_distribution=PowerLawPrimaryMassRatio(**x),
                transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            ),
            pl_args_per_component,
            is_leaf=lambda x: isinstance(x, dict),
        )
        tilts_pl = [
            get_spin_misalignment_dist(
                mean_tilt_1=1.0,
                std_dev_tilt_1=params.get(
                    f"std_dev_tilt1_pl_{i}", params.get("std_dev_tilt1_pl")
                ),
                mean_tilt_2=1.0,
                std_dev_tilt_2=params.get(
                    f"std_dev_tilt2_pl_{i}", params.get("std_dev_tilt2_pl")
                ),
            )
            for i in range(N_pl)
        ]
        pl_component_dist = jtr.map(
            lambda pl, tilts: JointDistribution(pl, *tilts),
            powerlaws,
            tilts_pl,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

    if N_g > 0:
        g_args_per_component = [
            {
                "loc": jnp.array(
                    [
                        params.get(f"loc_m1_{i}", params.get("loc_m1")),
                        params.get(f"loc_m2_{i}", params.get("loc_m2")),
                    ]
                ),
                "covariance_matrix": jnp.array(
                    [
                        [
                            jnp.square(
                                params.get(f"scale_m1_{i}", params.get("scale_m1"))
                            ),
                            0.0,
                        ],
                        [
                            0.0,
                            jnp.square(
                                params.get(f"scale_m2_{i}", params.get("scale_m2"))
                            ),
                        ],
                    ]
                ),
            }
            for i in range(N_g)
        ]
        gaussians = jtr.map(
            lambda x: MultivariateNormal(
                loc=x["loc"],
                covariance_matrix=x["covariance_matrix"],
                validate_args=True,
            ),
            g_args_per_component,
            is_leaf=lambda x: isinstance(x, dict),
        )

        tilts_g = [
            get_spin_misalignment_dist(
                mean_tilt_1=1.0,
                std_dev_tilt_1=params.get(
                    f"std_dev_tilt1_g_{i}", params.get("std_dev_tilt1_g")
                ),
                mean_tilt_2=1.0,
                std_dev_tilt_2=params.get(
                    f"std_dev_tilt2_g_{i}", params.get("std_dev_tilt2_g")
                ),
            )
            for i in range(N_g)
        ]
        g_component_dist = jtr.map(
            lambda g, tilts: JointDistribution(g, *tilts),
            gaussians,
            tilts_g,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

    if N_pl == 0 and N_g != 0:
        component_dists = g_component_dist
    elif N_g == 0 and N_pl != 0:
        component_dists = pl_component_dist
    else:
        component_dists = pl_component_dist + g_component_dist

    N = N_pl + N_g
    mixing_dist = CategoricalProbs(probs=jnp.divide(jnp.ones(N), N), validate_args=True)

    return MixtureGeneral(
        mixing_dist,
        component_dists,
        support=constraints.real_vector,
        validate_args=True,
    )


def NPowerLawMGaussian(N_pl, N_g, **params) -> MixtureGeneral:
    r"""Mixture of N power-law and M Gaussians.

    :param N_pl: number of power-law components
    :param N_g: number of Gaussian components
    :param alpha_i: Power law index for primary mass for ith component, where
        :math:`0\leq i < N_{pl}`
    :param beta_i: Power law index for mass ratio for ith component, where
        :math:`0\leq i < N_{pl}`
    :param mmin_i: Minimum mass for ith component, where :math:`0\leq i < N_{pl}`
    :param mmax_i: Maximum mass for ith component, where :math:`0\leq i < N_{pl}`
    :param loc_m1_i: Mean of the primary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param scale_m1_i: Width of the primary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param loc_m2_i: Mean of the secondary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param scale_m2_i: Width of the secondary mass distribution for ith component, where
        :math:`0\leq i < N_{g}`
    :param alpha: default value for :code:`alpha`
    :param beta: default value for :code:`beta`
    :param mmin: default value for :code:`mmin`
    :param mmax: default value for :code:`mmax`
    :param loc_m1: default value for :code:`loc_m1`
    :param scale_m1: default value for :code:`scale_m1`
    :param loc_m2: default value for :code:`loc_m2`
    :param scale_m2: default value for :code:`scale_m2`
    :return: Mixture of N power-law and M Gaussians
    """
    if N_pl > 0:
        pl_arg_names = PowerLawPrimaryMassRatio.arg_constraints.keys()
        pl_args_per_component = [
            {arg: params.get(f"{arg}_{i}", params.get(arg)) for arg in pl_arg_names}
            for i in range(N_pl)
        ]
        pl_component_dist = jtr.map(
            lambda x: TransformedDistribution(
                base_distribution=PowerLawPrimaryMassRatio(**x),
                transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            ),
            pl_args_per_component,
            is_leaf=lambda x: isinstance(x, dict),
        )

    if N_g > 0:
        g_args_per_component = [
            {
                "loc": jnp.array(
                    [
                        params.get(f"loc_m1_{i}", params.get("loc_m1")),
                        params.get(f"loc_m2_{i}", params.get("loc_m2")),
                    ]
                ),
                "covariance_matrix": jnp.array(
                    [
                        [
                            jnp.square(
                                params.get(f"scale_m1_{i}", params.get("scale_m1"))
                            ),
                            0.0,
                        ],
                        [
                            0.0,
                            jnp.square(
                                params.get(f"scale_m2_{i}", params.get("scale_m2"))
                            ),
                        ],
                    ]
                ),
            }
            for i in range(N_g)
        ]
        g_component_dist = jtr.map(
            lambda x: MultivariateNormal(
                loc=x["loc"],
                covariance_matrix=x["covariance_matrix"],
                validate_args=True,
            ),
            g_args_per_component,
            is_leaf=lambda x: isinstance(x, dict),
        )

    if N_pl == 0 and N_g != 0:
        component_dists = g_component_dist
    elif N_g == 0 and N_pl != 0:
        component_dists = pl_component_dist
    else:
        component_dists = pl_component_dist + g_component_dist

    N = N_pl + N_g
    mixing_dist = CategoricalProbs(probs=jnp.divide(jnp.ones(N), N), validate_args=True)

    return MixtureGeneral(
        mixing_dist,
        component_dists,
        support=constraints.real_vector,
        validate_args=True,
    )


class MassGapModel(Distribution):
    r"""See Eq. (2) of `No evidence for a dip in the binary black hole mass spectrum
    <http://arxiv.org/abs/2406.11111>`_.

    :param alpha: Spectral-index of power-law component
    :param lam: Fraction of masses in Gaussian peaks
    :param lam1: Fraction of peak-masses in lower-mass peak
    :param mu1: Location of lower-mass peak
    :param sigma1: Width of lower-mass peak
    :param mu2: Location of upper-mass peak
    :param sigma2: Width of upper-mass peak
    :param gamma_low: Lower-edge location of gap
    :param gamma_high: Upper-edge location of gap
    :param eta_low: Sharpness of gap's lower-edge
    :param eta_high: Sharpness of gap's upper-edge
    :param depth_of_gap: Depth of gap
    :param mmin: Maximum allowed mass
    :param mmax: Minimum allowed mass
    :param delta_m: Length of minimum-mass roll-off
    :param beta_q: Power-law index of pairing function
    """

    arg_constraints = {
        "alpha": constraints.real,
        "lam": constraints.unit_interval,
        "lam1": constraints.unit_interval,
        "mu1": constraints.positive,
        "sigma1": constraints.positive,
        "mu2": constraints.positive,
        "sigma2": constraints.positive,
        "gamma_low": constraints.real,
        "gamma_high": constraints.real,
        "eta_low": constraints.real,
        "eta_high": constraints.real,
        "depth_of_gap": constraints.unit_interval,
        "mmin": constraints.dependent,
        "mmax": constraints.dependent,
        "delta_m": constraints.positive,
        "beta_q": constraints.real,
    }
    reparametrized_params = [
        "alpha",
        "lam",
        "lam1",
        "mu1",
        "sigma1",
        "mu2",
        "sigma2",
        "gamma_low",
        "gamma_high",
        "eta_low",
        "eta_high",
        "depth_of_gap",
        "mmin",
        "mmax",
        "delta_m",
        "beta_q",
    ]

    def __init__(
        self,
        alpha,
        lam,
        lam1,
        mu1,
        sigma1,
        mu2,
        sigma2,
        gamma_low,
        gamma_high,
        eta_low,
        eta_high,
        depth_of_gap,
        mmin,
        mmax,
        delta_m,
        beta_q,
    ):
        (
            self.alpha,
            self.lam,
            self.lam1,
            self.mu1,
            self.sigma1,
            self.mu2,
            self.sigma2,
            self.gamma_low,
            self.gamma_high,
            self.eta_low,
            self.eta_high,
            self.depth_of_gap,
            self.mmin,
            self.mmax,
            self.delta_m,
            self.beta_q,
        ) = promote_shapes(
            alpha,
            lam,
            lam1,
            mu1,
            sigma1,
            mu2,
            sigma2,
            gamma_low,
            gamma_high,
            eta_low,
            eta_high,
            depth_of_gap,
            mmin,
            mmax,
            delta_m,
            beta_q,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(lam),
            jnp.shape(lam1),
            jnp.shape(mu1),
            jnp.shape(sigma1),
            jnp.shape(mu2),
            jnp.shape(sigma2),
            jnp.shape(gamma_low),
            jnp.shape(gamma_high),
            jnp.shape(eta_low),
            jnp.shape(eta_high),
            jnp.shape(depth_of_gap),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(delta_m),
            jnp.shape(beta_q),
        )
        super(MassGapModel, self).__init__(
            batch_shape, event_shape=(2,), validate_args=True
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return mass_sandwich(self.mmin, self.mmax)

    def log_notch_filter(self, mass):
        log_m = jnp.log(mass)
        log_gamma_low = jnp.log(self.gamma_low)
        log_gamma_high = jnp.log(self.gamma_high)
        notch_filter_val = (
            jnp.log(self.depth_of_gap)
            - softplus(self.eta_low * (log_m - log_gamma_low))
            - softplus(self.eta_high * (log_gamma_high - log_m))
        )
        return jnp.log(-jnp.expm1(notch_filter_val))

    def smoothing_kernel(self, mass: Array | Real) -> Array | Real:
        r"""See equation B4 in `Population Properties of Compact Objects from the
        Second LIGO-Virgo Gravitational-Wave Transient Catalog 
        <https://arxiv.org/abs/2010.14533>`_.
        
        .. math::
            S(m\mid m_{\min}, \delta) = \begin{cases}
                0 & \text{if } m < m_{\min}, \\
                \left[\displaystyle 1 + \exp\left(\frac{\delta}{m}
                +\frac{\delta}{m-\delta}\right)\right]^{-1}
                & \text{if } m_{\min} \leq m < m_{\min} + \delta, \\
                1 & \text{if } m \geq m_{\min} + \delta
            \end{cases}
    
        :param mass: mass of the primary black hole
        :return: smoothing kernel value
        """
        mass_min_shifted = jnp.add(self.mmin, self.delta_m)

        shifted_mass = jnp.nan_to_num(
            jnp.divide(jnp.subtract(mass, self.mmin), self.delta_m), nan=0.0
        )
        shifted_mass = jnp.clip(shifted_mass, 1e-6, 1.0 - 1e-6)
        neg_exponent = jnp.subtract(
            jnp.reciprocal(jnp.subtract(1.0, shifted_mass)),
            jnp.reciprocal(shifted_mass),
        )
        window = expit(neg_exponent)
        conditions = [
            jnp.less(mass, self.mmin),
            jnp.logical_and(
                jnp.less_equal(self.mmin, mass),
                jnp.less_equal(mass, mass_min_shifted),
            ),
            jnp.greater(mass, mass_min_shifted),
        ]
        choices = [jnp.zeros(mass.shape), window, jnp.ones(mass.shape)]
        return jnp.select(conditions, choices, default=jnp.zeros(mass.shape))

    def log_prob_three_component_single(self, m_i):
        component_probs = jnp.stack(
            [
                doubly_truncated_powerlaw_log_prob(
                    value=m_i, alpha=-self.alpha, low=self.mmin, high=self.mmax
                ),
                truncnorm.logpdf(
                    m_i,
                    a=(self.mmin - self.mu1) / self.sigma1,
                    b=(self.mmax - self.mu1) / self.sigma1,
                    loc=self.mu1,
                    scale=self.sigma1,
                ),
                truncnorm.logpdf(
                    m_i,
                    a=(self.mmin - self.mu2) / self.sigma2,
                    b=(self.mmax - self.mu2) / self.sigma2,
                    loc=self.mu2,
                    scale=self.sigma2,
                ),
            ],
            axis=-1,
        )
        mixing_probs = jnp.array(
            [
                jnp.log1p(-self.lam),
                jnp.log(self.lam) + jnp.log(self.lam1),
                jnp.log(self.lam) + jnp.log1p(-self.lam1),
            ]
        )
        log_prob_val = logsumexp(component_probs + mixing_probs)
        return log_prob_val

    def log_prob_mi(self, m_i):
        log_prob_mi_val = self.log_prob_three_component_single(m_i=m_i)
        log_prob_mi_val += self.log_notch_filter(m_i)
        log_prob_mi_val += jnp.log(self.smoothing_kernel(m_i))
        log_prob_mi_val -= self.log_norm_mi()
        return log_prob_mi_val

    def log_norm_mi(self):
        m_i = jrd.uniform(
            jrd.PRNGKey(0), shape=(10000,), minval=self.mmin, maxval=self.mmax
        )
        log_prob_mi_val = self.log_prob_three_component_single(m_i=m_i)
        log_prob_mi_val += self.log_notch_filter(m_i)
        log_prob_mi_val += jnp.log(self.smoothing_kernel(m_i))
        return (
            logsumexp(log_prob_mi_val)
            - jnp.log(m_i.shape[0])
            + jnp.log(jnp.prod(self.mmax - self.mmin))
        )

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        m2 = value[..., 1]
        log_prob_val = self.log_prob_mi(m1)
        log_prob_val += self.log_prob_mi(m2)
        log_prob_val += doubly_truncated_powerlaw_log_prob(
            value=jnp.divide(m2, m1),
            alpha=self.beta_q,
            low=jnp.divide(self.mmin, self.mmax),
            high=1.0,
        )
        return log_prob_val


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
        "low": constraints.dependent,
        "high": constraints.dependent,
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
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value):
        return doubly_truncated_powerlaw_log_prob(
            value=value, alpha=self.alpha, low=self.low, high=self.high
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        u = jrd.uniform(key, shape=sample_shape + self.batch_shape + self.event_shape)
        return doubly_truncated_powerlaw_icdf(
            u, alpha=self.alpha, low=self.low, high=self.high
        )


def FlexibleMixtureModel(
    N: int,
    weights: Array,
    mu_Mc: Optional[Real] = None,
    sigma_Mc: Optional[Real] = None,
    mu_sz: Optional[Real] = None,
    sigma_sz: Optional[Real] = None,
    alpha_q: Optional[Real] = None,
    q_min: Optional[Real] = None,
    **params,
) -> MixtureGeneral:
    r"""Eq. (B9) of `Population of Merging Compact Binaries Inferred Using
    Gravitational Waves through GWTC-3 <https://doi.org/10.1103/PhysRevX.13.011048>`_.

    .. math::
        p(\mathcal{M},q,s_{1z},s_{2z}\mid\lambda) = \sum_{i=1}^{N} w_i
        \mathcal{N}(\mathcal{M}\mid \mu^\mathcal{M}_i,\sigma^\mathcal{M}_i)
        \mathcal{P}(q\mid \alpha^q_i,q^{\min}_i,1)
        \mathcal{N}(s_{1z}\mid \mu^{s_z}_i,\sigma^{s_z}_i)
        \mathcal{N}(s_{2z}\mid \mu^{s_z}_i,\sigma^{s_z}_i)

    where :math:`w_i` is the weight for the ith component and :math:`p_i(\theta)` is the

    :param N: Number of components
    :param weights: weights for each component
    :param mu_Mc_i: ith value for :code:`mu_Mc`
    :param sigma_Mc_i: ith value for :code:`sigma_Mc`
    :param mu_sz_i: ith value for :code:`mu_sz`
    :param sigma_sz_i: ith value for :code:`sigma_sz`
    :param alpha_q_i: ith value for :code:`alpha_q`
    :param q_min_i: ith value for :code:`q_min`
    :param mu_Mc: default value for :code:`mu_Mc`
    :param sigma_Mc: default value for :code:`sigma_Mc`
    :param mu_sz: default value for :code:`mu_sz`
    :param sigma_sz: default value for :code:`sigma_sz`
    :param alpha_q: default value for :code:`alpha_q`
    :param q_min: default value for :code:`q_min`
    """
    chex.assert_axis_dimension(weights, 0, N)
    ranges = list(range(N))
    mu_Mc_list = jtr.map(lambda i: params.get(f"mu_Mc_{i}", mu_Mc), ranges)
    sigma_Mc_list = jtr.map(lambda i: params.get(f"sigma_Mc_{i}", sigma_Mc), ranges)
    mu_sz_list = jtr.map(lambda i: params.get(f"mu_sz_{i}", mu_sz), ranges)
    sigma_sz_list = jtr.map(lambda i: params.get(f"sigma_sz_{i}", sigma_sz), ranges)
    alpha_q_list = jtr.map(lambda i: params.get(f"alpha_q_{i}", alpha_q), ranges)
    q_min_list = jtr.map(lambda i: params.get(f"q_min_{i}", q_min), ranges)

    component_dists = jtr.map(
        lambda mu_Mc_i,
        sigma_Mc_i,
        mu_sz_i,
        sigma_sz_i,
        alpha_q_i,
        q_min_i: JointDistribution(
            Normal(loc=mu_Mc_i, scale=sigma_Mc_i, validate_args=True),
            TruncatedPowerLaw(alpha=alpha_q_i, low=q_min_i, high=1.0),
            Normal(loc=mu_sz_i, scale=sigma_sz_i, validate_args=True),
            Normal(loc=mu_sz_i, scale=sigma_sz_i, validate_args=True),
        ),
        mu_Mc_list,
        sigma_Mc_list,
        mu_sz_list,
        sigma_sz_list,
        alpha_q_list,
        q_min_list,
    )

    return MixtureGeneral(
        mixing_distribution=CategoricalProbs(probs=weights, validate_args=True),
        component_distributions=component_dists,
        support=unique_intervals((0.0, 0.0, -1.0, -1.0), (jnp.inf, 1.0, 1.0, 1.0)),
        validate_args=True,
    )
