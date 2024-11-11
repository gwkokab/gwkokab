# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from typing_extensions import Optional

import chex
from jax import lax, numpy as jnp, random as jrd, tree as jtr
from jax.nn import softplus
from jax.scipy.special import expit, logsumexp
from jax.scipy.stats import norm, truncnorm, uniform
from jaxtyping import Array, Int, Real
from numpyro.distributions import (
    CategoricalProbs,
    constraints,
    Distribution,
    DoublyTruncatedPowerlaw,
    MixtureGeneral,
    MultivariateNormal,
    Normal,
    TruncatedNormal,
    Uniform,
)
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils import log_planck_taper_window
from .constraints import mass_ratio_mass_sandwich, mass_sandwich
from .utils import (
    doubly_truncated_power_law_icdf,
    doubly_truncated_power_law_log_prob,
    JointDistribution,
)


__all__ = [
    "FlexibleMixtureModel",
    "GaussianSpinModel",
    "IndependentSpinOrientationGaussianIsotropic",
    "MassGapModel",
    "NDistribution",
    "PowerlawPrimaryMassRatio",
    "SmoothedGaussianPrimaryMassRatio",
    "SmoothedPowerlawPrimaryMassRatio",
    "Wysocki2019MassModel",
]


def GaussianSpinModel(
    mu_eff, sigma_eff, mu_p, sigma_p, rho, *, validate_args=None
) -> MultivariateNormal:
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
        validate_args=validate_args,
    )


def IndependentSpinOrientationGaussianIsotropic(
    zeta, sigma1, sigma2, *, validate_args=None
) -> MixtureGeneral:
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
    component_0_dist = Uniform(low=-1, high=1, validate_args=validate_args)
    component_1_dist = TruncatedNormal(
        loc=1.0,
        scale=jnp.array([sigma1, sigma2]),
        low=-1,
        high=1,
        validate_args=validate_args,
    )
    return MixtureGeneral(
        mixing_distribution=CategoricalProbs(
            probs=mixing_probs, validate_args=validate_args
        ),
        component_distributions=[component_0_dist, component_1_dist],
        support=constraints.real,
        validate_args=validate_args,
    )


def NDistribution(
    distribution: Distribution, n: Int, *, validate_args=None, **params
) -> MixtureGeneral:
    """Mixture of any :math:`n` distributions.

    :param distribution: distribution to mix
    :param n: number of components
    :return: Mixture of :math:`n` distributions
    """
    arg_names = distribution.arg_constraints.keys()
    mixing_dist = CategoricalProbs(
        probs=jnp.divide(jnp.ones(n), n), validate_args=validate_args
    )
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
        validate_args=validate_args,
    )


class PowerlawPrimaryMassRatio(Distribution):
    r"""Power law model for two-dimensional mass distribution, modelling primary mass
    and conditional mass ratio distribution.

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
        "mmin": constraints.positive,
        "mmax": constraints.positive,
    }
    reparametrized_params = ["alpha", "beta", "mmin", "mmax"]
    pytree_aux_fields = ("_support",)

    def __init__(self, alpha, beta, mmin, mmax, *, validate_args=None) -> None:
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
        super(PowerlawPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        log_prob_m1 = doubly_truncated_power_law_log_prob(
            x=m1, alpha=self.alpha, low=self.mmin, high=self.mmax
        )
        # as low approaches to high, mathematically it shoots off to infinity
        # And autograd does not behave nicely around limiting values.
        # These two links provide the solution to the problem.
        # https://github.com/jax-ml/jax/issues/1052#issuecomment-514083352
        # https://github.com/jax-ml/jax/issues/5039#issuecomment-735430180
        mmin_over_m1 = jnp.where(self.mmin < m1, jnp.divide(self.mmin, m1), 0.0)
        log_prob_q = jnp.where(
            self.mmin < m1,
            doubly_truncated_power_law_log_prob(
                x=q, alpha=self.beta, low=mmin_over_m1, high=1.0
            ),
            -jnp.inf,
        )
        return jnp.add(log_prob_m1, log_prob_q)

    def sample(self, key, sample_shape=()):
        key_m1, key_q = jrd.split(key)
        u_m1 = jrd.uniform(key_m1, shape=sample_shape)
        u_q = jrd.uniform(key_q, shape=sample_shape)
        m1 = doubly_truncated_power_law_icdf(
            q=u_m1, alpha=self.alpha, low=self.mmin, high=self.mmax
        )
        q = doubly_truncated_power_law_icdf(
            q=u_q, alpha=self.beta, low=jnp.divide(self.mmin, m1), high=1.0
        )
        return jnp.stack((m1, q), axis=-1)


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
        "mmin": constraints.positive,
        "mmax": constraints.positive,
    }
    reparametrized_params = ["alpha_m", "mmin", "mmax"]
    pytree_aux_fields = ("_support",)

    def __init__(self, alpha_m, mmin, mmax, *, validate_args=None) -> None:
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
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        m2 = value[..., 1]
        log_prob_m1 = doubly_truncated_power_law_log_prob(
            x=m1, alpha=jnp.negative(self.alpha_m), low=self.mmin, high=self.mmax
        )
        log_prob_m2_given_m1 = uniform.logpdf(
            m2, loc=self.mmin, scale=jnp.subtract(m1, self.mmin)
        )
        return jnp.add(log_prob_m1, log_prob_m2_given_m1)

    def sample(self, key, sample_shape=()) -> Array:
        key_m1, key_m2 = jrd.split(key)
        u_m1 = jrd.uniform(key_m1, shape=sample_shape)
        m1 = doubly_truncated_power_law_icdf(
            q=u_m1, alpha=-self.alpha_m, low=self.mmin, high=self.mmax
        )
        m2 = jrd.uniform(key_m2, shape=sample_shape, minval=self.mmin, maxval=m1)
        return jnp.stack((m1, m2), axis=-1)


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
        *,
        validate_args=None,
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
            batch_shape, event_shape=(2,), validate_args=validate_args
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
                doubly_truncated_power_law_log_prob(
                    x=m_i, alpha=-self.alpha, low=self.mmin, high=self.mmax
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
        log_prob_val += doubly_truncated_power_law_log_prob(
            x=jnp.divide(m2, m1),
            alpha=self.beta_q,
            low=jnp.divide(self.mmin, self.mmax),
            high=1.0,
        )
        return log_prob_val


def FlexibleMixtureModel(
    N: int,
    weights: Array,
    mu_Mc: Optional[Real] = None,
    sigma_Mc: Optional[Real] = None,
    mu_sz: Optional[Real] = None,
    sigma_sz: Optional[Real] = None,
    alpha_q: Optional[Real] = None,
    q_min: Optional[Real] = None,
    *,
    validate_args=None,
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
            Normal(loc=mu_Mc_i, scale=sigma_Mc_i, validate_args=validate_args),
            DoublyTruncatedPowerlaw(
                alpha=alpha_q_i, low=q_min_i, high=1.0, validate_args=validate_args
            ),
            Normal(loc=mu_sz_i, scale=sigma_sz_i, validate_args=validate_args),
            Normal(loc=mu_sz_i, scale=sigma_sz_i, validate_args=validate_args),
        ),
        mu_Mc_list,
        sigma_Mc_list,
        mu_sz_list,
        sigma_sz_list,
        alpha_q_list,
        q_min_list,
    )

    return MixtureGeneral(
        mixing_distribution=CategoricalProbs(
            probs=weights, validate_args=validate_args
        ),
        component_distributions=component_dists,
        support=constraints.independent(
            constraints.interval(
                jnp.array([0.0, 0.0, -1.0, -1.0]), jnp.array([jnp.inf, 1.0, 1.0, 1.0])
            ),
            1,
        ),
        validate_args=validate_args,
    )


class SmoothedPowerlawPrimaryMassRatio(Distribution):
    r""":class:`PowerlawPrimaryMassRatio` with smoothing kernel on the lower edge.

    .. math::
        p(m_1,q\mid\alpha,\beta) = p(m_1\mid\alpha)p(q \mid m_1, \beta)

    .. math::
        \begin{align*}
            p(m_1\mid\alpha)&
            \propto m_1^{\alpha}S(m_1\mid m_{\text{min}},\delta),\qquad m_{\text{min}}\leq m_1\leq m_{\max}\\
            p(q\mid m_1,\beta)&
            \propto q^{\beta}S(m_1q\mid m_{\text{min}},\delta),\qquad \frac{m_{\text{min}}}{m_1}\leq q\leq 1
        \end{align*}

    Logarithm of smoothing kernel is :func:`~gwkokab.utils.kernel.log_planck_taper_window`.
    """

    arg_constraints = {
        "alpha": constraints.real,
        "beta": constraints.real,
        "mmin": constraints.positive,
        "mmax": constraints.positive,
        "delta": constraints.positive,
    }
    reparametrized_params = ["alpha", "beta", "mmin", "mmax", "delta"]
    pytree_aux_fields = ("_support",)

    def __init__(self, alpha, beta, mmin, mmax, delta, *, validate_args=None) -> None:
        """
        :param alpha: Power law index for primary mass
        :param beta: Power law index for mass ratio
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param delta: width of the smoothing window
        """
        self.alpha, self.beta, self.mmin, self.mmax, self.delta = promote_shapes(
            alpha, beta, mmin, mmax, delta
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(delta),
        )
        self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(SmoothedPowerlawPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        m2 = jnp.multiply(m1, q)
        log_smoothing_m1 = log_planck_taper_window(x=m1, a=self.mmin, b=self.delta)
        log_smoothing_q = log_planck_taper_window(x=m2, a=self.mmin, b=self.delta)

        log_prob_m1 = doubly_truncated_power_law_log_prob(
            x=m1, alpha=self.alpha, low=self.mmin, high=self.mmax
        )
        # as low approaches to high, mathematically it shoots off to infinity
        # And autograd does not behave nicely around limiting values.
        # These two links provide the solution to the problem.
        # https://github.com/jax-ml/jax/issues/1052#issuecomment-514083352
        # https://github.com/jax-ml/jax/issues/5039#issuecomment-735430180
        mmin_over_m1 = jnp.where(self.mmin < m1, jnp.divide(self.mmin, m1), 0.0)
        log_prob_q = jnp.where(
            self.mmin < m1,
            doubly_truncated_power_law_log_prob(
                x=q, alpha=self.beta, low=mmin_over_m1, high=1.0
            ),
            -jnp.inf,
        )
        return jnp.add(log_prob_m1, log_prob_q) + log_smoothing_m1 + log_smoothing_q


class SmoothedGaussianPrimaryMassRatio(Distribution):
    r""":class:`~numpyro.distributions.continuous.Normal` with smoothing kernel on
    the lower edge.

    .. math::
        p(m_1,q\mid\mu,\sigma^2,\beta) = \mathcal{N}(m_1\mid\mu,\sigma^2)p(q \mid m_1,\beta)

    .. math::
        p(q\mid m_1,\beta) \propto q^{\beta}S(m_1q\mid m_{\text{min}},\delta),\qquad \frac{m_{\text{min}}}{m_1}\leq q\leq 1

    Logarithm of smoothing kernel is :func:`~gwkokab.utils.kernel.log_planck_taper_window`.
    """

    arg_constraints = {
        "loc": constraints.positive,
        "scale": constraints.positive,
        "beta": constraints.real,
        "mmin": constraints.positive,
        "mmax": constraints.positive,
        "delta": constraints.positive,
    }
    reparametrized_params = ["loc", "scale", "beta", "mmin", "mmax", "delta"]
    pytree_aux_fields = ("_support",)

    def __init__(
        self, loc, scale, beta, mmin, mmax, delta, *, validate_args=None
    ) -> None:
        """
        :param loc: mean of the Gaussian distribution
        :param scale: standard deviation of the Gaussian distribution
        :param beta: Power law index for mass ratio
        :param mmin: Minimum mass
        :param mmax: Maximum mass
        :param delta: width of the smoothing window
        """
        self.loc, self.scale, self.beta, self.mmin, self.mmax, self.delta = (
            promote_shapes(loc, scale, beta, mmin, mmax, delta)
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc),
            jnp.shape(scale),
            jnp.shape(beta),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(delta),
        )
        self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(SmoothedGaussianPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        m2 = jnp.multiply(m1, q)
        log_smoothing_m1 = log_planck_taper_window(x=m1, a=self.mmin, b=self.delta)
        log_smoothing_q = log_planck_taper_window(x=m2, a=self.mmin, b=self.delta)

        log_prob_m1 = norm.logpdf(x=m1, loc=self.loc, scale=self.scale)
        # as low approaches to high, mathematically it shoots off to infinity
        # And autograd does not behave nicely around limiting values.
        # These two links provide the solution to the problem.
        # https://github.com/jax-ml/jax/issues/1052#issuecomment-514083352
        # https://github.com/jax-ml/jax/issues/5039#issuecomment-735430180
        mmin_over_m1 = jnp.where(self.mmin < m1, jnp.divide(self.mmin, m1), 0.0)
        log_prob_q = jnp.where(
            self.mmin < m1,
            doubly_truncated_power_law_log_prob(
                x=q, alpha=self.beta, low=mmin_over_m1, high=1.0
            ),
            -jnp.inf,
        )
        return jnp.add(log_prob_m1, log_prob_q) + log_smoothing_m1 + log_smoothing_q
