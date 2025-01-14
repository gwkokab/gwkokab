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


from functools import partial
from typing_extensions import Optional

import chex
import jax
from jax import lax, numpy as jnp, random as jrd, tree as jtr
from jax.nn import softplus
from jax.scipy.special import expit, logsumexp
from jax.scipy.stats import truncnorm, uniform
from jaxtyping import Array, ArrayLike
from numpyro.distributions import (
    CategoricalProbs,
    constraints,
    Distribution,
    DoublyTruncatedPowerLaw,
    MixtureGeneral,
    Normal,
    TruncatedNormal,
)
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils.kernel import log_planck_taper_window
from .constraints import mass_ratio_mass_sandwich, mass_sandwich
from .utils import (
    doubly_truncated_power_law_icdf,
    doubly_truncated_power_law_log_prob,
    JointDistribution,
)


__all__ = [
    "FlexibleMixtureModel",
    "MassGapModel",
    "PowerlawPrimaryMassRatio",
    "SmoothedGaussianPrimaryMassRatio",
    "SmoothedPowerlawPrimaryMassRatio",
    "Wysocki2019MassModel",
]


class PowerlawPrimaryMassRatio(Distribution):
    r"""Power law model for two-dimensional mass distribution, modelling primary mass and
    conditional mass ratio distribution.

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

    def __init__(
        self,
        alpha: ArrayLike,
        beta: ArrayLike,
        mmin: ArrayLike,
        mmax: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        alpha : ArrayLike
            Power law index for primary mass
        beta : ArrayLike
            Power law index for mass ratio
        mmin : ArrayLike
            Minimum mass
        mmax : ArrayLike
            Maximum mass
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
        log_prob_q = jnp.where(
            jnp.less_equal(m1, self.mmin),
            -jnp.inf,
            doubly_truncated_power_law_log_prob(
                x=q, alpha=self.beta, low=self.mmin / m1, high=1.0
            ),
        )
        return log_prob_m1 + log_prob_q

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

    def __init__(
        self,
        alpha_m: ArrayLike,
        mmin: ArrayLike,
        mmax: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        alpha_m : ArrayLike
            index of the power law distribution
        mmin : ArrayLike
            lower mass limit
        mmax : ArrayLike
            upper mass limit
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

    .. warning::
        This model is not rigorously tested and should be used with caution.

    Parameter
    ---------
    alpha : ArrayLike
        Spectral-index of power-law component
    lam : ArrayLike
        Fraction of masses in Gaussian peaks
    lam1 : ArrayLike
        Fraction of peak-masses in lower-mass peak
    mu1 : ArrayLike
        Location of lower-mass peak
    sigma1 : ArrayLike
        Width of lower-mass peak
    mu2 : ArrayLike
        Location of upper-mass peak
    sigma2 : ArrayLike
        Width of upper-mass peak
    gamma_low : ArrayLike
        Lower-edge location of gap
    gamma_high : ArrayLike
        Upper-edge location of gap
    eta_low : ArrayLike
        Sharpness of gap's lower-edge
    eta_high : ArrayLike
        Sharpness of gap's upper-edge
    depth_of_gap : ArrayLike
        Depth of gap
    mmin : ArrayLike
        Maximum allowed mass
    mmax : ArrayLike
        Minimum allowed mass
    delta_m : ArrayLike
        Length of minimum-mass roll-off
    beta_q : ArrayLike
        Power-law index of pairing function
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
        alpha: ArrayLike,
        lam: ArrayLike,
        lam1: ArrayLike,
        mu1: ArrayLike,
        sigma1: ArrayLike,
        mu2: ArrayLike,
        sigma2: ArrayLike,
        gamma_low: ArrayLike,
        gamma_high: ArrayLike,
        eta_low: ArrayLike,
        eta_high: ArrayLike,
        depth_of_gap: ArrayLike,
        mmin: ArrayLike,
        mmax: ArrayLike,
        delta_m: ArrayLike,
        beta_q: ArrayLike,
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

    def smoothing_kernel(self, mass: Array) -> Array:
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
    mu_Mc: Optional[float] = None,
    sigma_Mc: Optional[float] = None,
    mu_sz: Optional[float] = None,
    sigma_sz: Optional[float] = None,
    alpha_q: Optional[float] = None,
    q_min: Optional[float] = None,
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

    .. warning::
        This model is not rigorously tested and should be used with caution.

    Parameters
    ----------

    N : ArrayLike
        Number of components
    weights : ArrayLike
        weights for each component
    mu_Mc_i : ArrayLike
        ith value for :code:`mu_Mc`
    sigma_Mc_i : ArrayLike
        ith value for :code:`sigma_Mc`
    mu_sz_i : ArrayLike
        ith value for :code:`mu_sz`
    sigma_sz_i : ArrayLike
        ith value for :code:`sigma_sz`
    alpha_q_i : ArrayLike
        ith value for :code:`alpha_q`
    q_min_i : ArrayLike
        ith value for :code:`q_min`
    mu_Mc : ArrayLike
        default value for :code:`mu_Mc`
    sigma_Mc : ArrayLike
        default value for :code:`sigma_Mc`
    mu_sz : ArrayLike
        default value for :code:`mu_sz`
    sigma_sz : ArrayLike
        default value for :code:`sigma_sz`
    alpha_q : ArrayLike
        default value for :code:`alpha_q`
    q_min : ArrayLike
        default value for :code:`q_min`
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
            DoublyTruncatedPowerLaw(
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
        p(m_1,q\mid\alpha,\beta,m_{\text{min}},m_{\text{max}},\delta) = p(m_1\mid\alpha,m_{\text{min}},m_{\text{max}},\delta)p(q \mid m_1,\beta,m_{\text{min}},\delta)

    .. math::
        \begin{align*}
            p(m_1\mid\alpha,m_{\text{min}},m_{\text{max}},\delta)&
            \propto m_1^{\alpha}S\left(\frac{m_1 - m_{\text{min}}}{\delta}\right),\qquad m_{\text{min}}\leq m_1\leq m_{\max} \\
            p(q \mid m_1,\beta,m_{\text{min}},\delta)&
            \propto q^{\beta}S\left(\frac{m_1q - m_{\text{min}}}{\delta}\right),\qquad \frac{m_{\text{min}}}{m_1}\leq q\leq 1
        \end{align*}

    Logarithm of smoothing kernel is :func:`~gwkokab.utils.kernel.log_planck_taper_window`.
    """

    arg_constraints = {
        "alpha": constraints.real,
        "beta": constraints.real,
        "mmin": constraints.positive,
        "mmax": constraints.positive,
        "delta": constraints.positive,
        "log_scale": constraints.less_than_eq(0.0),
    }
    reparametrized_params = ["alpha", "beta", "mmin", "mmax", "delta", "log_scale"]
    pytree_aux_fields = ("_support",)

    def __init__(
        self,
        alpha: ArrayLike,
        beta: ArrayLike,
        mmin: ArrayLike,
        mmax: ArrayLike,
        delta: ArrayLike,
        log_scale: ArrayLike = 0.0,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        alpha : ArrayLike
            Power law index for primary mass
        beta : ArrayLike
            Power law index for mass ratio
        mmin : ArrayLike
            Minimum mass
        mmax : ArrayLike
            Maximum mass
        delta : ArrayLike
            width of the smoothing window
        log_scale : ArrayLike
            log of the scaling factor for the distribution
        """
        self.alpha, self.beta, self.mmin, self.mmax, self.delta, self.log_scale = (
            promote_shapes(alpha, beta, mmin, mmax, delta, log_scale)
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(delta),
            jnp.shape(log_scale),
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
        m2 = m1 * q
        log_smoothing_m1 = log_planck_taper_window(
            (m1 - self.mmin) / jnp.where(self.delta == 0.0, 1.0, self.delta)
        )
        log_smoothing_q = log_planck_taper_window(
            (m2 - self.mmin) / jnp.where(self.delta == 0.0, 1.0, self.delta)
        )
        log_prob_m1 = doubly_truncated_power_law_log_prob(
            x=m1, alpha=self.alpha, low=self.mmin, high=self.mmax
        )
        log_prob_q = jnp.where(
            jnp.less_equal(m1, self.mmin),
            -jnp.inf,
            doubly_truncated_power_law_log_prob(
                x=q, alpha=self.beta, low=self.mmin / m1, high=1.0
            ),
        )
        return (
            log_prob_m1
            + log_prob_q
            + log_smoothing_m1
            + log_smoothing_q
            + self.log_scale
        )


class SmoothedGaussianPrimaryMassRatio(Distribution):
    r""":class:`~numpyro.distributions.continuous.Normal` with smoothing kernel on the
    lower edge.

    .. math::
        p(m_1,q\mid\mu,\sigma^2,\beta,m_{\text{min}},m_{\text{max}},\delta) = \mathcal{N}(m_1\mid\mu,\sigma^2)S\left(\frac{m_1 - m_{\text{min}}}{\delta}\right)p(q \mid m_1,\beta,m_{\text{min}},\delta)

    .. math::
        p(q\mid m_1,\beta) \propto q^{\beta}S\left(\frac{m_1q - m_{\text{min}}}{\delta}\right),\qquad \frac{m_{\text{min}}}{m_1}\leq q\leq 1

    Logarithm of smoothing kernel is :func:`~gwkokab.utils.kernel.log_planck_taper_window`.

    .. attention::

        If :code:`low` or :code:`high` are not provided to the `TruncatedNormal`, they
        default to  :math:`-\infty` or :math:`+\infty`, respectively. This class relies
        on this behavior to produce the desired distribution when bounds are
        unspecified.
    """

    arg_constraints = {
        "loc": constraints.positive,
        "scale": constraints.positive,
        "beta": constraints.real,
        "mmin": constraints.positive,
        "delta": constraints.positive,
        "log_scale": constraints.less_than_eq(0.0),
    }
    reparametrized_params = ["loc", "scale", "beta", "mmin", "delta", "log_scale"]
    pytree_aux_fields = ("_support", "_norm")

    def __init__(
        self,
        loc,
        scale,
        beta,
        mmin,
        delta,
        low=None,
        high=None,
        log_scale=0.0,
        *,
        validate_args=None,
    ) -> None:
        """
        Parameters
        ----------
        loc : ArrayLike
            mean of the Gaussian distribution
        scale : ArrayLike
            standard deviation of the Gaussian distribution
        beta : ArrayLike
            Power law index for mass ratio
        mmin : ArrayLike
            Minimum mass
        delta : ArrayLike
            width of the smoothing window
        low : ArrayLike
            lower bound of the Gaussian distribution, defaults to -inf
        high : ArrayLike
            upper bound of the Gaussian distribution, defaults to inf
        log_scale : ArrayLike
            log of the scaling factor for the distribution
        """
        self.loc, self.scale, self.beta, self.mmin, self.delta, self.log_scale = (
            promote_shapes(loc, scale, beta, mmin, delta, log_scale)
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc),
            jnp.shape(scale),
            jnp.shape(beta),
            jnp.shape(mmin),
            jnp.shape(delta),
            jnp.shape(low),
            jnp.shape(high),
            jnp.shape(log_scale),
        )
        support_low = mmin
        if low is not None:
            support_low = jnp.minimum(mmin, low)
        self._support = mass_ratio_mass_sandwich(support_low, jnp.inf)
        super(SmoothedGaussianPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )
        self._norm = TruncatedNormal(
            loc=self.loc,
            scale=self.scale,
            low=low,
            high=high,
            validate_args=validate_args,
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        q = value[..., 1]
        m2 = m1 * q
        log_smoothing_m1 = log_planck_taper_window(
            (m1 - self.mmin) / jnp.where(self.delta == 0.0, 1.0, self.delta)
        )
        log_smoothing_q = log_planck_taper_window(
            (m2 - self.mmin) / jnp.where(self.delta == 0.0, 1.0, self.delta)
        )
        log_prob_m1 = self._norm.log_prob(m1)
        log_prob_q = jnp.where(
            jnp.less_equal(m1, self.mmin),
            -jnp.inf,
            doubly_truncated_power_law_log_prob(
                x=q, alpha=self.beta, low=self.mmin / m1, high=1.0
            ),
        )

        return (
            log_prob_m1
            + log_prob_q
            + log_smoothing_m1
            + log_smoothing_q
            + self.log_scale
        )


class SmoothedPowerlawAndPeak(Distribution):
    r"""It is a mixture of power law and Gaussian distribution with a smoothing kernel.

    .. math::

        p(m_1, q\mid \alpha, \beta, \mu, \sigma, m_{\text{min}}, m_{\text{max}}, \delta, \lambda_{\text{peak}}) =
        \left((1-\lambda_{\text{peak})m_1^{\alpha}+\lambda_{\text{peak}\mathcal{N}(m_1\mid\mu,\sigma)\right)
        q^{\beta}
        S\left(\frac{m_1 - m_{\text{min}}}{\delta}\right)
        S\left(\frac{m_1q - m_{\text{min}}}{\delta}\right),
        \qqquad m_{\text{min}}\leq m_1q \leq m_1\leq m_{\text{max}}
    """

    arg_constraints = {
        "alpha": constraints.real,
        "beta": constraints.real,
        "loc": constraints.real,
        "scale": constraints.positive,
        "mmin": constraints.positive,
        "mmax": constraints.positive,
        "delta": constraints.positive,
        "lambda_peak": constraints.unit_interval,
        "log_rate_pl": constraints.real,
        "log_rate_peak": constraints.real,
    }
    reparametrized_params = [
        "alpha",
        "beta",
        "loc",
        "scale",
        "mmin",
        "mmax",
        "delta",
        "lambda_peak",
        "log_rate_pl",
        "log_rate_peak",
    ]
    pytree_aux_fields = ("_support",)
    pytree_data_fields = ("_log_Z_m1", "_m1s", "_Z_q")

    def __init__(
        self,
        alpha: ArrayLike,
        beta: ArrayLike,
        loc: ArrayLike,
        scale: ArrayLike,
        mmin: ArrayLike,
        mmax: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        delta: ArrayLike,
        lambda_peak: ArrayLike,
        log_rate_pl: ArrayLike,
        log_rate_peak: ArrayLike,
        *,
        validate_args=None,
    ):
        """
        Parameters
        ----------
        alpha : ArrayLike
            Power law index for primary mass
        beta : ArrayLike
            Power law index for mass ratio
        loc : ArrayLike
            Mean of the Gaussian distribution
        scale : ArrayLike
            Standard deviation of the Gaussian distribution
        mmin : ArrayLike
            Minimum mass
        mmax : ArrayLike
            Maximum mass
        low : ArrayLike
            Lower bound of the Gaussian distribution
        high : ArrayLike
            Upper bound of the Gaussian distribution
        delta : ArrayLike
            Width of the smoothing window
        lambda_peak : ArrayLike
            Fraction of masses in the Gaussian peak
        log_rate_pl : ArrayLike
            Logarithm of the rate of the power law component
        log_rate_peak : ArrayLike
            Logarithm of the rate of the Gaussian peak component
        validate_args : bool, optional
            Whether to validate input, by default None
        """
        (
            self.alpha,
            self.beta,
            self.loc,
            self.scale,
            self.mmin,
            self.mmax,
            self.low,
            self.high,
            self.delta,
            self.lambda_peak,
            self.log_rate_pl,
            self.log_rate_peak,
        ) = promote_shapes(
            alpha,
            beta,
            loc,
            scale,
            mmin,
            mmax,
            low,
            high,
            delta,
            lambda_peak,
            log_rate_pl,
            log_rate_peak,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(loc),
            jnp.shape(scale),
            jnp.shape(mmin),
            jnp.shape(mmax),
            jnp.shape(low),
            jnp.shape(high),
            jnp.shape(delta),
            jnp.shape(lambda_peak),
            jnp.shape(log_rate_pl),
            jnp.shape(log_rate_peak),
        )
        self._support = mass_ratio_mass_sandwich(mmin, mmax)

        mmin = jnp.broadcast_to(mmin, batch_shape)
        mmax = jnp.broadcast_to(mmax, batch_shape)

        _m1s = jnp.linspace(mmin, mmax, 250, dtype=jnp.result_type(float))
        qs = jnp.linspace(
            jnp.zeros(batch_shape),
            jnp.ones(batch_shape),
            250,
            dtype=jnp.result_type(float),
        )

        if batch_shape:
            # TODO: check https://github.com/jax-ml/jax/issues/25696 and update accordingly
            meshgrid_fn = jax.vmap(
                partial(jnp.meshgrid, indexing="ij"), in_axes=(-1, -1), out_axes=-1
            )
        else:
            meshgrid_fn = partial(jnp.meshgrid, indexing="ij")

        _Z_m1 = jnp.trapezoid(jnp.exp(self._log_prob_m1(_m1s)), _m1s, axis=0)
        self._log_Z_m1 = jnp.where(self.delta == 0.0, 0.0, jnp.log(_Z_m1))

        m1qs_grid = jnp.stack(meshgrid_fn(_m1s, qs), axis=-1)
        _log_prob_q = self._log_prob_q(m1qs_grid)

        self._Z_q = jnp.trapezoid(
            jnp.exp(_log_prob_q), jnp.expand_dims(qs, axis=0), axis=1
        )

        self._m1s = _m1s

        super(SmoothedPowerlawAndPeak, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    def _log_prob_m1(
        self, m1: Array, log_rate_pl: Array = 0.0, log_rate_peak: Array = 0.0
    ) -> Array:
        log_smoothing_m1 = log_planck_taper_window(
            (m1 - self.mmin) / jnp.where(self.delta == 0.0, 1.0, self.delta)
        )
        log_prob_m1 = jnp.log(
            (1.0 - self.lambda_peak)
            * jnp.exp(
                log_rate_pl
                + doubly_truncated_power_law_log_prob(
                    x=m1, alpha=self.alpha, low=self.mmin, high=self.mmax
                )
            )
            + self.lambda_peak
            * jnp.exp(
                log_rate_peak
                + truncnorm.logpdf(
                    m1,
                    a=(self.loc - self.low) / self.scale,
                    b=(self.high - self.loc) / self.scale,
                    loc=self.loc,
                    scale=self.scale,
                )
            )
        )
        return log_prob_m1 + log_smoothing_m1

    @validate_sample
    def _log_prob_q(self, m1q: Array) -> Array:
        m1 = m1q[..., 0]
        q = m1q[..., 1]
        m2 = m1 * q
        log_smoothing_q = log_planck_taper_window(
            (m2 - self.mmin) / jnp.where(self.delta == 0.0, 1.0, self.delta)
        )
        log_prob_q = jnp.where(
            jnp.less_equal(m1, self.mmin),
            -jnp.inf,
            doubly_truncated_power_law_log_prob(
                x=q, alpha=self.beta, low=self.mmin / m1, high=1.0
            ),
        )
        return log_prob_q + log_smoothing_q

    @validate_sample
    def log_prob(self, value: ArrayLike) -> Array:
        m1 = value[..., 0]

        log_prob_m1 = self._log_prob_m1(
            m1, log_rate_pl=self.log_rate_pl, log_rate_peak=self.log_rate_peak
        )

        log_prob_q = self._log_prob_q(value)

        def _Z_q(m1s: ArrayLike, Z_qs: ArrayLike) -> ArrayLike:
            return jax.vmap(partial(jnp.interp, xp=m1s, fp=Z_qs, left=1.0, right=1.0))(
                m1
            )

        if self.batch_shape:
            log_Z_q = jnp.log(
                jax.vmap(
                    _Z_q,
                    in_axes=(-1, -1),
                    out_axes=-1,
                )(self._m1s, self._Z_q)
            )
            log_Z_q = jnp.reshape(log_Z_q, log_prob_q.shape)
        else:
            log_Z_q = jnp.log(_Z_q(self._m1s, self._Z_q))

        log_Z = lax.stop_gradient(self._log_Z_m1 + log_Z_q)

        return log_prob_m1 + log_prob_q - log_Z
