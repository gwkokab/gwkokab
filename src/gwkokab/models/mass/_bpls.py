# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import numpy as jnp
from jax._src.lax import lax
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample

from ...utils.kernel import log_planck_taper_window
from ...utils.math import truncnorm_logpdf
from ..constraints import mass_ratio_mass_sandwich
from ..utils import doubly_truncated_power_law_log_prob


@jax.jit
def _broken_powerlaw_log_prob(
    m1: Array,
    alpha1: Array,
    alpha2: Array,
    mmin: Array,
    mmax: Array,
    mbreak: Array,
) -> Array:
    r"""Calculate the log probability of the broken powerlaw two peak distribution.

    The broken powerlaw two peak distribution is defined as

    .. math::
        p(m_1\mid \alpha_1, \alpha_2, m_{\mathrm{min}}, m_{\mathrm{max}}, m_{\mathrm{break}}) \propto
        \begin{cases}
            \left(\frac{m_1}{m_{\mathrm{break}}}\right)^{-\alpha_1}
            & m_{\mathrm{min}} \leq m_1 < m_{\text{break}} \\
            \left(\frac{m_1}{m_{\mathrm{break}}}\right)^{-\alpha_2}
            & m_{\text{break}} \leq m_1 \leq m_{\mathrm{max}} \\
            0 & \text{otherwise}
        \end{cases}
    """
    log_norm = jnp.logaddexp(
        -doubly_truncated_power_law_log_prob(
            mbreak, alpha=-alpha1, low=mmin, high=mbreak
        ),
        -doubly_truncated_power_law_log_prob(
            mbreak, alpha=-alpha2, low=mbreak, high=mmax
        ),
    )
    log_prob = jnp.where(m1 < mbreak, alpha1, alpha2) * jnp.log(mbreak / m1)
    return log_prob - log_norm


class BrokenPowerlaw(Distribution):
    arg_constraints = {
        "alpha1": constraints.real,
        "alpha2": constraints.real,
        "mbreak": constraints.positive,
        "mmax": constraints.positive,
        "mmin": constraints.positive,
    }
    pytree_data_fields = ("_support", "alpha1", "alpha2", "mbreak", "mmax", "mmin")

    def __init__(
        self,
        alpha1: ArrayLike,
        alpha2: ArrayLike,
        mbreak: ArrayLike,
        mmax: ArrayLike,
        mmin: ArrayLike,
        validate_args: Optional[bool] = None,
    ) -> None:
        (self.alpha1, self.alpha2, self.mbreak, self.mmax, self.mmin) = promote_shapes(
            alpha1, alpha2, mbreak, mmax, mmin
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha1),
            jnp.shape(alpha2),
            jnp.shape(mbreak),
            jnp.shape(mmax),
            jnp.shape(mmin),
        )

        self._support = constraints.interval(mmin, mmax)
        super(BrokenPowerlaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        return _broken_powerlaw_log_prob(
            value, self.alpha1, self.alpha2, self.mmin, self.mmax, self.mbreak
        )


class BrokenPowerlawTwoPeak(Distribution):
    r"""Broken Powerlaw + 2 Peak is defined as a mixture of one Broken Powerlaw and two
    left truncated Normal distributions. For more details, see appendix B.3 of
    `GWTC-4.0: Population Properties of Merging Compact Binaries <https://arxiv.org/abs/2508.18083>`_.

    It is defined as follows:

    .. math::

        p_{\mathrm{BP}}(m_1 \mid \alpha_1, \alpha_2, m_{1,\mathrm{min}}, m_{\mathrm{max}},
        m_{\mathrm{break}}) \propto
        \begin{cases}
            \left(\frac{m_1}{m_{\mathrm{break}}}\right)^{-\alpha_1}
            & m_{1,\mathrm{min}} \leq m_1 < m_{\text{break}}  \\
            \left(\frac{m_1}{m_{\mathrm{break}}}\right)^{-\alpha_2}
            & m_{\text{break}} \leq m_1 \leq m_{\mathrm{max}} \\
            0 & \text{otherwise}
        \end{cases}

    .. math::

        p(m_1 \mid \alpha_1, \alpha_2, m_{1,\mathrm{min}}, m_{\mathrm{max}},
        m_{\mathrm{break}}, \lambda_0, \lambda_1, \mu_1, \sigma_1, \mu_2, \sigma_2)
        \propto
        \left(
        \lambda_0 p_{\mathrm{BP}}(m_1 \mid \alpha_1, \alpha_2, m_{1,\mathrm{min}}, m_{\mathrm{max}}, m_{\mathrm{break}}) +
        \lambda_1 \mathcal{N}_{[m_{1,\mathrm{min}}, \infty)}(m_1 \mid \mu_1, \sigma_1) +
        (1 - \lambda_0 - \lambda_1) \mathcal{N}_{[m_{1,\mathrm{min}}, \infty)}(m_1 \mid \mu_2, \sigma_2)
        \right)
        S\left(\frac{m_1 - m_{1,\mathrm{min}}}{\delta_{m_1}}\right)

    .. math::

        p(q \mid \beta, m_1, m_{2,\mathrm{min}}, \delta_{m_2}) \propto
        q^{\beta} S\left(\frac{m_1 q - m_{2,\mathrm{min}}}{\delta_{m_2}}\right)

    .. math::
        p(m_1, q \mid \alpha_1, \alpha_2, m_{1,\mathrm{min}}, m_{2,\mathrm{min}},
        m_{\mathrm{max}}, m_{\mathrm{break}}, \lambda_0, \lambda_1, \mu_1, \sigma_1,
        \mu_2, \sigma_2, \beta, \delta_{m_1}, \delta_{m_2}) =
        p(m_1 \mid \alpha_1, \alpha_2, m_{1,\mathrm{min}}, m_{\mathrm{max}}, m_{\mathrm{break}}, \lambda_0,
        \lambda_1, \mu_1, \sigma_1, \mu_2, \sigma_2)
        p(q \mid \beta, m_1, m_{2,\mathrm{min}}, \delta_{m_2})

    where :math:`\mathcal{N}_{[m_{1,\mathrm{min}}, \infty)}(m \mid \mu, \sigma)` is a left
    truncated normal distribution with mean :math:`\mu` and standard deviation
    :math:`\sigma`, :math:`\lambda_0` and :math:`\lambda_1` are the mixing fractions of
    the broken powerlaw and first Gaussian component respectively, :math:`\delta_{m_1}`
    is the smoothing scale for the primary mass, and :math:`S(x)` is the exponential of
    :func:`~gwkokab.utils.kernel.log_planck_taper_window`.
    """

    arg_constraints = {
        "alpha1": constraints.real,
        "alpha2": constraints.real,
        "beta": constraints.real,
        "delta_m1": constraints.positive,
        "delta_m2": constraints.positive,
        "lambda_0": constraints.unit_interval,
        "lambda_1": constraints.unit_interval,
        "loc1": constraints.positive,
        "loc2": constraints.positive,
        "m1min": constraints.positive,
        "m2min": constraints.positive,
        "mbreak": constraints.positive,
        "mmax": constraints.positive,
        "scale1": constraints.positive,
        "scale2": constraints.positive,
    }

    pytree_data_fields = (
        "_logZ",
        "_m1s",
        "_support",
        "_Z_q_given_m1",
        "alpha1",
        "alpha2",
        "beta",
        "delta_m1",
        "delta_m2",
        "lambda_0",
        "lambda_1",
        "loc1",
        "loc2",
        "m1min",
        "m2min",
        "mbreak",
        "mmax",
        "scale1",
        "scale2",
    )

    def __init__(
        self,
        alpha1: ArrayLike,
        alpha2: ArrayLike,
        beta: ArrayLike,
        loc1: ArrayLike,
        loc2: ArrayLike,
        scale1: ArrayLike,
        scale2: ArrayLike,
        delta_m1: ArrayLike,
        delta_m2: ArrayLike,
        lambda_0: ArrayLike,
        lambda_1: ArrayLike,
        m1min: ArrayLike,
        m2min: ArrayLike,
        mmax: ArrayLike,
        mbreak: ArrayLike,
        validate_args: Optional[bool] = None,
    ) -> None:
        (
            self.alpha1,
            self.alpha2,
            self.beta,
            self.loc1,
            self.loc2,
            self.scale1,
            self.scale2,
            self.delta_m1,
            self.delta_m2,
            self.lambda_0,
            self.lambda_1,
            self.m1min,
            self.m2min,
            self.mmax,
            self.mbreak,
        ) = promote_shapes(
            alpha1,
            alpha2,
            beta,
            loc1,
            loc2,
            scale1,
            scale2,
            delta_m1,
            delta_m2,
            lambda_0,
            lambda_1,
            m1min,
            m2min,
            mmax,
            mbreak,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha1),
            jnp.shape(alpha2),
            jnp.shape(beta),
            jnp.shape(loc1),
            jnp.shape(loc2),
            jnp.shape(scale1),
            jnp.shape(scale2),
            jnp.shape(delta_m1),
            jnp.shape(delta_m2),
            jnp.shape(lambda_0),
            jnp.shape(lambda_1),
            jnp.shape(m1min),
            jnp.shape(m2min),
            jnp.shape(mmax),
            jnp.shape(mbreak),
        )

        self._support = mass_ratio_mass_sandwich(m2min, mmax)
        super(BrokenPowerlawTwoPeak, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

        m1min = jnp.broadcast_to(m1min, batch_shape)
        m2min = jnp.broadcast_to(m2min, batch_shape)
        mmax = jnp.broadcast_to(mmax, batch_shape)

        m1s = jnp.linspace(m1min, mmax, 1000)
        self._logZ = jnp.log(
            jnp.trapezoid(
                jnp.exp(self._log_prob_m1_unnorm(m1s)),
                m1s,
                axis=0,
            )
        )

        self._m1s = jnp.linspace(m2min, mmax, 1000)
        _qs = jnp.linspace(0.005, 1.0, 500)
        _m1qs_grid = jnp.stack(jnp.meshgrid(self._m1s, _qs, indexing="ij"), axis=-1)

        _prob_q = jnp.exp(self._log_prob_q_unnorm(_m1qs_grid))

        self._Z_q_given_m1 = jnp.trapezoid(_prob_q, _qs, axis=1)

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    def _log_prob_m1_unnorm(self, m1: Array) -> Array:
        safe_delta = jnp.where(self.delta_m1 <= 0.0, 1.0, self.delta_m1)
        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / safe_delta)
        broken_powerlaw_log_prob = _broken_powerlaw_log_prob(
            m1, self.alpha1, self.alpha2, self.m1min, self.mmax, self.mbreak
        )
        log_prob_norm_0 = truncnorm_logpdf(
            xx=m1,
            loc=self.loc1,
            scale=self.scale1,
            low=self.m1min,
            high=self.mmax,
        )
        log_prob_norm_1 = truncnorm_logpdf(
            xx=m1,
            loc=self.loc2,
            scale=self.scale2,
            low=self.m1min,
            high=self.mmax,
        )

        log_prob_m1 = log_smoothing_m1 + jnp.log(
            self.lambda_0 * jnp.exp(broken_powerlaw_log_prob)
            + self.lambda_1 * jnp.exp(log_prob_norm_0)
            + (1 - (self.lambda_0 + self.lambda_1)) * jnp.exp(log_prob_norm_1)
        )

        return jnp.where(
            (self.delta_m1 <= 0.0) | (m1 < self.m1min), -jnp.inf, log_prob_m1
        )

    @validate_sample
    def _log_prob_q_unnorm(self, value: Array) -> Array:
        m1, q = jnp.unstack(value, axis=-1)
        m2 = m1 * q
        safe_delta = jnp.where(self.delta_m2 <= 0.0, 1.0, self.delta_m2)
        log_smoothing_q = log_planck_taper_window((m2 - self.m2min) / safe_delta)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_q

        return jnp.where(
            (self.delta_m2 <= 0.0) | (m2 < self.m2min), -jnp.inf, log_prob_q
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        m1, _ = jnp.unstack(value, axis=-1)
        log_prob_m1 = self._log_prob_m1_unnorm(m1) - self._logZ
        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1, left=1.0, right=1.0)
        safe_Z_q = jnp.where(_Z_q <= 0, 1.0, _Z_q)
        log_Z_q = jnp.where(_Z_q <= 0, 0.0, jnp.log(safe_Z_q))
        log_prob_q = self._log_prob_q_unnorm(value) - log_Z_q
        return log_prob_m1 + log_prob_q


class SmoothedBrokenPowerlawMassRatioPowerlaw(Distribution):
    arg_constraints = {
        "alpha1": constraints.real,
        "alpha2": constraints.real,
        "beta": constraints.real,
        "delta_m1": constraints.positive,
        "delta_m2": constraints.positive,
        "m1min": constraints.positive,
        "m2min": constraints.positive,
        "mbreak": constraints.positive,
        "mmax": constraints.positive,
    }
    pytree_data_fields = (
        "_logZ",
        "_m1s",
        "_support",
        "_Z_q_given_m1",
        "alpha1",
        "alpha2",
        "beta",
        "delta_m1",
        "delta_m2",
        "m1min",
        "m2min",
        "mbreak",
        "mmax",
    )

    def __init__(
        self,
        alpha1: ArrayLike,
        alpha2: ArrayLike,
        beta: ArrayLike,
        mbreak: ArrayLike,
        mmax: ArrayLike,
        m1min: ArrayLike,
        m2min: ArrayLike,
        delta_m1: ArrayLike,
        delta_m2: ArrayLike,
        validate_args: Optional[bool] = None,
    ) -> None:
        (
            self.alpha1,
            self.alpha2,
            self.beta,
            self.mbreak,
            self.mmax,
            self.m1min,
            self.m2min,
            self.delta_m1,
            self.delta_m2,
        ) = promote_shapes(
            alpha1, alpha2, beta, mbreak, mmax, m1min, m2min, delta_m1, delta_m2
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha1),
            jnp.shape(alpha2),
            jnp.shape(beta),
            jnp.shape(mbreak),
            jnp.shape(mmax),
            jnp.shape(m1min),
            jnp.shape(m2min),
            jnp.shape(delta_m1),
            jnp.shape(delta_m2),
        )

        self._support = mass_ratio_mass_sandwich(m2min, mmax)
        super(SmoothedBrokenPowerlawMassRatioPowerlaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

        m1min = jnp.broadcast_to(m1min, batch_shape)
        mmax = jnp.broadcast_to(mmax, batch_shape)

        # Compute the normalization constant for primary mass distribution

        self._m1s = jnp.linspace(m1min, mmax, 1000, dtype=jnp.result_type(float))

        _Z = jnp.trapezoid(jnp.exp(self._log_prob_m1(self._m1s)), self._m1s, axis=0)
        self._logZ = jnp.where(
            jnp.isnan(_Z) | jnp.isinf(_Z) | jnp.less(_Z, 0.0), 0.0, jnp.log(_Z)
        )

        # Compute the normalization constant for mass ratio distribution

        _qs = jnp.linspace(0.005, 1.0, 500, dtype=jnp.result_type(float))
        _m1qs_grid = jnp.stack(jnp.meshgrid(self._m1s, _qs, indexing="ij"), axis=-1)

        _prob_q = jnp.exp(self._log_prob_q(jnp.expand_dims(_m1qs_grid, axis=-2)))

        self._Z_q_given_m1 = jnp.clip(
            jnp.trapezoid(_prob_q, _qs, axis=1).reshape(
                *(self._m1s.shape + batch_shape)
            ),
            min=jnp.finfo(jnp.result_type(float)).tiny,
            max=jnp.finfo(jnp.result_type(float)).max,
        )
        del _m1qs_grid, _qs, _prob_q

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    def _log_prob_m1(self, m1: Array, logZ: ArrayLike = 0.0) -> Array:
        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / self.delta_m1)
        log_prob_powerlaw = _broken_powerlaw_log_prob(
            m1, self.alpha1, self.alpha2, self.m1min, self.mmax, self.mbreak
        )
        log_prob_m1 = log_prob_powerlaw + log_smoothing_m1 - logZ
        return jnp.nan_to_num(
            log_prob_m1,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

    def _log_prob_q(self, value: Array, logZ: ArrayLike = 0.0) -> Array:
        m1, q = jnp.unstack(value, axis=-1)
        m2 = m1 * q
        log_smoothing_q = log_planck_taper_window((m2 - self.m2min) / self.delta_m2)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_q - logZ
        mask = self.support.check(value)
        log_prob_q = jnp.where(mask, log_prob_q, -jnp.inf)
        return jnp.nan_to_num(
            log_prob_q,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        m1, _ = jnp.unstack(value, axis=-1)
        log_prob_m1 = self._log_prob_m1(m1, self._logZ)
        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1)
        log_Z_q = jnp.where(
            jnp.isnan(_Z_q) | jnp.isinf(_Z_q) | jnp.less(_Z_q, 0.0),
            0.0,
            jnp.log(_Z_q),
        )
        log_prob_q = self._log_prob_q(value, log_Z_q)
        return log_prob_m1 + log_prob_q
