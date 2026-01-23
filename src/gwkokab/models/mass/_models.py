# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import lax, numpy as jnp, random as jrd
from jax.scipy import special
from jax.scipy.stats import norm, uniform
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample

from ...utils.kernel import log_planck_taper_window
from ..constraints import mass_ratio_mass_sandwich, mass_sandwich
from ..utils import (
    doubly_truncated_power_law_icdf,
    doubly_truncated_power_law_log_norm_constant,
    doubly_truncated_power_law_log_prob,
)


class PowerlawPrimaryMassRatio(Distribution):
    r"""Power law model for two-dimensional mass distribution, modelling primary mass and
    conditional mass ratio distribution.

    .. math::
        p(m_1,q\mid\alpha,\beta) = p(m_1\mid\alpha)p(q \mid m_1, \beta)

    .. math::
        \begin{align*}
            p(m_1\mid\alpha)&
            \propto m_1^{-\alpha},\qquad m_{\text{min}}\leq m_1\leq m_{\max}\\
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
    pytree_data_fields = ("_support", "alpha", "beta", "mmax", "mmin")

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
        m1, q = jnp.unstack(value, axis=-1)
        log_prob_m1 = doubly_truncated_power_law_log_prob(
            x=m1, alpha=-self.alpha, low=self.mmin, high=self.mmax
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
            q=u_m1, alpha=-self.alpha, low=self.mmin, high=self.mmax
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
    pytree_data_fields = ("_support", "alpha_m", "mmax", "mmin")

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


class SmoothedTwoComponentPrimaryMassRatio(Distribution):
    arg_constraints = {
        "alpha": constraints.real,
        "beta": constraints.real,
        "delta": constraints.positive,
        "lambda_peak": constraints.unit_interval,
        "loc": constraints.positive,
        "mmax": constraints.positive,
        "mmin": constraints.positive,
        "scale": constraints.positive,
    }
    pytree_data_fields = (
        "_logZ",
        "_m1s",
        "_support",
        "_Z_q_given_m1",
        "alpha",
        "beta",
        "delta",
        "lambda_peak",
        "loc",
        "mmax",
        "mmin",
        "scale",
    )

    def __init__(
        self,
        alpha: ArrayLike,
        beta: ArrayLike,
        delta: ArrayLike,
        lambda_peak: ArrayLike,
        loc: ArrayLike,
        mmax: ArrayLike,
        mmin: ArrayLike,
        scale: ArrayLike,
        *,
        validate_args=None,
    ) -> None:
        (
            self.alpha,
            self.beta,
            self.delta,
            self.lambda_peak,
            self.loc,
            self.mmax,
            self.mmin,
            self.scale,
        ) = promote_shapes(
            alpha,
            beta,
            delta,
            lambda_peak,
            loc,
            mmax,
            mmin,
            scale,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(delta),
            jnp.shape(lambda_peak),
            jnp.shape(loc),
            jnp.shape(mmax),
            jnp.shape(mmin),
            jnp.shape(scale),
        )
        self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(SmoothedTwoComponentPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

        mmin = jnp.broadcast_to(mmin, batch_shape)
        mmax = jnp.broadcast_to(mmax, batch_shape)

        # Compute the normalization constant for primary mass distribution

        _m1s_delta = jnp.linspace(mmin, mmin + delta, 100)
        numerical_log_norm = jnp.trapezoid(
            jnp.exp(self._log_prob_m1_unnorm(_m1s_delta)),
            _m1s_delta,
            axis=0,
        )

        analytical_log_norm = (1 - self.lambda_peak) * jnp.exp(
            doubly_truncated_power_law_log_norm_constant(
                -self.alpha, self.mmin + self.delta, self.mmax
            )
            - doubly_truncated_power_law_log_norm_constant(
                -self.alpha, self.mmin, self.mmax
            )
        ) + self.lambda_peak * (
            special.ndtr((self.mmax - self.loc) / self.scale)
            - special.ndtr((self.mmin + self.delta - self.loc) / self.scale)
        )

        self._logZ = jnp.log(numerical_log_norm + analytical_log_norm)

        del _m1s_delta

        # Compute the normalization constant for mass ratio distribution

        self._m1s = jnp.linspace(mmin, mmax, 1000)
        _qs = jnp.linspace(0.005, 1.0, 500)
        _m1qs_grid = jnp.stack(jnp.meshgrid(self._m1s, _qs, indexing="ij"), axis=-1)

        _prob_q = jnp.exp(self._log_prob_q_unnorm(_m1qs_grid))

        self._Z_q_given_m1 = jnp.trapezoid(_prob_q, _qs, axis=1)
        del _m1qs_grid, _qs, _prob_q

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    def _log_prob_m1_unnorm(self, m1: Array) -> Array:
        safe_delta = jnp.where(self.delta <= 0.0, 1.0, self.delta)
        log_smoothing_m1 = log_planck_taper_window((m1 - self.mmin) / safe_delta)
        log_norm_powerlaw = doubly_truncated_power_law_log_norm_constant(
            -self.alpha, self.mmin, self.mmax
        )
        prob_norm = norm.pdf(m1, loc=self.loc, scale=self.scale)
        log_prob_m1 = (
            jnp.log(
                (1 - self.lambda_peak)
                * jnp.power(m1, -self.alpha)
                * jnp.exp(-log_norm_powerlaw)
                + self.lambda_peak * prob_norm
            )
            + log_smoothing_m1
        )

        return jnp.where(self.delta <= 0.0, -jnp.inf, log_prob_m1)

    @validate_sample
    def _log_prob_q_unnorm(self, value: Array) -> Array:
        m1, q = jnp.unstack(value, axis=-1)
        m2 = m1 * q
        safe_delta = jnp.where(self.delta <= 0.0, 1.0, self.delta)
        log_smoothing_q = log_planck_taper_window((m2 - self.mmin) / safe_delta)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_q

        return jnp.where(self.delta <= 0.0, -jnp.inf, log_prob_q)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        m1, _ = jnp.unstack(value, axis=-1)
        log_prob_m1 = self._log_prob_m1_unnorm(m1) - self._logZ
        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1, left=1.0, right=1.0)
        safe_Z_q = jnp.where(_Z_q <= 0, 1.0, _Z_q)
        log_Z_q = jnp.where(_Z_q <= 0, 0.0, jnp.log(safe_Z_q))
        log_prob_q = self._log_prob_q_unnorm(value) - log_Z_q
        return log_prob_m1 + log_prob_q
