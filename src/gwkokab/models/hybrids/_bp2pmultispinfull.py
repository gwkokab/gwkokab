# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import lax, numpy as jnp
from jax.scipy.stats import truncnorm
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints, Distribution, HalfNormal
from numpyro.distributions.util import promote_shapes, validate_sample

from ...utils.kernel import log_planck_taper_window
from ..constraints import all_constraint, mass_sandwich
from ..redshift import PowerlawRedshift
from ..utils import (
    doubly_truncated_power_law_log_norm_constant,
    JointDistribution,
    ScaledMixture,
)


class BrokenPowerlawTwoPeakMultiSpinMultiTilt(Distribution):
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
        "m1min": constraints.real,
        "m2min": constraints.real,
        "mbreak": constraints.real,
        "mmax": constraints.real,
        "scale1": constraints.positive,
        "scale2": constraints.positive,
        "a_1_loc_bpl1": constraints.unit_interval,
        "a_1_loc_bpl2": constraints.unit_interval,
        "a_1_loc_n1": constraints.unit_interval,
        "a_1_loc_n2": constraints.unit_interval,
        "a_2_loc_bpl1": constraints.unit_interval,
        "a_2_loc_bpl2": constraints.unit_interval,
        "a_2_loc_n1": constraints.unit_interval,
        "a_2_loc_n2": constraints.unit_interval,
        "a_1_scale_bpl1": constraints.positive,
        "a_1_scale_bpl2": constraints.positive,
        "a_1_scale_n1": constraints.positive,
        "a_1_scale_n2": constraints.positive,
        "a_2_scale_bpl1": constraints.positive,
        "a_2_scale_bpl2": constraints.positive,
        "a_2_scale_n1": constraints.positive,
        "a_2_scale_n2": constraints.positive,
        "cos_tilt_1_loc_bpl1": constraints.interval(-1.0, 1.0),
        "cos_tilt_1_loc_bpl2": constraints.interval(-1.0, 1.0),
        "cos_tilt_1_loc_n1": constraints.interval(-1.0, 1.0),
        "cos_tilt_1_loc_n2": constraints.interval(-1.0, 1.0),
        "cos_tilt_1_scale_bpl1": constraints.positive,
        "cos_tilt_1_scale_bpl2": constraints.positive,
        "cos_tilt_1_scale_n1": constraints.positive,
        "cos_tilt_1_scale_n2": constraints.positive,
        "cos_tilt_2_loc_bpl1": constraints.interval(-1.0, 1.0),
        "cos_tilt_2_loc_bpl2": constraints.interval(-1.0, 1.0),
        "cos_tilt_2_loc_n1": constraints.interval(-1.0, 1.0),
        "cos_tilt_2_loc_n2": constraints.interval(-1.0, 1.0),
        "cos_tilt_2_scale_bpl1": constraints.positive,
        "cos_tilt_2_scale_bpl2": constraints.positive,
        "cos_tilt_2_scale_n1": constraints.positive,
        "cos_tilt_2_scale_n2": constraints.positive,
        "cos_tilt_zeta_bpl1": constraints.unit_interval,
        "cos_tilt_zeta_bpl2": constraints.unit_interval,
        "cos_tilt_zeta_n1": constraints.unit_interval,
        "cos_tilt_zeta_n2": constraints.unit_interval,
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
        "a_1_loc_bpl1",
        "a_1_loc_bpl2",
        "a_1_loc_n1",
        "a_1_loc_n2",
        "a_2_loc_bpl1",
        "a_2_loc_bpl2",
        "a_2_loc_n1",
        "a_2_loc_n2",
        "a_1_scale_bpl1",
        "a_1_scale_bpl2",
        "a_1_scale_n1",
        "a_1_scale_n2",
        "a_2_scale_bpl1",
        "a_2_scale_bpl2",
        "a_2_scale_n1",
        "a_2_scale_n2",
        "cos_tilt_1_loc_bpl1",
        "cos_tilt_1_loc_bpl2",
        "cos_tilt_1_loc_n1",
        "cos_tilt_1_loc_n2",
        "cos_tilt_1_scale_bpl1",
        "cos_tilt_1_scale_bpl2",
        "cos_tilt_1_scale_n1",
        "cos_tilt_1_scale_n2",
        "cos_tilt_2_loc_bpl1",
        "cos_tilt_2_loc_bpl2",
        "cos_tilt_2_loc_n1",
        "cos_tilt_2_loc_n2",
        "cos_tilt_2_scale_bpl1",
        "cos_tilt_2_scale_bpl2",
        "cos_tilt_2_scale_n1",
        "cos_tilt_2_scale_n2",
        "cos_tilt_zeta_bpl1",
        "cos_tilt_zeta_bpl2",
        "cos_tilt_zeta_n1",
        "cos_tilt_zeta_n2",
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
        a_1_loc_bpl1: ArrayLike,
        a_1_loc_bpl2: ArrayLike,
        a_1_loc_n1: ArrayLike,
        a_1_loc_n2: ArrayLike,
        a_2_loc_bpl1: ArrayLike,
        a_2_loc_bpl2: ArrayLike,
        a_2_loc_n1: ArrayLike,
        a_2_loc_n2: ArrayLike,
        a_1_scale_bpl1: ArrayLike,
        a_1_scale_bpl2: ArrayLike,
        a_1_scale_n1: ArrayLike,
        a_1_scale_n2: ArrayLike,
        a_2_scale_bpl1: ArrayLike,
        a_2_scale_bpl2: ArrayLike,
        a_2_scale_n1: ArrayLike,
        a_2_scale_n2: ArrayLike,
        cos_tilt_1_loc_bpl1: ArrayLike,
        cos_tilt_1_loc_bpl2: ArrayLike,
        cos_tilt_1_loc_n1: ArrayLike,
        cos_tilt_1_loc_n2: ArrayLike,
        cos_tilt_1_scale_bpl1: ArrayLike,
        cos_tilt_1_scale_bpl2: ArrayLike,
        cos_tilt_1_scale_n1: ArrayLike,
        cos_tilt_1_scale_n2: ArrayLike,
        cos_tilt_2_loc_bpl1: ArrayLike,
        cos_tilt_2_loc_bpl2: ArrayLike,
        cos_tilt_2_loc_n1: ArrayLike,
        cos_tilt_2_loc_n2: ArrayLike,
        cos_tilt_2_scale_bpl1: ArrayLike,
        cos_tilt_2_scale_bpl2: ArrayLike,
        cos_tilt_2_scale_n1: ArrayLike,
        cos_tilt_2_scale_n2: ArrayLike,
        cos_tilt_zeta_bpl1: ArrayLike,
        cos_tilt_zeta_bpl2: ArrayLike,
        cos_tilt_zeta_n1: ArrayLike,
        cos_tilt_zeta_n2: ArrayLike,
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
            self.a_1_loc_bpl1,
            self.a_1_loc_bpl2,
            self.a_1_loc_n1,
            self.a_1_loc_n2,
            self.a_2_loc_bpl1,
            self.a_2_loc_bpl2,
            self.a_2_loc_n1,
            self.a_2_loc_n2,
            self.a_1_scale_bpl1,
            self.a_1_scale_bpl2,
            self.a_1_scale_n1,
            self.a_1_scale_n2,
            self.a_2_scale_bpl1,
            self.a_2_scale_bpl2,
            self.a_2_scale_n1,
            self.a_2_scale_n2,
            self.cos_tilt_1_loc_bpl1,
            self.cos_tilt_1_loc_bpl2,
            self.cos_tilt_1_loc_n1,
            self.cos_tilt_1_loc_n2,
            self.cos_tilt_1_scale_bpl1,
            self.cos_tilt_1_scale_bpl2,
            self.cos_tilt_1_scale_n1,
            self.cos_tilt_1_scale_n2,
            self.cos_tilt_2_loc_bpl1,
            self.cos_tilt_2_loc_bpl2,
            self.cos_tilt_2_loc_n1,
            self.cos_tilt_2_loc_n2,
            self.cos_tilt_2_scale_bpl1,
            self.cos_tilt_2_scale_bpl2,
            self.cos_tilt_2_scale_n1,
            self.cos_tilt_2_scale_n2,
            self.cos_tilt_zeta_bpl1,
            self.cos_tilt_zeta_bpl2,
            self.cos_tilt_zeta_n1,
            self.cos_tilt_zeta_n2,
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
            a_1_loc_bpl1,
            a_1_loc_bpl2,
            a_1_loc_n1,
            a_1_loc_n2,
            a_2_loc_bpl1,
            a_2_loc_bpl2,
            a_2_loc_n1,
            a_2_loc_n2,
            a_1_scale_bpl1,
            a_1_scale_bpl2,
            a_1_scale_n1,
            a_1_scale_n2,
            a_2_scale_bpl1,
            a_2_scale_bpl2,
            a_2_scale_n1,
            a_2_scale_n2,
            cos_tilt_1_loc_bpl1,
            cos_tilt_1_loc_bpl2,
            cos_tilt_1_loc_n1,
            cos_tilt_1_loc_n2,
            cos_tilt_1_scale_bpl1,
            cos_tilt_1_scale_bpl2,
            cos_tilt_1_scale_n1,
            cos_tilt_1_scale_n2,
            cos_tilt_2_loc_bpl1,
            cos_tilt_2_loc_bpl2,
            cos_tilt_2_loc_n1,
            cos_tilt_2_loc_n2,
            cos_tilt_2_scale_bpl1,
            cos_tilt_2_scale_bpl2,
            cos_tilt_2_scale_n1,
            cos_tilt_2_scale_n2,
            cos_tilt_zeta_bpl1,
            cos_tilt_zeta_bpl2,
            cos_tilt_zeta_n1,
            cos_tilt_zeta_n2,
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
            jnp.shape(a_1_loc_bpl1),
            jnp.shape(a_1_loc_bpl2),
            jnp.shape(a_1_loc_n1),
            jnp.shape(a_1_loc_n2),
            jnp.shape(a_2_loc_bpl1),
            jnp.shape(a_2_loc_bpl2),
            jnp.shape(a_2_loc_n1),
            jnp.shape(a_2_loc_n2),
            jnp.shape(a_1_scale_bpl1),
            jnp.shape(a_1_scale_bpl2),
            jnp.shape(a_1_scale_n1),
            jnp.shape(a_1_scale_n2),
            jnp.shape(a_2_scale_bpl1),
            jnp.shape(a_2_scale_bpl2),
            jnp.shape(a_2_scale_n1),
            jnp.shape(a_2_scale_n2),
            jnp.shape(cos_tilt_1_loc_bpl1),
            jnp.shape(cos_tilt_1_loc_bpl2),
            jnp.shape(cos_tilt_1_loc_n1),
            jnp.shape(cos_tilt_1_loc_n2),
            jnp.shape(cos_tilt_1_scale_bpl1),
            jnp.shape(cos_tilt_1_scale_bpl2),
            jnp.shape(cos_tilt_1_scale_n1),
            jnp.shape(cos_tilt_1_scale_n2),
            jnp.shape(cos_tilt_2_loc_bpl1),
            jnp.shape(cos_tilt_2_loc_bpl2),
            jnp.shape(cos_tilt_2_loc_n1),
            jnp.shape(cos_tilt_2_loc_n2),
            jnp.shape(cos_tilt_2_scale_bpl1),
            jnp.shape(cos_tilt_2_scale_bpl2),
            jnp.shape(cos_tilt_2_scale_n1),
            jnp.shape(cos_tilt_2_scale_n2),
            jnp.shape(cos_tilt_zeta_bpl1),
            jnp.shape(cos_tilt_zeta_bpl2),
            jnp.shape(cos_tilt_zeta_n1),
            jnp.shape(cos_tilt_zeta_n2),
        )

        self._support = all_constraint(
            [
                mass_sandwich(m2min, mmax),
                constraints.unit_interval,
                constraints.unit_interval,
                constraints.interval(-1.0, 1.0),
                constraints.interval(-1.0, 1.0),
            ],
            [(0, 2), 2, 3, 4, 5],
        )
        super(BrokenPowerlawTwoPeakMultiSpinMultiTilt, self).__init__(
            batch_shape=batch_shape, event_shape=(6,), validate_args=validate_args
        )

        m1min = jnp.broadcast_to(m1min, batch_shape)
        m2min = jnp.broadcast_to(m2min, batch_shape)
        mmax = jnp.broadcast_to(mmax, batch_shape)

        m1s = jnp.linspace(m1min, mmax, 1000)
        self._logZ = jnp.log(
            jnp.trapezoid(
                jnp.exp(self._log_prob_m1_unnorm_component(m1s)).sum(axis=0),
                m1s,
                axis=0,
            )
        )

        self._m1s = jnp.linspace(m2min, mmax, 1000)
        _qs = jnp.linspace(0.005, 1.0, 500)
        _m1_grid, qs_grid = jnp.meshgrid(self._m1s, _qs, indexing="ij")

        _prob_q = jnp.exp(self._log_prob_q_unnorm(_m1_grid, qs_grid))

        self._Z_q_given_m1 = jnp.trapezoid(_prob_q, _qs, axis=1)

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    def _log_prob_m1_unnorm_component(self, m1: Array) -> Array:
        safe_delta = jnp.where(self.delta_m1 <= 0.0, 1.0, self.delta_m1)
        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / safe_delta)
        log_mbreak = jnp.log(self.mbreak)
        log_m1 = jnp.log(m1)
        log_norm_bpl = jnp.logaddexp(
            self.alpha1 * log_mbreak
            + doubly_truncated_power_law_log_norm_constant(
                -self.alpha1, self.m1min, self.mbreak
            ),
            self.alpha2 * log_mbreak
            + doubly_truncated_power_law_log_norm_constant(
                -self.alpha2, self.mbreak, self.mmax
            ),
        )
        log_prob_bpl_1 = jnp.where(
            m1 < self.mbreak, self.alpha1 * (log_mbreak - log_m1), -jnp.inf
        )
        log_prob_bpl_2 = jnp.where(
            m1 < self.mbreak, -jnp.inf, self.alpha2 * (log_mbreak - log_m1)
        )
        log_prob_norm_0 = truncnorm.logpdf(
            m1,
            a=(self.m1min - self.loc1) / self.scale1,
            b=(self.mmax - self.loc1) / self.scale1,
            loc=self.loc1,
            scale=self.scale1,
        )
        log_prob_norm_1 = truncnorm.logpdf(
            m1,
            a=(self.m1min - self.loc2) / self.scale2,
            b=(self.mmax - self.loc2) / self.scale2,
            loc=self.loc2,
            scale=self.scale2,
        )

        log_prob_m1_component = jnp.asarray(
            [
                jnp.log(self.lambda_0)
                + log_prob_bpl_1
                - log_norm_bpl
                + log_smoothing_m1,
                jnp.log(self.lambda_0)
                + log_prob_bpl_2
                - log_norm_bpl
                + log_smoothing_m1,
                jnp.log(self.lambda_1) + log_prob_norm_0 + log_smoothing_m1,
                jnp.log1p(-self.lambda_0 - self.lambda_1)
                + log_prob_norm_1
                + log_smoothing_m1,
            ]
        )

        return jnp.where(
            (self.delta_m1 <= 0.0) | (m1 < self.m1min),
            -jnp.inf,
            log_prob_m1_component,
        )

    def _log_prob_q_unnorm(self, m1: Array, q: Array) -> ArrayLike:
        safe_delta = jnp.where(self.delta_m2 <= 0.0, 1.0, self.delta_m2)
        log_smoothing_q = log_planck_taper_window((m1 * q - self.m2min) / safe_delta)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_q
        return jnp.where(
            (self.delta_m2 <= 0.0) | (m1 * q < self.m2min), -jnp.inf, log_prob_q
        )

    def _log_prob_a1_components(self, a1: ArrayLike) -> ArrayLike:
        comp_bpl1 = truncnorm.logpdf(
            a1,
            a=(0.0 - self.a_1_loc_bpl1) / self.a_1_scale_bpl1,
            b=(1.0 - self.a_1_loc_bpl1) / self.a_1_scale_bpl1,
            loc=self.a_1_loc_bpl1,
            scale=self.a_1_scale_bpl1,
        )
        comp_bpl2 = truncnorm.logpdf(
            a1,
            a=(0.0 - self.a_1_loc_bpl2) / self.a_1_scale_bpl2,
            b=(1.0 - self.a_1_loc_bpl2) / self.a_1_scale_bpl2,
            loc=self.a_1_loc_bpl2,
            scale=self.a_1_scale_bpl2,
        )
        comp_n1 = truncnorm.logpdf(
            a1,
            a=(0.0 - self.a_1_loc_n1) / self.a_1_scale_n1,
            b=(1.0 - self.a_1_loc_n1) / self.a_1_scale_n1,
            loc=self.a_1_loc_n1,
            scale=self.a_1_scale_n1,
        )
        comp_n2 = truncnorm.logpdf(
            a1,
            a=(0.0 - self.a_1_loc_n2) / self.a_1_scale_n2,
            b=(1.0 - self.a_1_loc_n2) / self.a_1_scale_n2,
            loc=self.a_1_loc_n2,
            scale=self.a_1_scale_n2,
        )
        return jnp.asarray([comp_bpl1, comp_bpl2, comp_n1, comp_n2])

    def _log_prob_a2_components(self, a2: ArrayLike) -> ArrayLike:
        comp_bpl1 = truncnorm.logpdf(
            a2,
            a=(0.0 - self.a_2_loc_bpl1) / self.a_2_scale_bpl1,
            b=(1.0 - self.a_2_loc_bpl1) / self.a_2_scale_bpl1,
            loc=self.a_2_loc_bpl1,
            scale=self.a_2_scale_bpl1,
        )
        comp_bpl2 = truncnorm.logpdf(
            a2,
            a=(0.0 - self.a_2_loc_bpl2) / self.a_2_scale_bpl2,
            b=(1.0 - self.a_2_loc_bpl2) / self.a_2_scale_bpl2,
            loc=self.a_2_loc_bpl2,
            scale=self.a_2_scale_bpl2,
        )
        comp_n1 = truncnorm.logpdf(
            a2,
            a=(0.0 - self.a_2_loc_n1) / self.a_2_scale_n1,
            b=(1.0 - self.a_2_loc_n1) / self.a_2_scale_n1,
            loc=self.a_2_loc_n1,
            scale=self.a_2_scale_n1,
        )
        comp_n2 = truncnorm.logpdf(
            a2,
            a=(0.0 - self.a_2_loc_n2) / self.a_2_scale_n2,
            b=(1.0 - self.a_2_loc_n2) / self.a_2_scale_n2,
            loc=self.a_2_loc_n2,
            scale=self.a_2_scale_n2,
        )
        return jnp.asarray([comp_bpl1, comp_bpl2, comp_n1, comp_n2])

    def _log_prob_t1_t2_components(self, t1: ArrayLike, t2: ArrayLike) -> ArrayLike:
        hyper_params = [
            (
                self.cos_tilt_zeta_bpl1,
                self.cos_tilt_1_loc_bpl1,
                self.cos_tilt_1_scale_bpl1,
                self.cos_tilt_2_loc_bpl1,
                self.cos_tilt_2_scale_bpl1,
            ),
            (
                self.cos_tilt_zeta_bpl2,
                self.cos_tilt_1_loc_bpl2,
                self.cos_tilt_1_scale_bpl2,
                self.cos_tilt_2_loc_bpl2,
                self.cos_tilt_2_scale_bpl2,
            ),
            (
                self.cos_tilt_zeta_n1,
                self.cos_tilt_1_loc_n1,
                self.cos_tilt_1_scale_n1,
                self.cos_tilt_2_loc_n1,
                self.cos_tilt_2_scale_n1,
            ),
            (
                self.cos_tilt_zeta_n2,
                self.cos_tilt_1_loc_n2,
                self.cos_tilt_1_scale_n2,
                self.cos_tilt_2_loc_n2,
                self.cos_tilt_2_scale_n2,
            ),
        ]

        comp_log_probs = []

        for zeta, loc1, scale1, loc2, scale2 in hyper_params:
            comp_gaussian = (
                jnp.log(zeta)
                + truncnorm.logpdf(
                    t1,
                    a=(-1.0 - loc1) / scale1,
                    b=(1.0 - loc1) / scale1,
                    loc=loc1,
                    scale=scale1,
                )
                + truncnorm.logpdf(
                    t2,
                    a=(-1.0 - loc2) / scale2,
                    b=(1.0 - loc2) / scale2,
                    loc=loc2,
                    scale=scale2,
                )
            )
            comp_uniform = jnp.log1p(-zeta) + jnp.log(0.25)
            comp_log_prob = jnp.logaddexp(comp_gaussian, comp_uniform)
            comp_log_probs.append(comp_log_prob)

        return jnp.asarray(comp_log_probs)

    @validate_sample
    def log_prob(self, value: Array) -> ArrayLike:
        m1, m2, a1, a2, t1, t2 = jnp.unstack(value, axis=-1)
        log_prob_m1_component = self._log_prob_m1_unnorm_component(m1)
        log_prob_a1_component = self._log_prob_a1_components(a1)
        log_prob_a2_component = self._log_prob_a2_components(a2)
        log_prob_t1_t2_component = self._log_prob_t1_t2_components(t1, t2)

        log_prob_m1_a1_a2_t1_t2_component = (
            log_prob_m1_component
            + log_prob_a1_component
            + log_prob_a2_component
            + log_prob_t1_t2_component
        )

        log_prob_m1_a1_a2_t1_t2 = jax.nn.logsumexp(
            log_prob_m1_a1_a2_t1_t2_component,
            where=~jnp.isneginf(log_prob_m1_a1_a2_t1_t2_component),
            axis=0,
        )

        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1, left=1.0, right=1.0)
        safe_Z_q = jnp.where(_Z_q <= 0, 1.0, _Z_q)
        log_Z_q = jnp.where(_Z_q <= 0, 0.0, jnp.log(safe_Z_q))
        log_prob_q = self._log_prob_q_unnorm(m1, m2 / m1) - log_Z_q

        return jnp.log(m1) + log_prob_m1_a1_a2_t1_t2 + log_prob_q - self._logZ


def BrokenPowerlawTwoPeakMultiSpinMultiTiltFull(
    use_eccentricity: bool = False,
    validate_args: Optional[bool] = None,
    **params: Array,
) -> ScaledMixture:
    smoothing_model = BrokenPowerlawTwoPeakMultiSpinMultiTilt(
        alpha1=params["alpha1"],
        alpha2=params["alpha2"],
        beta=params["beta"],
        loc1=params["loc1"],
        loc2=params["loc2"],
        scale1=params["scale1"],
        scale2=params["scale2"],
        delta_m1=params["delta_m1"],
        delta_m2=params["delta_m2"],
        lambda_0=params["lambda_0"],
        lambda_1=params["lambda_1"],
        m1min=params["m1min"],
        m2min=params["m2min"],
        mmax=params["mmax"],
        mbreak=params["mbreak"],
        a_1_loc_bpl1=params["a_1_loc_bpl1"],
        a_1_loc_bpl2=params["a_1_loc_bpl2"],
        a_1_loc_n1=params["a_1_loc_n1"],
        a_1_loc_n2=params["a_1_loc_n2"],
        a_2_loc_bpl1=params["a_2_loc_bpl1"],
        a_2_loc_bpl2=params["a_2_loc_bpl2"],
        a_2_loc_n1=params["a_2_loc_n1"],
        a_2_loc_n2=params["a_2_loc_n2"],
        a_1_scale_bpl1=params["a_1_scale_bpl1"],
        a_1_scale_bpl2=params["a_1_scale_bpl2"],
        a_1_scale_n1=params["a_1_scale_n1"],
        a_1_scale_n2=params["a_1_scale_n2"],
        a_2_scale_bpl1=params["a_2_scale_bpl1"],
        a_2_scale_bpl2=params["a_2_scale_bpl2"],
        a_2_scale_n1=params["a_2_scale_n1"],
        a_2_scale_n2=params["a_2_scale_n2"],
        cos_tilt_1_loc_bpl1=params["cos_tilt_1_loc_bpl1"],
        cos_tilt_1_loc_bpl2=params["cos_tilt_1_loc_bpl2"],
        cos_tilt_1_loc_n1=params["cos_tilt_1_loc_n1"],
        cos_tilt_1_loc_n2=params["cos_tilt_1_loc_n2"],
        cos_tilt_1_scale_bpl1=params["cos_tilt_1_scale_bpl1"],
        cos_tilt_1_scale_bpl2=params["cos_tilt_1_scale_bpl2"],
        cos_tilt_1_scale_n1=params["cos_tilt_1_scale_n1"],
        cos_tilt_1_scale_n2=params["cos_tilt_1_scale_n2"],
        cos_tilt_2_loc_bpl1=params["cos_tilt_2_loc_bpl1"],
        cos_tilt_2_loc_bpl2=params["cos_tilt_2_loc_bpl2"],
        cos_tilt_2_loc_n1=params["cos_tilt_2_loc_n1"],
        cos_tilt_2_loc_n2=params["cos_tilt_2_loc_n2"],
        cos_tilt_2_scale_bpl1=params["cos_tilt_2_scale_bpl1"],
        cos_tilt_2_scale_bpl2=params["cos_tilt_2_scale_bpl2"],
        cos_tilt_2_scale_n1=params["cos_tilt_2_scale_n1"],
        cos_tilt_2_scale_n2=params["cos_tilt_2_scale_n2"],
        cos_tilt_zeta_bpl1=params["cos_tilt_zeta_bpl1"],
        cos_tilt_zeta_bpl2=params["cos_tilt_zeta_bpl2"],
        cos_tilt_zeta_n1=params["cos_tilt_zeta_n1"],
        cos_tilt_zeta_n2=params["cos_tilt_zeta_n2"],
        validate_args=validate_args,
    )

    component_distributions = [smoothing_model]

    if use_eccentricity:
        ecc_dist = HalfNormal(
            scale=params["eccentricity_scale"], validate_args=validate_args
        )
        component_distributions.append(ecc_dist)

    z_max = params["z_max"]
    kappa = params["kappa"]
    powerlaw_z = PowerlawRedshift(z_max=z_max, kappa=kappa, validate_args=validate_args)
    component_distributions.append(powerlaw_z)

    component_distributions = [
        JointDistribution(*component_distributions, validate_args=validate_args)
    ]

    return ScaledMixture(
        log_scales=jnp.asarray([params["log_rate"]]),
        component_distributions=component_distributions,
        support=component_distributions[0]._support,  # type: ignore
        validate_args=validate_args,
    )
