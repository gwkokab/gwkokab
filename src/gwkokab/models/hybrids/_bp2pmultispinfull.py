# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import jax
from jax import lax, numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints, Distribution, HalfNormal
from numpyro.distributions.util import promote_shapes, validate_sample

from ...utils.kernel import log_planck_taper_window
from ...utils.math import truncnorm_logpdf
from ..constraints import all_constraint, mass_sandwich
from ..redshift import PowerlawRedshift
from ..utils import (
    doubly_truncated_power_law_log_prob,
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
        "m1min": constraints.positive,
        "m2min": constraints.positive,
        "mbreak": constraints.positive,
        "mmax": constraints.positive,
        "scale1": constraints.positive,
        "scale2": constraints.positive,
        "_a1_locs": constraints.unit_interval,
        "_a1_scales": constraints.positive,
        "_a2_locs": constraints.unit_interval,
        "_a2_scales": constraints.positive,
        "_tilt_zetas": constraints.unit_interval,
        "_tilt1_locs": constraints.interval(-1.0, 1.0),
        "_tilt1_scales": constraints.positive,
        "_tilt2_locs": constraints.interval(-1.0, 1.0),
        "_tilt2_scales": constraints.positive,
    }

    pytree_data_fields = (
        "_logZ",
        "_m1s",
        "_support",
        "_log_Z_q_given_m1",
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
        "_a1_locs",
        "_a1_scales",
        "_a2_locs",
        "_a2_scales",
        "_tilt_zetas",
        "_tilt1_locs",
        "_tilt1_scales",
        "_tilt2_locs",
        "_tilt2_scales",
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

        m1min_bc = jnp.broadcast_to(m1min, batch_shape)
        m2min_bc = jnp.broadcast_to(m2min, batch_shape)
        mmax_bc = jnp.broadcast_to(mmax, batch_shape)

        # OPTIMIZATION 2: Reduce grid size (1000 -> 400)
        m1s = jnp.linspace(m1min_bc, mmax_bc, 1000)

        # Vectorized computation
        log_probs_m1_comp = jax.vmap(self._log_prob_m1_unnorm_component, in_axes=0)(m1s)

        self._logZ = jnp.log(
            jnp.trapezoid(
                jnp.nan_to_num(jnp.exp(log_probs_m1_comp).sum(axis=-1)),
                m1s,
                axis=0,
            )
            * (delta_m1 != 0.0)
            + 1.0 * (delta_m1 == 0.0)
        )

        self._m1s = jnp.linspace(m2min_bc, mmax_bc, 1000)
        _qs = jnp.linspace(0.01, 1.0, 500)

        _log_prob_q = jax.vmap(
            jax.vmap(self._log_prob_q_unnorm, in_axes=(None, 0)), in_axes=(0, None)
        )(self._m1s, _qs)

        _prob_q = jnp.exp(_log_prob_q) * (delta_m2 != 0.0) + 1.0 * (delta_m2 == 0.0)

        _Z_q_given_m1 = jnp.nan_to_num(jnp.trapezoid(_prob_q, _qs, axis=1))

        safe_Z_q = jnp.where(_Z_q_given_m1 <= 0, 1.0, _Z_q_given_m1)
        self._log_Z_q_given_m1 = jnp.where(_Z_q_given_m1 <= 0, 0.0, jnp.log(safe_Z_q))

        self._a1_locs = jnp.stack([a_1_loc_bpl1, a_1_loc_bpl2, a_1_loc_n1, a_1_loc_n2])
        self._a1_scales = jnp.stack(
            [
                a_1_scale_bpl1,
                a_1_scale_bpl2,
                a_1_scale_n1,
                a_1_scale_n2,
            ]
        )
        self._a2_locs = jnp.stack([a_2_loc_bpl1, a_2_loc_bpl2, a_2_loc_n1, a_2_loc_n2])
        self._a2_scales = jnp.stack(
            [
                a_2_scale_bpl1,
                a_2_scale_bpl2,
                a_2_scale_n1,
                a_2_scale_n2,
            ]
        )
        self._tilt_zetas = jnp.stack(
            [
                cos_tilt_zeta_bpl1,
                cos_tilt_zeta_bpl2,
                cos_tilt_zeta_n1,
                cos_tilt_zeta_n2,
            ]
        )
        self._tilt1_locs = jnp.stack(
            [
                cos_tilt_1_loc_bpl1,
                cos_tilt_1_loc_bpl2,
                cos_tilt_1_loc_n1,
                cos_tilt_1_loc_n2,
            ]
        )
        self._tilt1_scales = jnp.stack(
            [
                cos_tilt_1_scale_bpl1,
                cos_tilt_1_scale_bpl2,
                cos_tilt_1_scale_n1,
                cos_tilt_1_scale_n2,
            ]
        )
        self._tilt2_locs = jnp.stack(
            [
                cos_tilt_2_loc_bpl1,
                cos_tilt_2_loc_bpl2,
                cos_tilt_2_loc_n1,
                cos_tilt_2_loc_n2,
            ]
        )
        self._tilt2_scales = jnp.stack(
            [
                cos_tilt_2_scale_bpl1,
                cos_tilt_2_scale_bpl2,
                cos_tilt_2_scale_n1,
                cos_tilt_2_scale_n2,
            ]
        )

        super(BrokenPowerlawTwoPeakMultiSpinMultiTilt, self).__init__(
            batch_shape=batch_shape, event_shape=(6,), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    def _log_prob_m1_unnorm_component(self, m1: Array) -> Array:
        invalid = (self.delta_m1 <= 0.0) | (m1 < self.m1min)

        safe_delta = jnp.maximum(self.delta_m1, 1e-10)
        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / safe_delta)

        log_mbreak = jnp.log(self.mbreak)
        log_m1 = jnp.log(m1)

        log_norm_bpl = jnp.logaddexp(
            -doubly_truncated_power_law_log_prob(
                self.mbreak, alpha=-self.alpha1, low=self.m1min, high=self.mbreak
            ),
            -doubly_truncated_power_law_log_prob(
                self.mbreak, alpha=-self.alpha2, low=self.mbreak, high=self.mmax
            ),
        )

        log_prob_bpl_1 = jnp.where(
            m1 < self.mbreak, self.alpha1 * (log_mbreak - log_m1), -jnp.inf
        )
        log_prob_bpl_2 = jnp.where(
            m1 >= self.mbreak, self.alpha2 * (log_mbreak - log_m1), -jnp.inf
        )

        log_prob_norm_0 = truncnorm_logpdf(
            m1,
            loc=self.loc1,
            scale=self.scale1,
            low=self.m1min,
            high=self.mmax,
        )
        log_prob_norm_1 = truncnorm_logpdf(
            m1,
            loc=self.loc2,
            scale=self.scale2,
            low=self.m1min,
            high=self.mmax,
        )

        log_prob_m1_component = jnp.array(
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
                jnp.log1p(-(self.lambda_0 + self.lambda_1))
                + log_prob_norm_1
                + log_smoothing_m1,
            ]
        )

        return jnp.where(invalid, -jnp.inf, log_prob_m1_component)

    def _log_prob_q_unnorm(self, m1: Array, q: Array) -> ArrayLike:
        safe_delta = jnp.maximum(self.delta_m2, 1e-10)
        log_smoothing_q = log_planck_taper_window((m1 * q - self.m2min) / safe_delta)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_q
        return jnp.where(
            (self.delta_m2 <= 0.0) | (m1 * q < self.m2min), -jnp.inf, log_prob_q
        )

    def _log_prob_a1_components(self, a1: ArrayLike) -> ArrayLike:
        return jax.vmap(
            lambda loc, scale: truncnorm_logpdf(
                a1, loc=loc, scale=scale, low=0.0, high=1.0
            )
        )(self._a1_locs, self._a1_scales)

    def _log_prob_a2_components(self, a2: ArrayLike) -> ArrayLike:
        return jax.vmap(
            lambda loc, scale: truncnorm_logpdf(
                a2, loc=loc, scale=scale, low=0.0, high=1.0
            )
        )(self._a2_locs, self._a2_scales)

    def _log_prob_t1_t2_components(self, t1: ArrayLike, t2: ArrayLike) -> ArrayLike:
        def compute_comp(zeta, loc1, scale1, loc2, scale2):
            comp_gaussian = zeta * jnp.exp(
                truncnorm_logpdf(t1, loc=loc1, scale=scale1, low=-1.0, high=1.0)
                + truncnorm_logpdf(t2, loc=loc2, scale=scale2, low=-1.0, high=1.0)
            )
            comp_uniform = (1.0 - zeta) * 0.25
            return jnp.log(comp_gaussian + comp_uniform)

        return jax.vmap(compute_comp)(
            self._tilt_zetas,
            self._tilt1_locs,
            self._tilt1_scales,
            self._tilt2_locs,
            self._tilt2_scales,
        )

    @validate_sample
    def log_prob(self, value: Array) -> ArrayLike:
        m1, m2, a1, a2, t1, t2 = jnp.unstack(value, axis=-1)

        # Compute all components
        log_prob_m1_component = self._log_prob_m1_unnorm_component(m1)
        log_prob_a1_component = self._log_prob_a1_components(a1)
        log_prob_a2_component = self._log_prob_a2_components(a2)
        log_prob_t1_t2_component = self._log_prob_t1_t2_components(t1, t2)

        # Sum all components
        log_prob_m1_a1_a2_t1_t2_component = (
            log_prob_m1_component
            + log_prob_a1_component
            + log_prob_a2_component
            + log_prob_t1_t2_component
        )

        # OPTIMIZATION 8: More efficient logsumexp with where
        log_prob_m1_a1_a2_t1_t2 = jax.nn.logsumexp(
            log_prob_m1_a1_a2_t1_t2_component,
            axis=0,
            b=~jnp.isneginf(log_prob_m1_a1_a2_t1_t2_component),
        )

        # OPTIMIZATION 9: Use precomputed log
        log_Z_q = jnp.interp(m1, self._m1s, self._log_Z_q_given_m1, left=0.0, right=0.0)
        log_prob_q = self._log_prob_q_unnorm(m1, m2 / m1) - log_Z_q

        return -jnp.log(m1) + log_prob_m1_a1_a2_t1_t2 + log_prob_q - self._logZ


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
