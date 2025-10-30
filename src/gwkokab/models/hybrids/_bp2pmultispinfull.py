# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import lax, numpy as jnp
from jax.scipy.stats import truncnorm
from jaxtyping import Array, ArrayLike
from numpyro.distributions import (
    constraints,
    Distribution,
    HalfNormal,
)
from numpyro.distributions.util import promote_shapes, validate_sample

from ...utils.kernel import log_planck_taper_window
from ..constraints import all_constraint, mass_sandwich
from ..redshift import PowerlawRedshift
from ..spin import MinimumTiltModel
from ..utils import (
    doubly_truncated_power_law_log_norm_constant,
    JointDistribution,
    ScaledMixture,
)


@jax.jit
def _broken_powerlaw_prob(
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
    log_mbreak = jnp.log(mbreak)
    log_m1 = jnp.log(m1)
    log_unnormalized = jnp.where(
        m1 < mbreak,
        -alpha1 * (log_m1 - log_mbreak),
        -alpha2 * (log_m1 - log_mbreak),
    )
    log_norm = jnp.logaddexp(
        alpha1 * log_mbreak
        + doubly_truncated_power_law_log_norm_constant(-alpha1, mmin, mbreak),
        alpha2 * log_mbreak
        + doubly_truncated_power_law_log_norm_constant(-alpha2, mbreak, mmax),
    )
    return jnp.exp(log_unnormalized - log_norm)


class BrokenPowerlawTwoPeakMultiSpin(Distribution):
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
        "loc_a1_bpl": constraints.unit_interval,
        "loc_a1_n1": constraints.unit_interval,
        "loc_a1_n2": constraints.unit_interval,
        "loc_a2_bpl": constraints.unit_interval,
        "loc_a2_n1": constraints.unit_interval,
        "loc_a2_n2": constraints.unit_interval,
        "scale_a1_bpl": constraints.positive,
        "scale_a1_n1": constraints.positive,
        "scale_a1_n2": constraints.positive,
        "scale_a2_bpl": constraints.positive,
        "scale_a2_n1": constraints.positive,
        "scale_a2_n2": constraints.positive,
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
        "loc_a1_bpl1",
        "loc_a1_bpl2",
        "loc_a1_n1",
        "loc_a1_n2",
        "loc_a2_bpl1",
        "loc_a2_bpl2",
        "loc_a2_n1",
        "loc_a2_n2",
        "scale_a1_bpl1",
        "scale_a1_bpl2",
        "scale_a1_n1",
        "scale_a1_n2",
        "scale_a2_bpl1",
        "scale_a2_bpl2",
        "scale_a2_n1",
        "scale_a2_n2",
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
        loc_a1_bpl1: ArrayLike,
        loc_a1_bpl2: ArrayLike,
        loc_a1_n1: ArrayLike,
        loc_a1_n2: ArrayLike,
        loc_a2_bpl1: ArrayLike,
        loc_a2_bpl2: ArrayLike,
        loc_a2_n1: ArrayLike,
        loc_a2_n2: ArrayLike,
        scale_a1_bpl1: ArrayLike,
        scale_a1_bpl2: ArrayLike,
        scale_a1_n1: ArrayLike,
        scale_a1_n2: ArrayLike,
        scale_a2_bpl1: ArrayLike,
        scale_a2_bpl2: ArrayLike,
        scale_a2_n1: ArrayLike,
        scale_a2_n2: ArrayLike,
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
            self.loc_a1_bpl1,
            self.loc_a1_bpl2,
            self.loc_a1_n1,
            self.loc_a1_n2,
            self.loc_a2_bpl1,
            self.loc_a2_bpl2,
            self.loc_a2_n1,
            self.loc_a2_n2,
            self.scale_a1_bpl1,
            self.scale_a1_bpl2,
            self.scale_a1_n1,
            self.scale_a1_n2,
            self.scale_a2_bpl1,
            self.scale_a2_bpl2,
            self.scale_a2_n1,
            self.scale_a2_n2,
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
            loc_a1_bpl1,
            loc_a1_bpl2,
            loc_a1_n1,
            loc_a1_n2,
            loc_a2_bpl1,
            loc_a2_bpl2,
            loc_a2_n1,
            loc_a2_n2,
            scale_a1_bpl1,
            scale_a1_bpl2,
            scale_a1_n1,
            scale_a1_n2,
            scale_a2_bpl1,
            scale_a2_bpl2,
            scale_a2_n1,
            scale_a2_n2,
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
            jnp.shape(loc_a1_bpl1),
            jnp.shape(loc_a1_bpl2),
            jnp.shape(loc_a1_n1),
            jnp.shape(loc_a1_n2),
            jnp.shape(loc_a2_bpl1),
            jnp.shape(loc_a2_bpl2),
            jnp.shape(loc_a2_n1),
            jnp.shape(loc_a2_n2),
            jnp.shape(scale_a1_bpl1),
            jnp.shape(scale_a1_bpl2),
            jnp.shape(scale_a1_n1),
            jnp.shape(scale_a1_n2),
            jnp.shape(scale_a2_bpl1),
            jnp.shape(scale_a2_bpl2),
            jnp.shape(scale_a2_n1),
            jnp.shape(scale_a2_n2),
        )

        self._support = all_constraint(
            [
                mass_sandwich(m2min, mmax),
                constraints.unit_interval,
                constraints.unit_interval,
            ],
            [(0, 2), 2, 3],
        )
        super(BrokenPowerlawTwoPeakMultiSpin, self).__init__(
            batch_shape=batch_shape, event_shape=(4,), validate_args=validate_args
        )

        m1min = jnp.broadcast_to(m1min, batch_shape)
        m2min = jnp.broadcast_to(m2min, batch_shape)
        mmax = jnp.broadcast_to(mmax, batch_shape)

        m1s = jnp.linspace(m1min, mmax, 1000)
        self._logZ = jnp.log(
            jnp.trapezoid(self._prob_m1_unnorm_component(m1s).sum(axis=0), m1s, axis=0)
        )

        self._m1s = jnp.linspace(m2min, mmax, 1000)
        _qs = jnp.linspace(0.005, 1.0, 500)
        _m1_grid, qs_grid = jnp.meshgrid(self._m1s, _qs, indexing="ij")

        _prob_q = jnp.exp(self._log_prob_q_unnorm(_m1_grid, qs_grid))

        self._Z_q_given_m1 = jnp.trapezoid(_prob_q, _qs, axis=1)

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        return self._support

    def _prob_m1_unnorm_component(self, m1: Array) -> Array:
        safe_delta = jnp.where(self.delta_m1 <= 0.0, 1.0, self.delta_m1)
        smoothing_m1 = jnp.exp(log_planck_taper_window((m1 - self.m1min) / safe_delta))
        log_mbreak = jnp.log(self.mbreak)
        norm = jnp.exp(
            jnp.logaddexp(
                self.alpha1 * log_mbreak
                + doubly_truncated_power_law_log_norm_constant(
                    -self.alpha1, self.m1min, self.mbreak
                ),
                self.alpha2 * log_mbreak
                + doubly_truncated_power_law_log_norm_constant(
                    -self.alpha2, self.mbreak, self.mmax
                ),
            )
        )
        prob_bpl_1 = jnp.where(
            m1 < self.mbreak, jnp.power(m1 / self.mbreak, -self.alpha1) / norm, 0.0
        )
        prob_bpl_2 = jnp.where(
            m1 < self.mbreak, 0.0, jnp.power(m1 / self.mbreak, -self.alpha2) / norm
        )
        prob_norm_0 = truncnorm.pdf(
            m1,
            a=(self.m1min - self.loc1) / self.scale1,
            b=(self.mmax - self.loc1) / self.scale1,
            loc=self.loc1,
            scale=self.scale1,
        )
        prob_norm_1 = truncnorm.pdf(
            m1,
            a=(self.m1min - self.loc2) / self.scale2,
            b=(self.mmax - self.loc2) / self.scale2,
            loc=self.loc2,
            scale=self.scale2,
        )
        lambda_2 = 1.0 - self.lambda_0 - self.lambda_1

        log_prob_m1_component = jnp.asarray(
            [
                self.lambda_0 * prob_bpl_1 * smoothing_m1,
                self.lambda_0 * prob_bpl_2 * smoothing_m1,
                self.lambda_1 * prob_norm_0 * smoothing_m1,
                lambda_2 * prob_norm_1 * smoothing_m1,
            ]
        )

        return log_prob_m1_component

    def _log_prob_q_unnorm(self, m1: Array, m2: Array) -> ArrayLike:
        safe_delta = jnp.where(self.delta_m2 <= 0.0, 1.0, self.delta_m2)
        log_smoothing_q = log_planck_taper_window((m2 - self.m2min) / safe_delta)
        log_prob_q = self.beta * (jnp.log(m2) - jnp.log(m1)) + log_smoothing_q

        return jnp.where(
            (self.delta_m2 <= 0.0) | (m2 < self.m2min), -jnp.inf, log_prob_q
        )

    def _prob_a_1_components(self, a_1: ArrayLike) -> ArrayLike:
        comp_bpl1 = truncnorm.pdf(
            a_1,
            a=(0.0 - self.loc_a1_bpl1) / self.scale_a1_bpl1,
            b=(1.0 - self.loc_a1_bpl1) / self.scale_a1_bpl1,
            loc=self.loc_a1_bpl1,
            scale=self.scale_a1_bpl1,
        )
        comp_bpl2 = truncnorm.pdf(
            a_1,
            a=(0.0 - self.loc_a1_bpl2) / self.scale_a1_bpl2,
            b=(1.0 - self.loc_a1_bpl2) / self.scale_a1_bpl2,
            loc=self.loc_a1_bpl2,
            scale=self.scale_a1_bpl2,
        )
        comp_n1 = truncnorm.pdf(
            a_1,
            a=(0.0 - self.loc_a1_n1) / self.scale_a1_n1,
            b=(1.0 - self.loc_a1_n1) / self.scale_a1_n1,
            loc=self.loc_a1_n1,
            scale=self.scale_a1_n1,
        )
        comp_n2 = truncnorm.pdf(
            a_1,
            a=(0.0 - self.loc_a1_n2) / self.scale_a1_n2,
            b=(1.0 - self.loc_a1_n2) / self.scale_a1_n2,
            loc=self.loc_a1_n2,
            scale=self.scale_a1_n2,
        )
        return jnp.asarray([comp_bpl1, comp_bpl2, comp_n1, comp_n2])

    def _prob_a_2_components(self, a_2: ArrayLike) -> ArrayLike:
        comp_bpl1 = truncnorm.pdf(
            a_2,
            a=(0.0 - self.loc_a2_bpl1) / self.scale_a2_bpl1,
            b=(1.0 - self.loc_a2_bpl1) / self.scale_a2_bpl1,
            loc=self.loc_a2_bpl1,
            scale=self.scale_a2_bpl1,
        )
        comp_bpl2 = truncnorm.pdf(
            a_2,
            a=(0.0 - self.loc_a2_bpl2) / self.scale_a2_bpl2,
            b=(1.0 - self.loc_a2_bpl2) / self.scale_a2_bpl2,
            loc=self.loc_a2_bpl2,
            scale=self.scale_a2_bpl2,
        )
        comp_n1 = truncnorm.pdf(
            a_2,
            a=(0.0 - self.loc_a2_n1) / self.scale_a2_n1,
            b=(1.0 - self.loc_a2_n1) / self.scale_a2_n1,
            loc=self.loc_a2_n1,
            scale=self.scale_a2_n1,
        )
        comp_n2 = truncnorm.pdf(
            a_2,
            a=(0.0 - self.loc_a2_n2) / self.scale_a2_n2,
            b=(1.0 - self.loc_a2_n2) / self.scale_a2_n2,
            loc=self.loc_a2_n2,
            scale=self.scale_a2_n2,
        )
        return jnp.asarray([comp_bpl1, comp_bpl2, comp_n1, comp_n2])

    @validate_sample
    def log_prob(self, value: Array) -> ArrayLike:
        m1, m2, a_1, a_2 = jnp.unstack(value, axis=-1)
        prob_m1_component = self._prob_m1_unnorm_component(m1)
        prob_a1_component = self._prob_a_1_components(a_1)
        prob_a2_component = self._prob_a_2_components(a_2)

        log_prob_m1_a1_a2 = jnp.log(
            (prob_m1_component * prob_a1_component * prob_a2_component).sum(axis=0)
        )

        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1, left=1.0, right=1.0)
        safe_Z_q = jnp.where(_Z_q <= 0, 1.0, _Z_q)
        log_Z_q = jnp.where(_Z_q <= 0, 0.0, jnp.log(safe_Z_q))
        log_prob_q = self._log_prob_q_unnorm(m1, m2) - log_Z_q

        return jnp.log(m1) + log_prob_m1_a1_a2 + log_prob_q - self._logZ


def BrokenPowerlawTwoPeakMultiSpinFull(
    use_eccentricity: bool = False,
    validate_args: Optional[bool] = None,
    **params: Array,
) -> ScaledMixture:
    smoothing_model = BrokenPowerlawTwoPeakMultiSpin(
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
        loc_a1_bpl1=params["loc_a1_bpl1"],
        loc_a1_bpl2=params["loc_a1_bpl2"],
        loc_a1_n1=params["loc_a1_n1"],
        loc_a1_n2=params["loc_a1_n2"],
        loc_a2_bpl1=params["loc_a2_bpl1"],
        loc_a2_bpl2=params["loc_a2_bpl2"],
        loc_a2_n1=params["loc_a2_n1"],
        loc_a2_n2=params["loc_a2_n2"],
        scale_a1_bpl1=params["scale_a1_bpl1"],
        scale_a1_bpl2=params["scale_a1_bpl2"],
        scale_a1_n1=params["scale_a1_n1"],
        scale_a1_n2=params["scale_a1_n2"],
        scale_a2_bpl1=params["scale_a2_bpl1"],
        scale_a2_bpl2=params["scale_a2_bpl2"],
        scale_a2_n1=params["scale_a2_n1"],
        scale_a2_n2=params["scale_a2_n2"],
        validate_args=validate_args,
    )

    component_distributions = [smoothing_model]

    tilt_dist = MinimumTiltModel(
        zeta=params["cos_tilt_zeta"],
        loc=params["cos_tilt_loc"],
        scale=params["cos_tilt_scale"],
        minimum=params.get("cos_tilt_minimum", -1.0),
        validate_args=validate_args,
    )
    component_distributions.append(tilt_dist)

    if use_eccentricity:
        ecc_dist = HalfNormal(
            scale=params["eccentricity_scale"], validate_args=validate_args
        )
        component_distributions.append(ecc_dist)

    z_max = params["z_max"]
    kappa = params["kappa"]
    powerlaw_z = PowerlawRedshift(z_max=z_max, kappa=kappa, validate_args=validate_args)
    component_distributions.append(powerlaw_z)

    if len(component_distributions) > 1:
        component_distributions = [
            JointDistribution(*component_distributions, validate_args=validate_args)
        ]

    return ScaledMixture(
        log_scales=jnp.asarray([params["log_rate"]]),
        component_distributions=component_distributions,
        support=component_distributions[0]._support,  # type: ignore
        validate_args=validate_args,
    )
