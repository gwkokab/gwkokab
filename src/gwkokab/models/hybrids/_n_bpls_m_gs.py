# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import validate_sample

from ...cosmology import PLANCK_2015_Cosmology
from ...utils.kernel import log_planck_taper_window
from ...utils.math import truncnorm_logpdf
from ..constraints import all_constraint, mass_sandwich
from ..mass._bp2p import _broken_powerlaw_log_prob


def _n_bpls_m_gs_component_log_prob(
    N_pl: int, N_g: int, mm: Array, **params: Array
) -> Array:
    _lambdas = [params[f"lambda_{i}"] for i in range(N_pl + N_g - 1)]
    _lambdas.append(jnp.asarray(1.0) - sum(_lambdas, start=jnp.asarray(0.0)))
    lambdas = jnp.stack(_lambdas, axis=-1)

    alpha1s = [params[f"alpha1_bpl_{i}"] for i in range(N_pl)]
    alpha2s = [params[f"alpha2_bpl_{i}"] for i in range(N_pl)]
    mmins_bpl = [params[f"m1min_bpl_{i}"] for i in range(N_pl)]
    mbreaks_bpl = [params[f"m1break_bpl_{i}"] for i in range(N_pl)]
    mmaxs_bpl = [params[f"m1max_bpl_{i}"] for i in range(N_pl)]

    locs = [params[f"loc_g_{i}"] for i in range(N_g)]
    scales = [params[f"scale_g_{i}"] for i in range(N_g)]
    mmins_g = [params[f"mmin_g_{i}"] for i in range(N_g)]
    mmaxs_g = [params[f"mmax_g_{i}"] for i in range(N_g)]

    powerlaws_log_prob = [
        _broken_powerlaw_log_prob(mm, alpha1, alpha2, mmin, mmax, mbreak)
        for alpha1, alpha2, mmin, mmax, mbreak in zip(
            alpha1s, alpha2s, mmins_bpl, mmaxs_bpl, mbreaks_bpl
        )
    ]

    gaussians_log_prob = [
        truncnorm_logpdf(mm, loc, scale, mmin, mmax)
        for loc, scale, mmin, mmax in zip(locs, scales, mmins_g, mmaxs_g)
    ]

    components_log_prob = jnp.stack(
        powerlaws_log_prob + gaussians_log_prob, axis=-1
    ) + jnp.log(lambdas)

    return components_log_prob


def NBrokenPowerlawMGaussian(
    N_pl: int,
    N_g: int,
    beta: Array,
    delta_m1: Array,
    delta_m2: Array,
    kappa: Array,
    log_rate: Array,
    m2min: Array,
    z_max: Array,
    m1min: Array,
    m1max: Array,
    validate_args: Optional[bool] = None,
    **params: Array,
) -> Distribution:
    mm = jnp.linspace(m1min, m1max, 1000)
    _n_bpls_m_gs_prob = jnp.exp(
        _n_bpls_m_gs_component_log_prob(
            N_pl,
            N_g,
            mm,
            m1min=m1min,
            **{
                k: v
                for k, v in params.items()
                if any(
                    (
                        k.startswith(name)
                        for name in (
                            "alpha1",
                            "alpha2",
                            "lambda",
                            "loc",
                            "m1break",
                            "m1max",
                            "m1min",
                            "scale",
                        )
                    )
                )
            },
        )
    ).sum(axis=-1) * jnp.exp(log_planck_taper_window((mm - m1min) / delta_m1))

    Z = jnp.trapezoid(_n_bpls_m_gs_prob, mm, axis=0)
    logZ = jnp.log(Z)

    qq = jnp.linspace(0.001, 1.0, 500)
    mm_grid, qq_grid = jnp.meshgrid(mm, qq, indexing="ij")

    m2 = mm_grid * qq_grid
    safe_delta = jnp.where(delta_m2 <= 0.0, 1.0, delta_m2)
    log_smoothing_q = log_planck_taper_window((m2 - m2min) / safe_delta)
    log_prob_q = beta * jnp.log(qq) + log_smoothing_q
    prob_q = jnp.where((delta_m2 <= 0.0) | (m2 < m2min), 0.0, jnp.exp(log_prob_q))
    _Z_q_given_m1 = jnp.trapezoid(prob_q, qq, axis=1)
    safe_Z_q_given_m1 = jnp.where(_Z_q_given_m1 <= 0, 1.0, _Z_q_given_m1)
    _log_Z_q = jnp.where(
        _Z_q_given_m1 <= 0, jnp.nan_to_num(-jnp.inf), jnp.log(safe_Z_q_given_m1)
    )

    class _NSmoothingPowerlawMGaussian(Distribution):
        arg_constraints = (
            {
                "_log_Z_q": constraints.real_vector,
                "beta": constraints.real,
                "delta_m1": constraints.positive,
                "delta_m2": constraints.positive,
                "kappa": constraints.real,
                "log_rate": constraints.real,
                "logZ": constraints.real,
                "m1max": constraints.positive,
                "m1min": constraints.positive,
                "m2min": constraints.positive,
                "z_max": constraints.positive,
            }
            | {f"alpha1_bpl_{i}": constraints.real for i in range(N_pl)}
            | {f"alpha2_bpl_{i}": constraints.real for i in range(N_pl)}
            | {f"loc_g_{i}": constraints.real for i in range(N_g)}
            | {f"m1max_g_{i}": constraints.positive for i in range(N_g)}
            | {f"m1max_bpl_{i}": constraints.positive for i in range(N_pl)}
            | {f"m1min_g_{i}": constraints.positive for i in range(N_g)}
            | {f"m1min_bpl_{i}": constraints.positive for i in range(N_pl)}
            | {f"m1break_bpl_{i}": constraints.positive for i in range(N_pl)}
            | {f"scale_g_{i}": constraints.positive for i in range(N_g)}
            | {f"lambda_{i}": constraints.unit_interval for i in range(N_pl + N_g - 1)}
            | {f"a1_loc_bpl_{i}": constraints.unit_interval for i in range(N_pl)}
            | {f"a1_scale_bpl_{i}": constraints.positive for i in range(N_pl)}
            | {f"a2_loc_bpl_{i}": constraints.unit_interval for i in range(N_pl)}
            | {f"a2_scale_bpl_{i}": constraints.positive for i in range(N_pl)}
            | {f"t1_loc_bpl_{i}": constraints.interval(-1.0, 1.0) for i in range(N_pl)}
            | {f"t1_scale_bpl_{i}": constraints.positive for i in range(N_pl)}
            | {f"t2_loc_bpl_{i}": constraints.interval(-1.0, 1.0) for i in range(N_pl)}
            | {f"t2_scale_bpl_{i}": constraints.positive for i in range(N_pl)}
            | {f"a1_loc_g_{i}": constraints.unit_interval for i in range(N_g)}
            | {f"a1_scale_g_{i}": constraints.positive for i in range(N_g)}
            | {f"a2_loc_g_{i}": constraints.unit_interval for i in range(N_g)}
            | {f"a2_scale_g_{i}": constraints.positive for i in range(N_g)}
            | {f"t1_loc_g_{i}": constraints.interval(-1.0, 1.0) for i in range(N_g)}
            | {f"t1_scale_g_{i}": constraints.positive for i in range(N_g)}
            | {f"t2_loc_g_{i}": constraints.interval(-1.0, 1.0) for i in range(N_g)}
            | {f"t2_scale_g_{i}": constraints.positive for i in range(N_g)}
        )
        pytree_aux_fields = ("N_pl", "N_g")
        pytree_data_fields = (
            (
                "_log_Z_q",
                "beta",
                "delta_m1",
                "delta_m2",
                "kappa",
                "log_rate",
                "logZ",
                "m1max",
                "m1min",
                "m2min",
                "z_max",
            )
            + tuple(f"alpha1_bpl_{i}" for i in range(N_pl))
            + tuple(f"alpha2_bpl_{i}" for i in range(N_pl))
            + tuple(f"m1min_bpl_{i}" for i in range(N_pl))
            + tuple(f"m1max_bpl_{i}" for i in range(N_pl))
            + tuple(f"m1break_bpl_{i}" for i in range(N_pl))
            + tuple(f"loc_g_{i}" for i in range(N_g))
            + tuple(f"scale_g_{i}" for i in range(N_g))
            + tuple(f"m1min_g_{i}" for i in range(N_g))
            + tuple(f"m1max_g_{i}" for i in range(N_g))
            + tuple(f"lambda_{i}" for i in range(N_pl + N_g - 1))
            + tuple(f"a1_loc_g_{i}" for i in range(N_g))
            + tuple(f"a1_loc_bpl_{i}" for i in range(N_pl))
            + tuple(f"a1_scale_g_{i}" for i in range(N_g))
            + tuple(f"a1_scale_bpl_{i}" for i in range(N_pl))
            + tuple(f"a2_loc_g_{i}" for i in range(N_g))
            + tuple(f"a2_loc_bpl_{i}" for i in range(N_pl))
            + tuple(f"a2_scale_g_{i}" for i in range(N_g))
            + tuple(f"a2_scale_bpl_{i}" for i in range(N_pl))
            + tuple(f"t1_loc_g_{i}" for i in range(N_g))
            + tuple(f"t1_loc_bpl_{i}" for i in range(N_pl))
            + tuple(f"t1_scale_g_{i}" for i in range(N_g))
            + tuple(f"t1_scale_bpl_{i}" for i in range(N_pl))
            + tuple(f"t2_loc_g_{i}" for i in range(N_g))
            + tuple(f"t2_loc_bpl_{i}" for i in range(N_pl))
            + tuple(f"t2_scale_g_{i}" for i in range(N_g))
            + tuple(f"t2_scale_bpl_{i}" for i in range(N_pl))
        )

        def __init__(
            self,
            N_pl: int,
            N_g: int,
            _log_Z_q: Array,
            beta: Array,
            delta_m1: Array,
            delta_m2: Array,
            kappa: Array,
            log_rate: Array,
            logZ: Array,
            m2min: Array,
            m1min: Array,
            m1max: Array,
            z_max: Array,
            validate_args: Optional[bool] = None,
            **kwargs: ArrayLike,
        ) -> None:
            self.N_pl = N_pl
            self.N_g = N_g
            self.log_rate = log_rate
            self.delta_m1 = delta_m1
            self.delta_m2 = delta_m2
            self.beta = beta
            self.m2min = m2min
            self.m1min = m1min
            self.m1max = m1max
            self.logZ = logZ
            self.kappa = kappa
            self.z_max = z_max
            self._log_Z_q = _log_Z_q
            for k, v in kwargs.items():
                setattr(self, k, v)
            self._support = all_constraint(
                [
                    mass_sandwich(m2min, m1max),  # type: ignore
                    constraints.unit_interval,
                    constraints.unit_interval,
                    constraints.interval(-1.0, 1.0),
                    constraints.interval(-1.0, 1.0),
                    constraints.positive,
                ],
                ((0, 2), 2, 3, 4, 5, 6),
            )
            super(_NSmoothingPowerlawMGaussian, self).__init__(
                batch_shape=(), event_shape=(7,), validate_args=validate_args
            )

        @constraints.dependent_property(event_dim=-1, is_discrete=False)
        def support(self):
            return self._support

        @validate_sample
        def log_prob(self, value):
            m1, m2, a1, a2, t1, t2, z = jnp.unstack(value, axis=-1)
            m1_log_prob = _n_bpls_m_gs_component_log_prob(
                self.N_pl,
                self.N_g,
                m1,
                **{
                    k: getattr(self, k)
                    for k in self.pytree_data_fields
                    if any(
                        (
                            k.startswith(name)
                            for name in (
                                "alpha1",
                                "alpha2",
                                "lambda",
                                "loc",
                                "m1break",
                                "m1max",
                                "m1min",
                                "scale",
                            )
                        )
                    )
                },
            )

            safe_delta_m2 = jnp.where(self.delta_m2 <= 0.0, 1.0, self.delta_m2)
            log_smoothing_q = log_planck_taper_window((m2 - self.m2min) / safe_delta_m2)
            log_prob_q = self.beta * jnp.log(m2 / m1) + log_smoothing_q
            log_prob_q = jnp.where(
                (self.delta_m2 <= 0.0) | (m2 < self.m2min) | (m2 > m1),
                -jnp.inf,
                log_prob_q,
            )

            aa_log_prob = []
            tt_log_prob = []

            for n_pl in range(self.N_pl):
                a1_loc = getattr(self, f"a1_loc_bpl_{n_pl}")
                a1_scale = getattr(self, f"a1_scale_bpl_{n_pl}")
                a2_loc = getattr(self, f"a2_loc_bpl_{n_pl}")
                a2_scale = getattr(self, f"a2_scale_bpl_{n_pl}")

                aa_log_prob.append(
                    truncnorm_logpdf(a1, a1_loc, a1_scale, 0.0, 1.0)
                    + truncnorm_logpdf(a2, a2_loc, a2_scale, 0.0, 1.0)
                )

                t1_loc = getattr(self, f"t1_loc_bpl_{n_pl}")
                t1_scale = getattr(self, f"t1_scale_bpl_{n_pl}")
                t2_loc = getattr(self, f"t2_loc_bpl_{n_pl}")
                t2_scale = getattr(self, f"t2_scale_bpl_{n_pl}")
                zeta_pl = getattr(self, f"zeta_bpl_{n_pl}")

                tt_log_prob.append(
                    jnp.logaddexp(
                        jnp.log(zeta_pl)
                        + truncnorm_logpdf(t1, t1_loc, t1_scale, -1.0, 1.0)
                        + truncnorm_logpdf(t2, t2_loc, t2_scale, -1.0, 1.0),
                        jnp.log1p(-zeta_pl) + jnp.log(0.25),
                    )
                )

            for n_g in range(self.N_g):
                a1_loc = getattr(self, f"a1_loc_g_{n_g}")
                a1_scale = getattr(self, f"a1_scale_g_{n_g}")
                a2_loc = getattr(self, f"a2_loc_g_{n_g}")
                a2_scale = getattr(self, f"a2_scale_g_{n_g}")

                aa_log_prob.append(
                    truncnorm_logpdf(a1, a1_loc, a1_scale, 0.0, 1.0)
                    + truncnorm_logpdf(a2, a2_loc, a2_scale, 0.0, 1.0)
                )

                t1_loc = getattr(self, f"t1_loc_g_{n_g}")
                t1_scale = getattr(self, f"t1_scale_g_{n_g}")
                t2_loc = getattr(self, f"t2_loc_g_{n_g}")
                t2_scale = getattr(self, f"t2_scale_g_{n_g}")
                zeta_g = getattr(self, f"zeta_g_{n_g}")

                tt_log_prob.append(
                    jnp.logaddexp(
                        jnp.log(zeta_g)
                        + truncnorm_logpdf(t1, t1_loc, t1_scale, -1.0, 1.0)
                        + truncnorm_logpdf(t2, t2_loc, t2_scale, -1.0, 1.0),
                        jnp.log1p(-zeta_g) + jnp.log(0.25),
                    )
                )

            aa_log_prob = jnp.stack(aa_log_prob, axis=-1)
            tt_log_prob = jnp.stack(tt_log_prob, axis=-1)

            log_prob_z = PLANCK_2015_Cosmology().logdVcdz(z) + (
                self.kappa - 1
            ) * jnp.log1p(z)

            safe_log_prob_z = jnp.where(
                (z < 0.0) | (z > self.z_max), -jnp.inf, log_prob_z
            )

            m1_aa_tt_log_prob = m1_log_prob + aa_log_prob + tt_log_prob

            safe_delta_m1 = jnp.where(self.delta_m1 <= 0.0, 1.0, self.delta_m1)
            log_smoothing_m1 = log_planck_taper_window(
                (m1 - self.m1min) / safe_delta_m1
            )

            log_prob_val = (
                -jnp.log(m1)
                + log_rate
                + safe_log_prob_z
                + log_prob_q
                + jax.nn.logsumexp(
                    m1_aa_tt_log_prob,
                    where=~jnp.isneginf(m1_aa_tt_log_prob),
                    axis=-1,
                )
                + log_smoothing_m1
                - self.logZ
                - jnp.interp(m1, mm, self._log_Z_q, left=0.0, right=0.0)
            )

            safe_log_prob_val = jnp.where(
                jnp.isnan(log_prob_val)
                | (self.delta_m1 <= 0.0)
                | (self.delta_m2 <= 0.0)
                | jnp.isneginf(log_prob_val)
                | (z < 0.0)
                | (z > self.z_max),
                -jnp.inf,
                log_prob_val,
            )

            return safe_log_prob_val

    return _NSmoothingPowerlawMGaussian(
        N_pl=N_pl,
        N_g=N_g,
        log_rate=log_rate,
        delta_m1=delta_m1,
        delta_m2=delta_m2,
        beta=beta,
        m2min=m2min,
        m1min=m1min,
        m1max=m1max,
        logZ=logZ,
        z_max=z_max,
        kappa=kappa,
        _log_Z_q=_log_Z_q,
        **params,
        validate_args=validate_args,
    )
