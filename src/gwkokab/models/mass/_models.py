# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Two-dimensional mass models for compact binary populations.

Every distribution here is a joint model over a pair of mass coordinates -- either
:math:`(m_1, m_2)` or :math:`(m_1, q)` -- with an ``event_shape`` of ``(2,)`` and a
support drawn from :mod:`gwkokab.models.constraints`, so the ordering
:math:`m_2 \leq m_1` and the mass bounds are enforced by construction.

The models divide into two groups. The plain ones (:class:`PowerlawPrimaryMassRatio`,
:class:`Wysocki2019MassModel`, :class:`GaussianPrimaryMassRatio`) have closed-form
normalisation and support :meth:`sample`. The *smoothed* ones multiply the primary
mass and mass ratio densities by a Planck taper window
(:func:`~gwkokab.utils.kernel.log_planck_taper_window`) that rolls the density off
over a width :math:`\delta` above the minimum mass instead of cutting it dead. That
window has no closed-form integral, so those models normalise numerically on a fixed
grid at construction time -- the primary mass normalisation once, and the
conditional mass ratio normalisation as a function of :math:`m_1` that
:meth:`log_prob` then interpolates. They evaluate densities but do not implement
``sample``.

See Also
--------
gwkokab.models.mass._bpls : Broken power law variants.
"""

from typing import Optional

from jax import lax, numpy as jnp, random as jrd
from jax.scipy import special
from jax.scipy.stats import norm, truncnorm, uniform
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

    Parameters
    ----------
    alpha : ArrayLike
        Power law index for primary mass.
    beta : ArrayLike
        Power law index for mass ratio.
    mmin : ArrayLike
        Minimum mass.
    mmax : ArrayLike
        Maximum mass.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.
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
        r"""The support of the distribution.

        Returns
        -------
        constraints.Constraint
            The mass sandwich :math:`m_{\min} \leq m_2 \leq m_1 \leq m_{\max}`, expressed in
            :math:`(m_1, q)` coordinates.
        """
        return self._support

    @validate_sample
    def log_prob(self, value):
        r"""Log probability density at ``value``.

        Parameters
        ----------
        value : Array
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        Array
            The log density, of shape ``batch_shape``. It is
            :math:`-\infty` where :math:`m_1 \leq m_{\min}`, since the conditional support
            of :math:`q` is then empty.
        """
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
        """Draw samples by inverse transform sampling.

        :math:`m_1` is drawn from its marginal power law and :math:`q` from the power law
        conditional on it.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key.
        sample_shape : tuple[int, ...]
            Shape of the sample batch to draw. Defaults to ``()``.

        Returns
        -------
        Array
            Samples of shape ``sample_shape + (2,)``, holding :math:`(m_1, q)`.
        """
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
    r"""Double side truncated power law mass model.

    As described in Equation 7 of `Reconstructing phenomenological distributions of
    compact binaries via gravitational wave observations
    <https://arxiv.org/abs/1805.06442>`_: the primary mass follows a truncated power
    law and the secondary is uniform between :math:`m_{\text{min}}` and :math:`m_1`.

    .. math::
        p(m_1,m_2\mid\alpha,m_{\text{min}},m_{\text{max}},M_{\text{max}})\propto
        \frac{m_1^{-\alpha}}{m_1-m_{\text{min}}}

    Parameters
    ----------
    alpha_m : ArrayLike
        Index of the power law distribution.
    mmin : ArrayLike
        Lower mass limit.
    mmax : ArrayLike
        Upper mass limit.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.
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
        r"""The support of the distribution.

        Returns
        -------
        constraints.Constraint
            The mass sandwich :math:`m_{\min} \leq m_2 \leq m_1 \leq m_{\max}`, expressed in
            :math:`(m_1, m_2)` coordinates.
        """
        return self._support

    @validate_sample
    def log_prob(self, value):
        """Log probability density at ``value``.

        Parameters
        ----------
        value : Array
            Values whose last axis is :math:`(m_1, m_2)`.

        Returns
        -------
        Array
            The log density, of shape ``batch_shape``.
        """
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
        r"""Draw samples from the model.

        :math:`m_1` is drawn from the truncated power law by inverse transform sampling,
        then :math:`m_2` uniformly on :math:`[m_{\min}, m_1]`.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key.
        sample_shape : tuple[int, ...]
            Shape of the sample batch to draw. Defaults to ``()``.

        Returns
        -------
        Array
            Samples of shape ``sample_shape + (2,)``, holding :math:`(m_1, m_2)`.
        """
        key_m1, key_m2 = jrd.split(key)
        u_m1 = jrd.uniform(key_m1, shape=sample_shape)
        m1 = doubly_truncated_power_law_icdf(
            q=u_m1, alpha=-self.alpha_m, low=self.mmin, high=self.mmax
        )
        m2 = jrd.uniform(key_m2, shape=sample_shape, minval=self.mmin, maxval=m1)
        return jnp.stack((m1, m2), axis=-1)


class SmoothedTwoComponentPrimaryMassRatio(Distribution):
    r"""Smoothed power-law-plus-peak primary mass with a conditional mass ratio power
    law.

    The primary mass is a mixture of a truncated power law and a Gaussian peak, both
    tapered on the lower edge by a Planck window of width :math:`\delta`:

    .. math::
        p(m_1) \propto \left[(1-\lambda)\,\frac{m_1^{-\alpha}}{Z_{\text{pl}}}
        + \lambda\,\mathcal{N}(m_1\mid\mu,\sigma^2)\right]
        S\!\left(\frac{m_1 - m_{\text{min}}}{\delta}\right)

    .. math::
        p(q\mid m_1) \propto q^{\beta}
        S\!\left(\frac{m_1 q - m_{\text{min}}}{\delta}\right)

    Normalisation is computed once at construction: the primary mass constant splits
    into a numerical integral over the taper region :math:`[m_{\min}, m_{\min}+\delta]`
    and the analytic integral above it, while the conditional mass ratio constant is
    tabulated over :math:`m_1` and interpolated in :meth:`log_prob`.

    Parameters
    ----------
    alpha : ArrayLike
        Power law index for primary mass.
    beta : ArrayLike
        Power law index for mass ratio.
    delta : ArrayLike
        Width of the smoothing window above :math:`m_{\text{min}}`.
    lambda_peak : ArrayLike
        Mixing fraction :math:`\lambda` of the Gaussian peak, in :math:`[0, 1]`.
    loc : ArrayLike
        Location :math:`\mu` of the Gaussian peak.
    mmax : ArrayLike
        Maximum mass.
    mmin : ArrayLike
        Minimum mass.
    scale : ArrayLike
        Scale :math:`\sigma` of the Gaussian peak.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Notes
    -----
    Sampling is not implemented; this model evaluates densities only.
    """

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
        r"""The support of the distribution.

        Returns
        -------
        constraints.Constraint
            The mass sandwich :math:`m_{\min} \leq m_2 \leq m_1 \leq m_{\max}`, expressed in
            :math:`(m_1, q)` coordinates.
        """
        return self._support

    def _log_prob_m1_unnorm(self, m1: Array) -> Array:
        r"""Unnormalised log density of the primary mass.

        Parameters
        ----------
        m1 : Array
            Primary masses.

        Returns
        -------
        Array
            The log of the tapered power-law-plus-peak density, before dividing by
            :attr:`_logZ`. A non-positive ``delta`` gives :math:`-\infty`.
        """
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
        r"""Unnormalised log density of the mass ratio given the primary mass.

        Parameters
        ----------
        value : Array
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        Array
            :math:`\beta\log q` plus the log taper applied to :math:`m_2 = m_1 q`,
            before dividing by the conditional normalisation. A non-positive ``delta``
            gives :math:`-\infty`.
        """
        m1, q = jnp.unstack(value, axis=-1)
        m2 = m1 * q
        safe_delta = jnp.where(self.delta <= 0.0, 1.0, self.delta)
        log_smoothing_q = log_planck_taper_window((m2 - self.mmin) / safe_delta)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_q

        return jnp.where(self.delta <= 0.0, -jnp.inf, log_prob_q)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """Log probability density at ``value``.

        Parameters
        ----------
        value : ArrayLike
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        ArrayLike
            The normalised log density, of shape ``batch_shape``. The conditional
            normalisation of :math:`q` is interpolated from the grid built at construction.
        """
        m1, _ = jnp.unstack(value, axis=-1)
        log_prob_m1 = self._log_prob_m1_unnorm(m1) - self._logZ
        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1, left=1.0, right=1.0)
        safe_Z_q = jnp.where(_Z_q <= 0, 1.0, _Z_q)
        log_Z_q = jnp.where(_Z_q <= 0, 0.0, jnp.log(safe_Z_q))
        log_prob_q = self._log_prob_q_unnorm(value) - log_Z_q
        return log_prob_m1 + log_prob_q


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

    Parameters
    ----------
    loc : ArrayLike
        Location :math:`\mu` of the primary mass.
    scale : ArrayLike
        Scale :math:`\sigma` of the primary mass.
    beta : ArrayLike
        Power law index for mass ratio.
    m1min : ArrayLike
        Minimum primary mass.
    m2min : ArrayLike
        Minimum secondary mass.
    mmax : ArrayLike
        Maximum mass.
    delta_m1 : ArrayLike
        Width of the smoothing window above :math:`m_{1,\text{min}}`.
    delta_m2 : ArrayLike
        Width of the smoothing window above :math:`m_{2,\text{min}}`.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Notes
    -----
    Both normalisation constants are computed numerically at construction, the conditional
    one as a function of :math:`m_1` that :meth:`log_prob` interpolates. Sampling is not
    implemented; this model evaluates densities only.
    """

    arg_constraints = {
        "loc": constraints.positive,
        "scale": constraints.positive,
        "beta": constraints.real,
        "delta_m1": constraints.positive,
        "delta_m2": constraints.positive,
        "m1min": constraints.positive,
        "m2min": constraints.positive,
        "mmax": constraints.positive,
    }
    pytree_data_fields = (
        "_logZ",
        "_m1s",
        "_support",
        "_Z_q_given_m1",
        "beta",
        "delta_m1",
        "delta_m2",
        "loc",
        "m1min",
        "m2min",
        "mmax",
        "scale",
    )

    def __init__(
        self,
        loc,
        scale,
        beta,
        m1min,
        m2min,
        mmax,
        delta_m1,
        delta_m2,
        *,
        validate_args=None,
    ) -> None:
        (
            self.beta,
            self.delta_m1,
            self.delta_m2,
            self.loc,
            self.m1min,
            self.m2min,
            self.mmax,
            self.scale,
        ) = promote_shapes(
            beta,
            delta_m1,
            delta_m2,
            loc,
            m1min,
            m2min,
            mmax,
            scale,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc),
            jnp.shape(scale),
            jnp.shape(beta),
            jnp.shape(m1min),
            jnp.shape(m2min),
            jnp.shape(mmax),
            jnp.shape(delta_m1),
            jnp.shape(delta_m2),
        )
        self._support = mass_ratio_mass_sandwich(m2min, mmax)
        super(SmoothedGaussianPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
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
        r"""The support of the distribution.

        Returns
        -------
        constraints.Constraint
            The mass sandwich :math:`m_{\min} \leq m_2 \leq m_1 \leq m_{\max}`, expressed in
            :math:`(m_1, q)` coordinates.
        """
        return self._support

    def _log_prob_m1(self, m1: Array, logZ: ArrayLike = 0.0) -> Array:
        r"""Log density of the primary mass, optionally normalised.

        Parameters
        ----------
        m1 : Array
            Primary masses.
        logZ : ArrayLike, optional
            Log normalisation constant to subtract. Defaults to ``0.0``, giving the
            unnormalised density.

        Returns
        -------
        Array
            The log of the tapered normal density. Non-finite values are mapped to
            :math:`-\infty` so they do not propagate NaNs.
        """
        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / self.delta_m1)
        log_prob_norm = norm.logpdf(m1, loc=self.loc, scale=self.scale)
        log_prob_m1 = log_prob_norm + log_smoothing_m1 - logZ
        return jnp.nan_to_num(
            log_prob_m1,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

    def _log_prob_q(self, value: Array, logZ: ArrayLike = 0.0) -> Array:
        r"""Log density of the mass ratio given the primary mass, optionally normalised.

        Parameters
        ----------
        value : Array
            Values whose last axis is :math:`(m_1, q)`.
        logZ : ArrayLike, optional
            Log normalisation constant to subtract. Defaults to ``0.0``, giving the
            unnormalised density.

        Returns
        -------
        Array
            :math:`\beta\log q` plus the log taper applied to :math:`m_2 = m_1 q`,
            masked to :math:`-\infty` outside the support.
        """
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
        """Log probability density at ``value``.

        Parameters
        ----------
        value : ArrayLike
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        ArrayLike
            The normalised log density, of shape ``batch_shape``.
        """
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


class GaussianPrimaryMassRatio(Distribution):
    r"""Truncated normal primary mass with a conditional mass ratio power law.

    .. math::
        p(m_1, q) = p(m_1\mid\mu,\sigma^2)\,p(q\mid m_1, \beta)

    .. math::
        \begin{align*}
            p(m_1\mid\mu,\sigma^2)&
            \propto \mathcal{N}(m_1\mid\mu,\sigma^2),\qquad
            m_{\text{min}}\leq m_1\leq m_{\max}\\
            p(q\mid m_1,\beta)&
            \propto q^{\beta},\qquad \frac{m_{\text{min}}}{m_1}\leq q\leq 1
        \end{align*}

    Parameters
    ----------
    loc : ArrayLike
        Location parameter for primary mass.
    scale : ArrayLike
        Scale parameter for primary mass.
    beta : ArrayLike
        Power law index for mass ratio.
    mmin : ArrayLike
        Minimum mass.
    mmax : ArrayLike
        Maximum mass.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Notes
    -----
    Sampling is not implemented; this model evaluates densities only.
    """

    arg_constraints = {
        "loc": constraints.positive,
        "scale": constraints.positive,
        "beta": constraints.real,
        "mmin": constraints.positive,
        "mmax": constraints.positive,
    }
    reparametrized_params = ["loc", "scale", "beta", "mmin", "mmax"]
    pytree_data_fields = ("_support", "loc", "scale", "beta", "mmax", "mmin")

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        beta: ArrayLike,
        mmin: ArrayLike,
        mmax: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.loc, self.scale, self.beta, self.mmin, self.mmax = promote_shapes(
            loc, scale, beta, mmin, mmax
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc),
            jnp.shape(scale),
            jnp.shape(beta),
            jnp.shape(mmin),
            jnp.shape(mmax),
        )
        self._support = mass_ratio_mass_sandwich(mmin, mmax)
        super(GaussianPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        r"""The support of the distribution.

        Returns
        -------
        constraints.Constraint
            The mass sandwich :math:`m_{\min} \leq m_2 \leq m_1 \leq m_{\max}`, expressed in
            :math:`(m_1, q)` coordinates.
        """
        return self._support

    @validate_sample
    def log_prob(self, value):
        r"""Log probability density at ``value``.

        Parameters
        ----------
        value : Array
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        Array
            The log density, of shape ``batch_shape``. It is :math:`-\infty` where
            :math:`m_1 \leq m_{\min}`, since the conditional support of :math:`q` is then
            empty.
        """
        m1, q = jnp.unstack(value, axis=-1)
        log_prob_m1 = truncnorm.logpdf(
            m1,
            a=(self.mmin - self.loc) / self.scale,
            b=(self.mmax - self.loc) / self.scale,
            loc=self.loc,
            scale=self.scale,
        )
        log_prob_q = jnp.where(
            jnp.less_equal(m1, self.mmin),
            -jnp.inf,
            doubly_truncated_power_law_log_prob(
                x=q, alpha=self.beta, low=self.mmin / m1, high=1.0
            ),
        )

        return log_prob_m1 + log_prob_q


class SmoothedPowerlawPrimaryMassRatio(Distribution):
    r"""Smoothed power law primary mass with a conditional mass ratio power law.

    Both the primary mass and the mass ratio are tapered on their lower edges by a
    Planck window, each with its own width:

    .. math::
        p(m_1) \propto m_1^{-\alpha}
        S\!\left(\frac{m_1 - m_{1,\text{min}}}{\delta_{m_1}}\right)

    .. math::
        p(q\mid m_1) \propto q^{\beta}
        S\!\left(\frac{m_1 q - m_{2,\text{min}}}{\delta_{m_2}}\right)

    Both normalisation constants are computed numerically at construction, the
    conditional one as a function of :math:`m_1` that :meth:`log_prob` interpolates.

    Parameters
    ----------
    alpha : ArrayLike
        Power law index for primary mass.
    beta : ArrayLike
        Power law index for mass ratio.
    delta_m1 : ArrayLike
        Width of the smoothing window above :math:`m_{1,\text{min}}`.
    delta_m2 : ArrayLike
        Width of the smoothing window above :math:`m_{2,\text{min}}`.
    mmax : ArrayLike
        Maximum mass.
    m1min : ArrayLike
        Minimum primary mass.
    m2min : ArrayLike
        Minimum secondary mass.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Notes
    -----
    Sampling is not implemented; this model evaluates densities only.
    """

    arg_constraints = {
        "alpha": constraints.real,
        "beta": constraints.real,
        "delta_m1": constraints.positive,
        "delta_m2": constraints.positive,
        "mmax": constraints.positive,
        "m1min": constraints.positive,
        "m2min": constraints.positive,
    }
    pytree_data_fields = (
        "_logZ",
        "_m1s",
        "_support",
        "_Z_q_given_m1",
        "alpha",
        "beta",
        "delta_m1",
        "delta_m2",
        "mmax",
        "m1min",
        "m2min",
    )

    def __init__(
        self,
        alpha: ArrayLike,
        beta: ArrayLike,
        delta_m1: ArrayLike,
        delta_m2: ArrayLike,
        mmax: ArrayLike,
        m1min: ArrayLike,
        m2min: ArrayLike,
        *,
        validate_args=None,
    ) -> None:
        (
            self.alpha,
            self.beta,
            self.delta_m1,
            self.delta_m2,
            self.mmax,
            self.m1min,
            self.m2min,
        ) = promote_shapes(
            alpha,
            beta,
            delta_m1,
            delta_m2,
            mmax,
            m1min,
            m2min,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(beta),
            jnp.shape(delta_m1),
            jnp.shape(delta_m2),
            jnp.shape(mmax),
            jnp.shape(m1min),
            jnp.shape(m2min),
        )
        self._support = mass_ratio_mass_sandwich(m2min, mmax)
        super(SmoothedPowerlawPrimaryMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

        mmin = jnp.broadcast_to(m2min, batch_shape)
        mmax = jnp.broadcast_to(mmax, batch_shape)

        # Compute the normalization constant for primary mass distribution

        self._m1s = jnp.linspace(mmin, mmax, 1000)
        _Z = jnp.trapezoid(
            jnp.exp(self._log_prob_m1_unnorm(self._m1s)),
            self._m1s,
            axis=0,
        )
        self._logZ = jnp.where(
            jnp.isnan(_Z) | jnp.isinf(_Z) | jnp.less_equal(_Z, 0.0),
            0.0,
            jnp.log(_Z),
        )

        # Compute the normalization constant for mass ratio distribution

        _qs = jnp.linspace(0.005, 1.0, 500)
        _m1qs_grid = jnp.stack(jnp.meshgrid(self._m1s, _qs, indexing="ij"), axis=-1)

        _prob_q = jnp.exp(self._log_prob_q_unnorm(_m1qs_grid))

        self._Z_q_given_m1 = jnp.trapezoid(_prob_q, _qs, axis=1)

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        r"""The support of the distribution.

        Returns
        -------
        constraints.Constraint
            The mass sandwich :math:`m_{\min} \leq m_2 \leq m_1 \leq m_{\max}`, expressed in
            :math:`(m_1, q)` coordinates.
        """
        return self._support

    def _log_prob_m1_unnorm(self, m1: Array) -> Array:
        r"""Unnormalised log density of the primary mass.

        Parameters
        ----------
        m1 : Array
            Primary masses.

        Returns
        -------
        Array
            :math:`-\alpha\log m_1` plus the log taper. A non-positive ``delta_m1``
            gives :math:`-\infty`.
        """
        safe_delta = jnp.where(self.delta_m1 <= 0.0, 1.0, self.delta_m1)
        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / safe_delta)

        log_prob_m1 = -self.alpha * jnp.log(m1) + log_smoothing_m1

        return jnp.where(self.delta_m1 <= 0.0, -jnp.inf, log_prob_m1)

    def _log_prob_q_unnorm(self, value: Array) -> Array:
        r"""Unnormalised log density of the mass ratio given the primary mass.

        Parameters
        ----------
        value : Array
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        Array
            :math:`\beta\log q` plus the log taper applied to :math:`m_2 = m_1 q`. It is
            :math:`-\infty` where ``delta_m2`` is non-positive or :math:`m_2 < m_{2,\min}`.
        """
        m1, q = jnp.unstack(value, axis=-1)
        m2 = m1 * q
        safe_delta = jnp.where(self.delta_m2 <= 0.0, 1.0, self.delta_m2)
        log_smoothing_q = log_planck_taper_window((m2 - self.m2min) / safe_delta)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_q

        return jnp.where(
            (self.delta_m2 <= 0.0) | (self.m2min > m2), -jnp.inf, log_prob_q
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """Log probability density at ``value``.

        Parameters
        ----------
        value : ArrayLike
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        ArrayLike
            The normalised log density, of shape ``batch_shape``.
        """
        m1, _ = jnp.unstack(value, axis=-1)
        log_prob_m1 = self._log_prob_m1_unnorm(m1) - self._logZ
        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1, left=1.0, right=1.0)
        safe_Z_q = jnp.where(_Z_q <= 0, 1.0, _Z_q)
        log_Z_q = jnp.where(_Z_q <= 0, 0.0, jnp.log(safe_Z_q))
        log_prob_q = self._log_prob_q_unnorm(value) - log_Z_q
        return log_prob_m1 + log_prob_q


class GenericSmoothedPowerlawMassRatio(Distribution):
    r"""Smoothed conditional mass ratio power law over an arbitrary primary mass model.

    Generalises :class:`SmoothedPowerlawPrimaryMassRatio` by taking the primary mass
    distribution as an argument instead of fixing it to a power law. The mass bounds
    are read off that distribution's interval support, and the same Planck taper is
    applied to both coordinates.

    .. math::
        p(m_1) \propto p_{\text{prim}}(m_1)
        S\!\left(\frac{m_1 - m_{1,\text{min}}}{\delta_{m_1}}\right)

    .. math::
        p(q\mid m_1) \propto q^{\beta}
        S\!\left(\frac{m_1 q - m_{2,\text{min}}}{\delta_{m_2}}\right)

    Parameters
    ----------
    primary_mass_distribution : Distribution
        Distribution of the primary mass. Must have an
        :class:`~numpyro.distributions.constraints.interval` support, whose bounds
        supply :math:`m_{1,\text{min}}` and :math:`m_{\max}`.
    beta : ArrayLike
        Power law index for mass ratio.
    delta_m1 : ArrayLike
        Width of the smoothing window above :math:`m_{1,\text{min}}`.
    delta_m2 : ArrayLike
        Width of the smoothing window above :math:`m_{2,\text{min}}`.
    m2min : ArrayLike
        Minimum secondary mass.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Raises
    ------
    ValueError
        If ``primary_mass_distribution`` does not have an interval support.

    Notes
    -----
    Only the conditional mass ratio density is normalised numerically; the primary mass
    term is taken as given by ``primary_mass_distribution``, so the taper leaves it
    slightly sub-normalised. Sampling is not implemented.
    """

    args_constraints = {
        "beta": constraints.real,
        "delta_m1": constraints.positive,
        "delta_m2": constraints.positive,
        "m1min": constraints.positive,
        "m2min": constraints.positive,
    }
    pytree_data_fields = (
        "_m1s",
        "_support",
        "_Z_q_given_m1",
        "beta",
        "delta_m1",
        "delta_m2",
        "m1min",
        "m2min",
        "mmax",
        "primary_mass_distribution",
    )

    def __init__(
        self,
        primary_mass_distribution: Distribution,
        beta: ArrayLike,
        delta_m1: ArrayLike,
        delta_m2: ArrayLike,
        m2min: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        if not isinstance(primary_mass_distribution.support, constraints.interval):
            raise ValueError(
                "primary_mass_distribution must have an interval support constraint"
            )
        m1min = primary_mass_distribution.support.lower_bound
        mmax = primary_mass_distribution.support.upper_bound
        self.primary_mass_distribution = primary_mass_distribution
        (
            self.beta,
            self.delta_m1,
            self.delta_m2,
            self.m1min,
            self.m2min,
            self.mmax,
        ) = promote_shapes(
            beta,
            delta_m1,
            delta_m2,
            m1min,
            m2min,
            mmax,
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(beta),
            jnp.shape(delta_m1),
            jnp.shape(delta_m2),
            jnp.shape(m1min),
            jnp.shape(m2min),
            jnp.shape(mmax),
        )

        self._m1s = jnp.linspace(m1min, mmax, 1000)
        _qs = jnp.linspace(0.005, 1.0, 500)

        _m1qs_grid = jnp.stack(jnp.meshgrid(self._m1s, _qs, indexing="ij"), axis=-1)

        _prob_q = jnp.exp(self._log_prob_q_unnorm(_m1qs_grid))

        self._Z_q_given_m1 = jnp.trapezoid(_prob_q, _qs, axis=1)

        self._support = mass_ratio_mass_sandwich(m2min, mmax)

        super(GenericSmoothedPowerlawMassRatio, self).__init__(
            batch_shape=batch_shape, event_shape=(2,), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self) -> constraints.Constraint:
        r"""The support of the distribution.

        Returns
        -------
        constraints.Constraint
            The mass sandwich :math:`m_{\min} \leq m_2 \leq m_1 \leq m_{\max}`, expressed in
            :math:`(m_1, q)` coordinates.
        """
        return self._support

    def _log_prob_q_unnorm(self, value: Array) -> Array:
        r"""Unnormalised log density of the mass ratio given the primary mass.

        Parameters
        ----------
        value : Array
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        Array
            :math:`\beta\log q` plus the log taper applied to :math:`m_2 = m_1 q`. It is
            :math:`-\infty` where ``delta_m2`` is non-positive or :math:`m_2 < m_{2,\min}`.
        """
        m1, q = jnp.unstack(value, axis=-1)
        m2 = m1 * q
        safe_delta = jnp.where(self.delta_m2 <= 0.0, 1.0, self.delta_m2)
        log_smoothing_q = log_planck_taper_window((m2 - self.m2min) / safe_delta)
        log_prob_q = self.beta * jnp.log(q) + log_smoothing_q

        return jnp.where(
            (self.delta_m2 <= 0.0) | (self.m2min > m2), -jnp.inf, log_prob_q
        )

    def _log_prob_m1_unnorm(self, m1: Array) -> Array:
        r"""Unnormalised log density of the primary mass.

        Parameters
        ----------
        m1 : Array
            Primary masses.

        Returns
        -------
        Array
            The wrapped distribution's log density plus the log taper. A non-positive
            ``delta_m1`` gives :math:`-\infty`.
        """
        safe_delta = jnp.where(self.delta_m1 <= 0.0, 1.0, self.delta_m1)
        log_smoothing_m1 = log_planck_taper_window((m1 - self.m1min) / safe_delta)

        log_prob_m1 = self.primary_mass_distribution.log_prob(m1) + log_smoothing_m1

        return jnp.where(self.delta_m1 <= 0.0, -jnp.inf, log_prob_m1)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """Log probability density at ``value``.

        Parameters
        ----------
        value : ArrayLike
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        ArrayLike
            The log density, of shape ``batch_shape``, with the conditional mass ratio
            term normalised by interpolation on the grid built at construction.
        """
        m1, _ = jnp.unstack(value, axis=-1)
        log_prob_m1 = self._log_prob_m1_unnorm(m1)
        _Z_q = jnp.interp(m1, self._m1s, self._Z_q_given_m1, left=1.0, right=1.0)
        safe_Z_q = jnp.where(_Z_q <= 0, 1.0, _Z_q)
        log_Z_q = jnp.where(_Z_q <= 0, 0.0, jnp.log(safe_Z_q))
        log_prob_q = self._log_prob_q_unnorm(value) - log_Z_q
        return log_prob_m1 + log_prob_q
