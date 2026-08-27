# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Doubly truncated power law with hand-written derivative rules.

A power law on :math:`[a, b]` has a normalisation constant whose closed form changes
at :math:`\alpha = -1`, where the algebraic expression becomes :math:`0/0`. Selecting
between the two branches with :func:`jax.numpy.where` gives the right value but a NaN
gradient, because the untaken branch is still differentiated.

Every function here therefore comes in a pair: a :func:`jax.custom_jvp` primal that
picks the branch, and an explicit JVP rule that supplies the tangents. On the
:math:`\alpha = -1` branch the tangent with respect to :math:`\alpha` has no clean
closed form and is approximated by a symmetric finite difference evaluated at
:math:`\alpha \pm \delta`.

The CDF and inverse-CDF implementations are adapted from `NumPyro's truncated
distributions
<https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py>`_.
"""

from typing import Optional

import jax
from jax import numpy as jnp
from jaxtyping import ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key


@jax.custom_jvp
def doubly_truncated_power_law_log_norm_constant(alpha, low, high):
    r"""Log normalisation constant of a doubly truncated power law.

    .. math::
        \log Z(\alpha, a, b) = \begin{cases}
            \log(\log b - \log a) & \alpha = -1, \\
            \log\dfrac{b^{1+\alpha} - a^{1+\alpha}}{1+\alpha} & \text{otherwise}.
        \end{cases}

    Parameters
    ----------
    alpha : ArrayLike
        Power law index :math:`\alpha`.
    low : ArrayLike
        Lower bound :math:`a`.
    high : ArrayLike
        Upper bound :math:`b`.

    Returns
    -------
    ArrayLike
        The log normalisation constant :math:`\log Z`.
    """
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def neq_neg1_fn():
        r"""Log normalisation constant on the generic :math:`\alpha \neq -1` branch."""
        one_more_alpha = 1.0 + neq_neg1_alpha
        return jnp.log(
            (jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha))
            / (one_more_alpha)
        )

    def eq_neg1_fn():
        r"""Log normalisation constant on the :math:`\alpha = -1` branch."""
        return jnp.log(jnp.log(high) - jnp.log(low))

    return jnp.where(neq_neg1_mask, neq_neg1_fn(), eq_neg1_fn())


@doubly_truncated_power_law_log_norm_constant.defjvp
def doubly_truncated_power_law_log_norm_constant_jvp(primals, tangents):
    r"""JVP rule for :func:`doubly_truncated_power_law_log_norm_constant`.

    The tangents with respect to ``low`` and ``high`` are exact on both branches. The
    tangent with respect to ``alpha`` is exact for :math:`\alpha \neq -1` and, at
    :math:`\alpha = -1`, is the average of the expression evaluated just either side of
    the singularity.

    Parameters
    ----------
    primals : tuple
        ``(alpha, low, high)``.
    tangents : tuple
        The corresponding tangent vectors.

    Returns
    -------
    tuple
        The primal output and its tangent.
    """
    alpha, low, high = primals
    alpha_t, low_t, high_t = tangents

    primal_out = doubly_truncated_power_law_log_norm_constant(*primals)

    log_low = jnp.log(low)
    log_high = jnp.log(high)

    # Mask and alpha values
    delta_eq_neg1 = 1e-5
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    # Alpha tangent with approximation
    # Variable part for all values alpha unequal -1
    def alpha_tangent_variable(alpha):
        r"""Exact :math:`\partial \log Z / \partial \alpha` for :math:`\alpha \neq
        -1`.
        """
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return -jnp.reciprocal(one_more_alpha) + (
            high_pow_one_more_alpha * log_high - low_pow_one_more_alpha * log_low
        ) / (high_pow_one_more_alpha - low_pow_one_more_alpha)

    # Alpha tangent
    alpha_tangent = jnp.where(
        neq_neg1_mask,
        alpha_tangent_variable(neq_neg1_alpha),
        # Approximate derivate with right an lefthand approximation
        (
            alpha_tangent_variable(alpha - delta_eq_neg1)
            + alpha_tangent_variable(alpha + delta_eq_neg1)
        )
        * 0.5,
    )

    # High and low tangents for alpha unequal -1
    one_more_alpha = 1.0 + neq_neg1_alpha
    low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
    high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
    change_neq_neg1_alpha = high_pow_one_more_alpha - low_pow_one_more_alpha
    low_tangent_neq_neg1 = (
        -one_more_alpha * jnp.power(low, neq_neg1_alpha) / change_neq_neg1_alpha
    )
    high_tangent_neq_neg1 = (
        one_more_alpha * jnp.power(high, neq_neg1_alpha) / change_neq_neg1_alpha
    )

    # High and low tangents for alpha equal -1
    change_eq_neg1_alpha = log_high - log_low
    low_tangent_eq_neg1 = -jnp.reciprocal(low * change_eq_neg1_alpha)
    high_tangent_eq_neg1 = jnp.reciprocal(high * change_eq_neg1_alpha)

    # High and low tangents
    low_tangent = jnp.where(neq_neg1_mask, low_tangent_neq_neg1, low_tangent_eq_neg1)
    high_tangent = jnp.where(neq_neg1_mask, high_tangent_neq_neg1, high_tangent_eq_neg1)

    # Final tangents
    tangent_out = alpha_tangent * alpha_t + low_tangent * low_t + high_tangent * high_t
    return primal_out, tangent_out


def doubly_truncated_power_law_log_prob(x, alpha, low, high):
    r"""Log density of a doubly truncated power law.

    .. math::
        \log f(x; \alpha, a, b) = \alpha \log x - \log Z(\alpha, a, b)

    Parameters
    ----------
    x : ArrayLike
        Points at which to evaluate the density; assumed to lie in :math:`[a, b]`.
    alpha : ArrayLike
        Power law index :math:`\alpha`.
    low : ArrayLike
        Lower bound :math:`a`.
    high : ArrayLike
        Upper bound :math:`b`.

    Returns
    -------
    ArrayLike
        The log density.
    """
    return alpha * jnp.log(x) - doubly_truncated_power_law_log_norm_constant(
        alpha, low, high
    )


@jax.custom_jvp
def doubly_truncated_power_law_cdf(x, alpha, low, high):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L545-L565
    r"""Cumulative distribution function of a doubly truncated power law.

    .. math::
        F(x; \alpha, a, b) = \begin{cases}
            \dfrac{\log(x/a)}{\log(b/a)} & \alpha = -1, \\[2ex]
            \dfrac{x^{1+\alpha} - a^{1+\alpha}}{b^{1+\alpha} - a^{1+\alpha}}
            & \text{otherwise}.
        \end{cases}

    Parameters
    ----------
    x : ArrayLike
        Points at which to evaluate the CDF.
    alpha : ArrayLike
        Power law index :math:`\alpha`.
    low : ArrayLike
        Lower bound :math:`a`.
    high : ArrayLike
        Upper bound :math:`b`.

    Returns
    -------
    ArrayLike
        The CDF, clipped to :math:`[0, 1]`.
    """
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def cdf_when_alpha_neq_neg1():
        r"""CDF on the generic :math:`\alpha \neq -1` branch."""
        one_more_alpha = 1.0 + neq_neg1_alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        return (jnp.power(x, one_more_alpha) - low_pow_one_more_alpha) / (
            jnp.power(high, one_more_alpha) - low_pow_one_more_alpha
        )

    def cdf_when_alpha_eq_neg1():
        r"""CDF on the :math:`\alpha = -1` branch, where the law is log-uniform."""
        return jnp.log(x / low) / jnp.log(high / low)

    cdf_val = jnp.where(
        neq_neg1_mask,
        cdf_when_alpha_neq_neg1(),
        cdf_when_alpha_eq_neg1(),
    )
    return jnp.clip(cdf_val, 0.0, 1.0)


@doubly_truncated_power_law_cdf.defjvp
def doubly_truncated_power_law_cdf_jvp(primals, tangents):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L567-L661
    r"""JVP rule for :func:`doubly_truncated_power_law_cdf`.

    Tangents with respect to ``x``, ``low`` and ``high`` are exact on both branches; the
    ``alpha`` tangent at :math:`\alpha = -1` is a symmetric finite difference.

    Parameters
    ----------
    primals : tuple
        ``(x, alpha, low, high)``.
    tangents : tuple
        The corresponding tangent vectors.

    Returns
    -------
    tuple
        The primal output and its tangent.
    """
    x, alpha, low, high = primals
    x_t, alpha_t, low_t, high_t = tangents

    log_low = jnp.log(low)
    log_high = jnp.log(high)
    log_x = jnp.log(x)

    delta_eq_neg1 = 10e-4
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    # Calculate primal
    primal_out = doubly_truncated_power_law_cdf(*primals)

    # Tangents for alpha not equals -1
    def x_neq_neg1(alpha):
        r"""Exact :math:`\partial F/\partial x` for :math:`\alpha \neq -1`."""
        one_more_alpha = 1.0 + alpha
        return (one_more_alpha * jnp.power(x, alpha)) / (
            jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha)
        )

    def alpha_neq_neg1(alpha):
        r"""Exact :math:`\partial F/\partial \alpha` for :math:`\alpha \neq -1`."""
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
        term1 = (x_pow_one_more_alpha * log_x - low_pow_one_more_alpha * log_low) / (
            high_pow_one_more_alpha - low_pow_one_more_alpha
        )
        term2 = (
            (x_pow_one_more_alpha - low_pow_one_more_alpha)
            * (high_pow_one_more_alpha * log_high - low_pow_one_more_alpha * log_low)
        ) / jnp.square(high_pow_one_more_alpha - low_pow_one_more_alpha)
        return term1 - term2

    def low_neq_neg1(alpha):
        r"""Exact :math:`\partial F/\partial a` for :math:`\alpha \neq -1`."""
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
        change = high_pow_one_more_alpha - low_pow_one_more_alpha
        term2 = one_more_alpha * jnp.power(low, alpha) / change
        term1 = term2 * (x_pow_one_more_alpha - low_pow_one_more_alpha) / change
        return term1 - term2

    def high_neq_neg1(alpha):
        r"""Exact :math:`\partial F/\partial b` for :math:`\alpha \neq -1`."""
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
        return -(
            one_more_alpha
            * jnp.power(high, alpha)
            * (x_pow_one_more_alpha - low_pow_one_more_alpha)
        ) / jnp.square(high_pow_one_more_alpha - low_pow_one_more_alpha)

    # Tangents for alpha equals -1
    def x_eq_neg1():
        r"""Exact :math:`\partial F/\partial x` at :math:`\alpha = -1`."""
        return jnp.reciprocal(x * (log_high - log_low))

    def low_eq_neg1():
        r"""Exact :math:`\partial F/\partial a` at :math:`\alpha = -1`."""
        return (log_x - log_low) / (
            jnp.square(log_high - log_low) * low
        ) - jnp.reciprocal((log_high - log_low) * low)

    def high_eq_neg1():
        r"""Exact :math:`\partial F/\partial b` at :math:`\alpha = -1`."""
        return -(log_x - log_low) / (jnp.square(log_high - log_low) * high)

    # Including approximation for alpha = -1
    tangent_out = (
        jnp.where(neq_neg1_mask, x_neq_neg1(neq_neg1_alpha), x_eq_neg1()) * x_t
        + jnp.where(
            neq_neg1_mask,
            alpha_neq_neg1(neq_neg1_alpha),
            (
                alpha_neq_neg1(alpha - delta_eq_neg1)
                + alpha_neq_neg1(alpha + delta_eq_neg1)
            )
            * 0.5,
        )
        * alpha_t
        + jnp.where(neq_neg1_mask, low_neq_neg1(neq_neg1_alpha), low_eq_neg1()) * low_t
        + jnp.where(neq_neg1_mask, high_neq_neg1(neq_neg1_alpha), high_eq_neg1())
        * high_t
    )

    return primal_out, tangent_out


@jax.custom_jvp
def doubly_truncated_power_law_icdf(q, alpha, low, high):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L680-L703
    r"""Inverse cumulative distribution function of a doubly truncated power law.

    .. math::
        F^{-1}(q; \alpha, a, b) = \begin{cases}
            a\left(\dfrac{b}{a}\right)^{q} & \alpha = -1, \\[2ex]
            \left(a^{1+\alpha} + q\left(b^{1+\alpha} - a^{1+\alpha}\right)\right)^{
            \frac{1}{1+\alpha}} & \text{otherwise}.
        \end{cases}

    Parameters
    ----------
    q : ArrayLike
        Quantiles in :math:`[0, 1]`.
    alpha : ArrayLike
        Power law index :math:`\alpha`.
    low : ArrayLike
        Lower bound :math:`a`.
    high : ArrayLike
        Upper bound :math:`b`.

    Returns
    -------
    ArrayLike
        The corresponding quantile values in :math:`[a, b]`.
    """
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def icdf_alpha_neq_neg1():
        r"""Inverse CDF on the generic :math:`\alpha \neq -1` branch."""
        one_more_alpha = 1.0 + neq_neg1_alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return jnp.power(
            low_pow_one_more_alpha
            + q * (high_pow_one_more_alpha - low_pow_one_more_alpha),
            jnp.reciprocal(one_more_alpha),
        )

    def icdf_alpha_eq_neg1():
        r"""Inverse CDF on the :math:`\alpha = -1` branch, where the law is log-uniform."""
        return jnp.power(high / low, q) * low

    icdf_val = jnp.where(
        neq_neg1_mask,
        icdf_alpha_neq_neg1(),
        icdf_alpha_eq_neg1(),
    )
    return icdf_val


@doubly_truncated_power_law_icdf.defjvp
def doubly_truncated_power_law_icdf_jvp(primals, tangents):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L705-L815
    r"""JVP rule for :func:`doubly_truncated_power_law_icdf`.

    Tangents with respect to the quantile, ``low`` and ``high`` are exact on both
    branches; the ``alpha`` tangent at :math:`\alpha = -1` is a symmetric finite
    difference.

    Parameters
    ----------
    primals : tuple
        ``(q, alpha, low, high)``.
    tangents : tuple
        The corresponding tangent vectors.

    Returns
    -------
    tuple
        The primal output and its tangent.
    """
    x, alpha, low, high = primals
    x_t, alpha_t, low_t, high_t = tangents

    log_low = jnp.log(low)
    log_high = jnp.log(high)
    high_over_low = jnp.divide(high, low)

    delta_eq_neg1 = 10e-4
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    primal_out = doubly_truncated_power_law_icdf(*primals)

    # Tangents for alpha not equal -1
    def x_neq_neg1(alpha):
        r"""Exact :math:`\partial F^{-1}/\partial q` for :math:`\alpha \neq -1`."""
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        change = high_pow_one_more_alpha - low_pow_one_more_alpha
        return (
            change
            * jnp.power(
                low_pow_one_more_alpha + x * change,
                jnp.reciprocal(one_more_alpha) - 1,
            )
        ) / one_more_alpha

    def alpha_neq_neg1(alpha):
        r"""Exact :math:`\partial F^{-1}/\partial \alpha` for :math:`\alpha \neq -1`."""
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        factor0 = low_pow_one_more_alpha + x * (
            high_pow_one_more_alpha - low_pow_one_more_alpha
        )
        term1 = jnp.power(factor0, jnp.reciprocal(one_more_alpha))
        term2 = (
            low_pow_one_more_alpha * log_low
            + x
            * (high_pow_one_more_alpha * log_high - low_pow_one_more_alpha * log_low)
        ) / (one_more_alpha * factor0)
        term3 = jnp.log(factor0) / jnp.square(one_more_alpha)
        return term1 * (term2 - term3)

    def low_neq_neg1(alpha):
        r"""Exact :math:`\partial F^{-1}/\partial a` for :math:`\alpha \neq -1`."""
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return (
            (1.0 - x)
            * jnp.power(low, alpha)
            * jnp.power(
                low_pow_one_more_alpha
                + x * (high_pow_one_more_alpha - low_pow_one_more_alpha),
                jnp.reciprocal(one_more_alpha) - 1,
            )
        )

    def high_neq_neg1(alpha):
        r"""Exact :math:`\partial F^{-1}/\partial b` for :math:`\alpha \neq -1`."""
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return (
            x
            * jnp.power(high, alpha)
            * jnp.power(
                low_pow_one_more_alpha
                + x * (high_pow_one_more_alpha - low_pow_one_more_alpha),
                jnp.reciprocal(one_more_alpha) - 1,
            )
        )

    # Tangents for alpha equals -1
    def dx_eq_neg1():
        r"""Exact :math:`\partial F^{-1}/\partial q` at :math:`\alpha = -1`."""
        return low * jnp.power(high_over_low, x) * (log_high - log_low)

    def low_eq_neg1():
        r"""Exact :math:`\partial F^{-1}/\partial a` at :math:`\alpha = -1`."""
        return (
            jnp.power(high_over_low, x)
            - (high * x * jnp.power(high_over_low, x - 1)) / low
        )

    def high_eq_neg1():
        r"""Exact :math:`\partial F^{-1}/\partial b` at :math:`\alpha = -1`."""
        return x * jnp.power(high_over_low, x - 1)

    # Including approximation for alpha = -1 \
    tangent_out = (
        jnp.where(neq_neg1_mask, x_neq_neg1(neq_neg1_alpha), dx_eq_neg1()) * x_t
        + jnp.where(
            neq_neg1_mask,
            alpha_neq_neg1(neq_neg1_alpha),
            (
                alpha_neq_neg1(alpha - delta_eq_neg1)
                + alpha_neq_neg1(alpha + delta_eq_neg1)
            )
            * 0.5,
        )
        * alpha_t
        + jnp.where(neq_neg1_mask, low_neq_neg1(neq_neg1_alpha), low_eq_neg1()) * low_t
        + jnp.where(neq_neg1_mask, high_neq_neg1(neq_neg1_alpha), high_eq_neg1())
        * high_t
    )

    return primal_out, tangent_out


class DoublyTruncatedPowerLaw(Distribution):
    r"""Power law distribution with index :math:`\alpha` and lower and upper bounds.

    The density is

    .. math::
        f(x; \alpha, a, b) = \frac{x^{\alpha}}{Z(\alpha, a, b)},

    where :math:`a` and :math:`b` are the lower and upper bounds respectively, and
    :math:`Z(\alpha, a, b)` is the normalization constant. It is defined as

    .. math::
        Z(\alpha, a, b) = \begin{cases}
            \log(b) - \log(a) & \text{if } \alpha = -1, \\
            \frac{b^{1 + \alpha} - a^{1 + \alpha}}{1 + \alpha} & \text{otherwise}.
        \end{cases}

    Density, CDF and inverse CDF all delegate to the module-level functions, which carry
    custom JVP rules so that the :math:`\alpha = -1` case differentiates cleanly.

    Parameters
    ----------
    alpha : ArrayLike
        Index :math:`\alpha` of the power law distribution.
    low : ArrayLike
        Lower bound :math:`a` of the distribution.
    high : ArrayLike
        Upper bound :math:`b` of the distribution.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.
    """

    arg_constraints = {
        "alpha": constraints.real,
        "low": constraints.greater_than_eq(0),
        "high": constraints.greater_than(0),
    }
    reparametrized_params = ["alpha", "low", "high"]
    pytree_aux_fields = ("_support",)
    pytree_data_fields = ("alpha", "low", "high")

    def __init__(
        self,
        alpha: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ):
        self.alpha, self.low, self.high = promote_shapes(alpha, low, high)
        self._support = constraints.interval(low, high)
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(alpha), jnp.shape(low), jnp.shape(high)
        )
        super(DoublyTruncatedPowerLaw, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        """The support of the distribution.

        Returns
        -------
        constraints.Constraint
            The interval :math:`[a, b]`.
        """
        return self._support

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        r"""Log probability density at ``value``.

        Parameters
        ----------
        value : ArrayLike
            Points at which to evaluate the density.

        Returns
        -------
        ArrayLike
            :math:`\alpha \log x - \log Z(\alpha, a, b)`.

        See Also
        --------
        doubly_truncated_power_law_log_prob : The underlying implementation.
        """
        return doubly_truncated_power_law_log_prob(
            value, self.alpha, self.low, self.high
        )

    def cdf(self, value: ArrayLike) -> ArrayLike:
        """Cumulative distribution function at ``value``.

        Parameters
        ----------
        value : ArrayLike
            Points at which to evaluate the CDF.

        Returns
        -------
        ArrayLike
            The CDF, clipped to :math:`[0, 1]`.

        See Also
        --------
        doubly_truncated_power_law_cdf : The underlying implementation.
        """
        return doubly_truncated_power_law_cdf(value, self.alpha, self.low, self.high)

    def icdf(self, q: ArrayLike) -> ArrayLike:
        """Inverse cumulative distribution function at ``q``.

        Parameters
        ----------
        q : ArrayLike
            Quantiles in :math:`[0, 1]`.

        Returns
        -------
        ArrayLike
            The corresponding values in :math:`[a, b]`.

        See Also
        --------
        doubly_truncated_power_law_icdf : The underlying implementation.
        """
        return doubly_truncated_power_law_icdf(q, self.alpha, self.low, self.high)

    def sample(
        self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        """Draw samples by inverse transform sampling.

        Parameters
        ----------
        key : jax.dtypes.prng_key
            JAX random key.
        sample_shape : tuple[int, ...]
            Shape of the sample batch to draw. Defaults to ``()``.

        Returns
        -------
        ArrayLike
            Samples of shape ``sample_shape + batch_shape``.
        """
        assert is_prng_key(key)
        u = jax.random.uniform(key, sample_shape + self.batch_shape)
        samples = self.icdf(u)
        return samples
