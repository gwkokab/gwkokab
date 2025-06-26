# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import jax
from jax import numpy as jnp


@jax.custom_jvp
def doubly_truncated_power_law_log_norm_constant(alpha, low, high):
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def neq_neg1_fn():
        one_more_alpha = 1.0 + neq_neg1_alpha
        return jnp.log(
            (jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha))
            / (one_more_alpha)
        )

    def eq_neg1_fn():
        return jnp.log(jnp.log(high) - jnp.log(low))

    return jnp.where(neq_neg1_mask, neq_neg1_fn(), eq_neg1_fn())


@doubly_truncated_power_law_log_norm_constant.defjvp
def doubly_truncated_power_law_log_norm_constant_jvp(primals, tangents):
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
    return alpha * jnp.log(x) - doubly_truncated_power_law_log_norm_constant(
        alpha, low, high
    )


@jax.custom_jvp
def doubly_truncated_power_law_cdf(x, alpha, low, high):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L545-L565
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def cdf_when_alpha_neq_neg1():
        one_more_alpha = 1.0 + neq_neg1_alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        return (jnp.power(x, one_more_alpha) - low_pow_one_more_alpha) / (
            jnp.power(high, one_more_alpha) - low_pow_one_more_alpha
        )

    def cdf_when_alpha_eq_neg1():
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
        one_more_alpha = 1.0 + alpha
        return (one_more_alpha * jnp.power(x, alpha)) / (
            jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha)
        )

    def alpha_neq_neg1(alpha):
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
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
        change = high_pow_one_more_alpha - low_pow_one_more_alpha
        term2 = one_more_alpha * jnp.power(low, alpha) / change
        term1 = term2 * (x_pow_one_more_alpha - low_pow_one_more_alpha) / change
        return term1 - term2

    def high_neq_neg1(alpha):
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
        return jnp.reciprocal(x * (log_high - log_low))

    def low_eq_neg1():
        return (log_x - log_low) / (
            jnp.square(log_high - log_low) * low
        ) - jnp.reciprocal((log_high - log_low) * low)

    def high_eq_neg1():
        return (log_x - log_low) / (jnp.square(log_high - log_low) * high)

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
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def icdf_alpha_neq_neg1():
        one_more_alpha = 1.0 + neq_neg1_alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return jnp.power(
            low_pow_one_more_alpha
            + q * (high_pow_one_more_alpha - low_pow_one_more_alpha),
            jnp.reciprocal(one_more_alpha),
        )

    def icdf_alpha_eq_neg1():
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
        return low * jnp.power(high_over_low, x) * (log_high - log_low)

    def low_eq_neg1():
        return (
            jnp.power(high_over_low, x)
            - (high * x * jnp.power(high_over_low, x - 1)) / low
        )

    def high_eq_neg1():
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
