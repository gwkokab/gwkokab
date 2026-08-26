# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.utils._doubletruncpowerlaw`."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose

from gwkokab.models.utils import (
    doubly_truncated_power_law_cdf,
    doubly_truncated_power_law_icdf,
    doubly_truncated_power_law_log_norm_constant,
    doubly_truncated_power_law_log_prob,
    DoublyTruncatedPowerLaw,
)


# alpha == -1 is the removable singularity of the normalisation constant and is
# special-cased throughout the module, so it appears in every parametrisation.
ALPHAS = [-3.0, -1.0 - 1e-3, -1.0, -1.0 + 1e-3, 0.0, 0.5, 2.0]
BOUNDS = [(2.0, 10.0), (1e-2, 1.0), (5.0, 5.5), (1.0, 1e3)]


###############################################################################
# the normalisation constant
###############################################################################


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_log_norm_constant_matches_the_quadrature(alpha, low, high):
    # integrate in log space: the integrand spans several decades for the steeper
    # indices, which a linear grid resolves badly
    log_x = jnp.linspace(jnp.log(low), jnp.log(high), 200_001)
    x = jnp.exp(log_x)
    integral = jnp.trapezoid(jnp.power(x, alpha) * x, log_x)
    log_norm = doubly_truncated_power_law_log_norm_constant(alpha, low, high)
    assert_allclose(jnp.exp(log_norm), integral, rtol=1e-6)


@pytest.mark.parametrize("low, high", BOUNDS)
def test_log_norm_constant_at_the_singular_index(low, high):
    # alpha == -1 integrates to log(high) - log(low)
    log_norm = doubly_truncated_power_law_log_norm_constant(-1.0, low, high)
    assert_allclose(log_norm, jnp.log(jnp.log(high) - jnp.log(low)), rtol=1e-12)


@pytest.mark.parametrize("low, high", BOUNDS)
def test_log_norm_constant_is_continuous_at_the_singular_index(low, high):
    at = doubly_truncated_power_law_log_norm_constant(-1.0, low, high)
    around = [
        doubly_truncated_power_law_log_norm_constant(-1.0 + eps, low, high)
        for eps in (-1e-6, 1e-6)
    ]
    assert_allclose(around, [at, at], rtol=1e-5)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_log_norm_constant_gradients_match_finite_differences(alpha, low, high):
    args = (alpha, low, high)
    grads = jax.grad(doubly_truncated_power_law_log_norm_constant, argnums=(0, 1, 2))(
        *args
    )
    steps = (1e-5, 1e-6 * low, 1e-6 * high)
    for i, (grad, step) in enumerate(zip(grads, steps)):
        forward = list(args)
        backward = list(args)
        forward[i] += step
        backward[i] -= step
        finite_difference = (
            doubly_truncated_power_law_log_norm_constant(*forward)
            - doubly_truncated_power_law_log_norm_constant(*backward)
        ) / (2.0 * step)
        # the alpha derivative at the singular index is a two-sided approximation
        tolerance = 1e-3 if (i == 0 and alpha == -1.0) else 1e-5
        assert_allclose(grad, finite_difference, rtol=tolerance, atol=1e-8)


###############################################################################
# the density
###############################################################################


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_log_prob_is_normalised(alpha, low, high):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    log_x = jnp.linspace(jnp.log(low), jnp.log(high), 200_001)
    x = jnp.exp(log_x)
    integral = jnp.trapezoid(jnp.exp(model.log_prob(x)) * x, log_x)
    assert_allclose(integral, 1.0, rtol=1e-5)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_log_prob_is_the_powerlaw_shape(alpha):
    low, high = 2.0, 10.0
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    x = jnp.asarray([2.5, 5.0, 9.0])
    assert_allclose(
        model.log_prob(x) - model.log_prob(jnp.asarray(low)),
        alpha * (jnp.log(x) - jnp.log(low)),
        rtol=1e-10,
    )


def test_flat_index_is_uniform():
    model = DoublyTruncatedPowerLaw(alpha=0.0, low=2.0, high=10.0)
    x = jnp.asarray([2.0, 5.0, 10.0])
    assert_allclose(model.log_prob(x), jnp.full(3, -jnp.log(8.0)), rtol=1e-12)


def test_singular_index_is_log_uniform():
    low, high = 2.0, 10.0
    model = DoublyTruncatedPowerLaw(alpha=-1.0, low=low, high=high)
    x = jnp.asarray([2.5, 5.0, 9.0])
    expected = -jnp.log(x) - jnp.log(jnp.log(high) - jnp.log(low))
    assert_allclose(model.log_prob(x), expected, rtol=1e-12)


def test_log_prob_free_function_matches_the_distribution():
    model = DoublyTruncatedPowerLaw(alpha=1.5, low=2.0, high=10.0)
    x = jnp.asarray([2.5, 5.0, 9.0])
    assert_allclose(
        doubly_truncated_power_law_log_prob(x, 1.5, 2.0, 10.0),
        model.log_prob(x),
        rtol=1e-12,
    )


###############################################################################
# cdf and icdf
###############################################################################


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_cdf_endpoints(alpha, low, high):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    assert_allclose(model.cdf(jnp.asarray(low)), 0.0, atol=1e-12)
    assert_allclose(model.cdf(jnp.asarray(high)), 1.0, atol=1e-12)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_cdf_is_monotonic_and_clipped(alpha, low, high):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    x = jnp.linspace(low, high, 501)
    cdf = model.cdf(x)
    assert jnp.all(jnp.diff(cdf) >= 0.0)
    assert jnp.all((cdf >= 0.0) & (cdf <= 1.0))
    assert_allclose(model.cdf(jnp.asarray(0.5 * low)), 0.0, atol=1e-12)
    assert_allclose(model.cdf(jnp.asarray(2.0 * high)), 1.0, atol=1e-12)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_cdf_is_the_integral_of_the_density(alpha, low, high):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    log_x = jnp.linspace(jnp.log(low), jnp.log(high), 100_001)
    x = jnp.exp(log_x)
    integrand = jnp.exp(model.log_prob(x)) * x
    cumulative = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(log_x)),
    ])
    assert_allclose(model.cdf(x), cumulative, atol=1e-5)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_icdf_inverts_the_cdf(alpha, low, high):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    x = jnp.linspace(low, high, 501)[1:-1]
    assert_allclose(model.icdf(model.cdf(x)), x, rtol=1e-8)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_cdf_inverts_the_icdf(alpha, low, high):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    q = jnp.linspace(0.0, 1.0, 501)
    assert_allclose(model.cdf(model.icdf(q)), q, atol=1e-8)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_icdf_endpoints(alpha, low, high):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    assert_allclose(model.icdf(jnp.asarray(0.0)), low, rtol=1e-10)
    assert_allclose(model.icdf(jnp.asarray(1.0)), high, rtol=1e-10)


@pytest.mark.parametrize("argnum", range(4))
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize(
    "fn", [doubly_truncated_power_law_cdf, doubly_truncated_power_law_icdf]
)
def test_cdf_and_icdf_gradients_match_finite_differences(fn, alpha, argnum):
    low, high = 2.0, 10.0
    first = 0.4 if fn is doubly_truncated_power_law_icdf else 5.0
    args = (first, alpha, low, high)
    grad = jax.grad(fn, argnums=argnum)(*args)
    step = 1e-6
    forward, backward = list(args), list(args)
    forward[argnum] += step
    backward[argnum] -= step
    finite_difference = (fn(*forward) - fn(*backward)) / (2.0 * step)
    # at the singular index the parameter derivative is a two-sided approximation
    tolerance = 2e-3 if alpha == -1.0 else 1e-4
    assert_allclose(grad, finite_difference, rtol=tolerance, atol=1e-6)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_cdf_gradient_signs(alpha):
    r"""The cdf rises in :math:`x` and falls as either bound is pushed outwards.

    Widening the support in either direction moves mass past a fixed :math:`x`, so
    :math:`\partial F/\partial a` and :math:`\partial F/\partial b` are both
    negative. This pins the sign of every branch of the custom JVP, including the two
    that are special-cased at :math:`\alpha = -1`.
    """
    d_x, d_alpha, d_low, d_high = jax.grad(
        doubly_truncated_power_law_cdf, argnums=(0, 1, 2, 3)
    )(5.0, alpha, 2.0, 10.0)
    del d_alpha  # its sign depends on which side of the median x sits
    assert d_x > 0.0
    assert d_low < 0.0
    assert d_high < 0.0


###############################################################################
# the distribution wrapper
###############################################################################


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("low, high", BOUNDS)
def test_shapes_and_support(alpha, low, high):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=low, high=high)
    assert model.batch_shape == ()
    assert model.event_shape == ()
    assert model.support.lower_bound == low
    assert model.support.upper_bound == high


@pytest.mark.parametrize("sample_shape", [(), (5,), (2, 3)])
def test_sample_shape(sample_shape):
    model = DoublyTruncatedPowerLaw(alpha=1.5, low=2.0, high=10.0)
    assert model.sample(jrd.key(0), sample_shape).shape == model.shape(sample_shape)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_samples_lie_in_the_support(alpha):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=2.0, high=10.0)
    assert jnp.all(model.support.check(model.sample(jrd.key(1), (8192,))))


@pytest.mark.parametrize("alpha", [-3.0, -1.0, 0.0, 2.0])
def test_samples_follow_the_cdf(alpha):
    model = DoublyTruncatedPowerLaw(alpha=alpha, low=2.0, high=10.0)
    samples = np.asarray(model.sample(jrd.key(2), (100_000,)))
    probe = np.linspace(2.0, 10.0, 21)
    empirical = np.mean(samples[:, None] <= probe[None, :], axis=0)
    assert_allclose(empirical, np.asarray(model.cdf(jnp.asarray(probe))), atol=6e-3)


def test_parameters_broadcast():
    model = DoublyTruncatedPowerLaw(alpha=jnp.asarray([-1.0, 2.0]), low=2.0, high=10.0)
    assert model.batch_shape == (2,)
    value = jnp.asarray(5.0)
    log_prob = model.log_prob(value)
    assert log_prob.shape == (2,)
    for i, alpha in enumerate([-1.0, 2.0]):
        expected = DoublyTruncatedPowerLaw(alpha=alpha, low=2.0, high=10.0).log_prob(
            value
        )
        assert_allclose(log_prob[i], expected, rtol=1e-12)
    assert model.sample(jrd.key(0), (3,)).shape == (3, 2)


def test_pytree_roundtrip(pytree_roundtrip):
    model = DoublyTruncatedPowerLaw(alpha=1.5, low=2.0, high=10.0)
    value = jnp.asarray(5.0)
    assert_allclose(pytree_roundtrip(model).log_prob(value), model.log_prob(value))


def test_under_jit_and_vmap():
    alphas = jnp.asarray([-2.0, -1.0, 0.5])
    value = jnp.asarray(5.0)

    def log_prob(alpha):
        return DoublyTruncatedPowerLaw(alpha=alpha, low=2.0, high=10.0).log_prob(value)

    assert_allclose(
        jax.jit(jax.vmap(log_prob))(alphas),
        jnp.stack([log_prob(a) for a in alphas]),
        rtol=1e-12,
    )


def test_warns_outside_the_support():
    model = DoublyTruncatedPowerLaw(alpha=1.5, low=2.0, high=10.0, validate_args=True)
    with pytest.warns(UserWarning, match="Out-of-support values"):
        model.log_prob(jnp.asarray(0.5))


def test_rejects_a_negative_lower_bound():
    with pytest.raises(ValueError):
        DoublyTruncatedPowerLaw(alpha=1.5, low=-1.0, high=10.0, validate_args=True)


def test_sample_rejects_a_plain_integer_key():
    model = DoublyTruncatedPowerLaw(alpha=1.5, low=2.0, high=10.0)
    with pytest.raises(AssertionError):
        model.sample(0)
