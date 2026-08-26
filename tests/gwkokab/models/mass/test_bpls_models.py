# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.mass._bpls`."""

import warnings

import jax
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from gwkokab.models.mass import (
    BrokenPowerlaw,
    BrokenPowerlawTwoPeak,
    SmoothedBrokenPowerlawMassRatioPowerlaw,
)
from gwkokab.models.mass._bpls import _broken_powerlaw_log_prob
from gwkokab.models.utils import DoublyTruncatedPowerLaw
from gwkokab.utils.kernel import log_planck_taper_window
from gwkokab.utils.math import truncnorm_logpdf


BROKEN_POWERLAW_PARAMS = [
    dict(alpha1=-1.0, alpha2=2.0, mbreak=30.0, mmax=50.0, mmin=10.0),
    dict(alpha1=0.5, alpha2=3.5, mbreak=70.0, mmax=100.0, mmin=50.0),
    dict(alpha1=-2.0, alpha2=1.0, mbreak=20.0, mmax=100.0, mmin=5.0),
    dict(alpha1=2.5, alpha2=-0.5, mbreak=60.0, mmax=80.0, mmin=40.0),
]

TWO_PEAK_PARAMS = dict(
    alpha1=1.0,
    alpha2=3.0,
    beta=1.0,
    loc1=10.0,
    loc2=35.0,
    scale1=2.0,
    scale2=5.0,
    delta_m1=3.0,
    delta_m2=3.0,
    lambda_0=0.5,
    lambda_1=0.3,
    m1min=5.0,
    m2min=5.0,
    mmax=100.0,
    mbreak=30.0,
)

SMOOTHED_BROKEN_PARAMS = dict(
    alpha1=1.0,
    alpha2=3.0,
    beta=1.0,
    mbreak=30.0,
    mmax=100.0,
    m1min=5.0,
    m2min=5.0,
    delta_m1=3.0,
    delta_m2=3.0,
)

TWO_DIMENSIONAL_MODELS = [
    (BrokenPowerlawTwoPeak, TWO_PEAK_PARAMS),
    (SmoothedBrokenPowerlawMassRatioPowerlaw, SMOOTHED_BROKEN_PARAMS),
]


###############################################################################
# BrokenPowerlaw
###############################################################################


@pytest.mark.parametrize("params", BROKEN_POWERLAW_PARAMS)
def test_broken_powerlaw_shapes_and_support(params):
    model = BrokenPowerlaw(**params)
    assert model.batch_shape == ()
    assert model.event_shape == ()
    assert model.support.lower_bound == params["mmin"]
    assert model.support.upper_bound == params["mmax"]


@pytest.mark.parametrize("params", BROKEN_POWERLAW_PARAMS)
def test_broken_powerlaw_is_normalised(params):
    model = BrokenPowerlaw(**params)
    x = jnp.linspace(params["mmin"], params["mmax"], 40_001)
    assert_allclose(jnp.trapezoid(jnp.exp(model.log_prob(x)), x), 1.0, atol=1e-5)


@pytest.mark.parametrize("params", BROKEN_POWERLAW_PARAMS)
def test_broken_powerlaw_is_continuous_at_the_break(params):
    model = BrokenPowerlaw(**params)
    mbreak = params["mbreak"]
    below = model.log_prob(jnp.asarray(mbreak * (1.0 - 1e-9)))
    above = model.log_prob(jnp.asarray(mbreak * (1.0 + 1e-9)))
    assert_allclose(below, above, rtol=1e-8)


@pytest.mark.parametrize("params", BROKEN_POWERLAW_PARAMS)
def test_broken_powerlaw_slopes(params):
    model = BrokenPowerlaw(**params)
    mbreak = params["mbreak"]
    at_break = model.log_prob(jnp.asarray(mbreak))
    below = jnp.asarray(0.5 * (params["mmin"] + mbreak))
    above = jnp.asarray(0.5 * (mbreak + params["mmax"]))
    assert_allclose(
        model.log_prob(below) - at_break,
        params["alpha1"] * jnp.log(mbreak / below),
        rtol=1e-10,
    )
    assert_allclose(
        model.log_prob(above) - at_break,
        params["alpha2"] * jnp.log(mbreak / above),
        rtol=1e-10,
    )


@pytest.mark.parametrize("alpha", [-1.0, 0.5, 2.0])
def test_broken_powerlaw_with_equal_slopes_is_a_plain_powerlaw(alpha):
    broken = BrokenPowerlaw(
        alpha1=alpha, alpha2=alpha, mbreak=30.0, mmax=50.0, mmin=10.0
    )
    plain = DoublyTruncatedPowerLaw(alpha=-alpha, low=10.0, high=50.0)
    x = jnp.asarray([12.0, 30.0, 45.0])
    assert_allclose(broken.log_prob(x), plain.log_prob(x), rtol=1e-10)


def test_broken_powerlaw_broadcasts_its_parameters():
    model = BrokenPowerlaw(
        alpha1=jnp.asarray([-1.0, 0.5]), alpha2=2.0, mbreak=30.0, mmax=50.0, mmin=10.0
    )
    assert model.batch_shape == (2,)
    value = jnp.asarray(20.0)
    log_prob = model.log_prob(value)
    assert log_prob.shape == (2,)
    for i, alpha1 in enumerate([-1.0, 0.5]):
        expected = BrokenPowerlaw(
            alpha1=alpha1, alpha2=2.0, mbreak=30.0, mmax=50.0, mmin=10.0
        ).log_prob(value)
        assert_allclose(log_prob[i], expected, rtol=1e-12)


def test_broken_powerlaw_does_not_implement_sampling():
    model = BrokenPowerlaw(**BROKEN_POWERLAW_PARAMS[0])
    with pytest.raises(NotImplementedError):
        model.sample(jax.random.key(0))


def test_broken_powerlaw_warns_outside_its_support():
    model = BrokenPowerlaw(**BROKEN_POWERLAW_PARAMS[0], validate_args=True)
    with pytest.warns(UserWarning, match="Out-of-support values"):
        model.log_prob(jnp.asarray(1.0))


def test_broken_powerlaw_log_prob_helper_is_normalised_by_construction():
    # the free function is the piece that both BrokenPowerlaw and the two composite
    # models reuse, so pin its normalisation independently
    x = jnp.linspace(10.0, 50.0, 40_001)
    log_prob = _broken_powerlaw_log_prob(x, -1.0, 2.0, 10.0, 50.0, 30.0)
    assert_allclose(jnp.trapezoid(jnp.exp(log_prob), x), 1.0, atol=1e-5)


###############################################################################
# the two-dimensional (m1, q) models
###############################################################################


@pytest.mark.parametrize("model_cls, params", TWO_DIMENSIONAL_MODELS)
def test_two_dimensional_shapes(model_cls, params):
    model = model_cls(**params)
    assert model.batch_shape == ()
    assert model.event_shape == (2,)
    assert model.support.event_dim == 1


@pytest.mark.parametrize("model_cls, params", TWO_DIMENSIONAL_MODELS)
@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_two_dimensional_log_prob_shape(model_cls, params, sample_shape):
    model = model_cls(**params)
    value = jnp.broadcast_to(jnp.asarray([40.0, 0.7]), sample_shape + (2,))
    assert model.log_prob(value).shape == sample_shape


@pytest.mark.parametrize("model_cls, params", TWO_DIMENSIONAL_MODELS)
def test_two_dimensional_normalisation(model_cls, params, trapz_nd):
    model = model_cls(**params)
    integral = trapz_nd(model, [(5.0, 100.0), (1e-6, 1.0)], num=1201)
    assert_allclose(integral, 1.0, atol=2e-3)


@pytest.mark.parametrize("model_cls, params", TWO_DIMENSIONAL_MODELS)
def test_two_dimensional_pytree_roundtrip(model_cls, params, pytree_roundtrip):
    model = model_cls(**params)
    value = jnp.asarray([40.0, 0.7])
    assert_allclose(pytree_roundtrip(model).log_prob(value), model.log_prob(value))


@pytest.mark.parametrize("model_cls, params", TWO_DIMENSIONAL_MODELS)
def test_two_dimensional_under_jit(model_cls, params):
    value = jnp.asarray([40.0, 0.7])

    def log_prob(value):
        return model_cls(**params).log_prob(value)

    assert_allclose(jax.jit(log_prob)(value), log_prob(value), rtol=1e-6)


@pytest.mark.parametrize("model_cls, params", TWO_DIMENSIONAL_MODELS)
def test_two_dimensional_rejects_out_of_support_values(model_cls, params):
    model = model_cls(**params)
    outside = jnp.asarray([[1.0, 0.5], [40.0, 1.5], [200.0, 0.5], [40.0, 0.01]])
    assert not jnp.any(model.support.check(outside))


@pytest.mark.parametrize("model_cls, params", TWO_DIMENSIONAL_MODELS)
def test_two_dimensional_needs_a_scalar_batch_shape(model_cls, params):
    with pytest.raises(ValueError):
        model_cls(**{**params, "alpha1": jnp.asarray([1.0, 2.0])})


@pytest.mark.parametrize("model_cls, params", TWO_DIMENSIONAL_MODELS)
@pytest.mark.parametrize("argname", ["alpha1", "alpha2", "beta"])
def test_two_dimensional_gradients_match_finite_differences(model_cls, params, argname):
    value = jnp.asarray([40.0, 0.7])

    def log_prob(x):
        return model_cls(**{**params, argname: x}).log_prob(value)

    x0 = jnp.asarray(float(params[argname]))
    step = 1e-5
    finite_difference = (log_prob(x0 + step) - log_prob(x0 - step)) / (2.0 * step)
    assert_allclose(jax.grad(log_prob)(x0), finite_difference, rtol=1e-4, atol=1e-6)


###############################################################################
# BrokenPowerlawTwoPeak specifics
###############################################################################


def test_two_peak_reduces_to_the_broken_powerlaw_when_both_peaks_are_off():
    model = BrokenPowerlawTwoPeak(**{
        **TWO_PEAK_PARAMS,
        "lambda_0": 1.0,
        "lambda_1": 0.0,
    })
    m1 = jnp.asarray([10.0, 30.0, 70.0])
    expected = _broken_powerlaw_log_prob(
        m1,
        TWO_PEAK_PARAMS["alpha1"],
        TWO_PEAK_PARAMS["alpha2"],
        TWO_PEAK_PARAMS["m1min"],
        TWO_PEAK_PARAMS["mmax"],
        TWO_PEAK_PARAMS["mbreak"],
    ) + log_planck_taper_window(
        (m1 - TWO_PEAK_PARAMS["m1min"]) / TWO_PEAK_PARAMS["delta_m1"]
    )
    assert_allclose(model._log_prob_m1_unnorm(m1), expected, rtol=1e-10)


@pytest.mark.parametrize(
    "lambda_0, lambda_1, loc_key, scale_key",
    [(0.0, 1.0, "loc1", "scale1"), (0.0, 0.0, "loc2", "scale2")],
)
def test_two_peak_reduces_to_a_single_gaussian(lambda_0, lambda_1, loc_key, scale_key):
    model = BrokenPowerlawTwoPeak(**{
        **TWO_PEAK_PARAMS,
        "lambda_0": lambda_0,
        "lambda_1": lambda_1,
    })
    m1 = jnp.asarray([10.0, 30.0, 70.0])
    expected = truncnorm_logpdf(
        xx=m1,
        loc=TWO_PEAK_PARAMS[loc_key],
        scale=TWO_PEAK_PARAMS[scale_key],
        low=TWO_PEAK_PARAMS["m1min"],
        high=TWO_PEAK_PARAMS["mmax"],
    ) + log_planck_taper_window(
        (m1 - TWO_PEAK_PARAMS["m1min"]) / TWO_PEAK_PARAMS["delta_m1"]
    )
    assert_allclose(model._log_prob_m1_unnorm(m1), expected, rtol=1e-10)


def test_two_peak_mixture_weights_are_a_convex_combination():
    m1 = jnp.asarray([12.0, 40.0, 80.0])
    mixed = BrokenPowerlawTwoPeak(**TWO_PEAK_PARAMS)._log_prob_m1_unnorm(m1)
    pieces = [
        (
            TWO_PEAK_PARAMS["lambda_0"],
            BrokenPowerlawTwoPeak(**{
                **TWO_PEAK_PARAMS,
                "lambda_0": 1.0,
                "lambda_1": 0.0,
            })._log_prob_m1_unnorm(m1),
        ),
        (
            TWO_PEAK_PARAMS["lambda_1"],
            BrokenPowerlawTwoPeak(**{
                **TWO_PEAK_PARAMS,
                "lambda_0": 0.0,
                "lambda_1": 1.0,
            })._log_prob_m1_unnorm(m1),
        ),
        (
            1.0 - TWO_PEAK_PARAMS["lambda_0"] - TWO_PEAK_PARAMS["lambda_1"],
            BrokenPowerlawTwoPeak(**{
                **TWO_PEAK_PARAMS,
                "lambda_0": 0.0,
                "lambda_1": 0.0,
            })._log_prob_m1_unnorm(m1),
        ),
    ]
    expected = jnp.log(sum(weight * jnp.exp(term) for weight, term in pieces))
    assert_allclose(mixed, expected, rtol=1e-10)


def test_two_peak_non_positive_mass_ratio_smoothing_kills_the_density():
    model = BrokenPowerlawTwoPeak(**{**TWO_PEAK_PARAMS, "delta_m2": 0.0})
    assert jnp.isneginf(model.log_prob(jnp.asarray([40.0, 0.7])))


def test_two_peak_non_positive_primary_smoothing_leaves_no_density():
    # the unnormalised primary density collapses to -inf everywhere, so its numerical
    # normalisation underflows too and the difference is NaN rather than -inf; either
    # way there is no finite density left
    model = BrokenPowerlawTwoPeak(**{**TWO_PEAK_PARAMS, "delta_m1": 0.0})
    assert not jnp.isfinite(model.log_prob(jnp.asarray([40.0, 0.7])))


def test_smoothed_broken_zero_smoothing_scale_degenerates_to_a_hard_cut():
    # unlike the other smoothed models this one has no explicit delta <= 0 guard: the
    # window argument becomes +inf above the edge, so the taper simply disappears
    model = SmoothedBrokenPowerlawMassRatioPowerlaw(**{
        **SMOOTHED_BROKEN_PARAMS,
        "delta_m1": 0.0,
    })
    m1 = jnp.asarray([10.0, 30.0, 70.0])
    expected = _broken_powerlaw_log_prob(
        m1,
        SMOOTHED_BROKEN_PARAMS["alpha1"],
        SMOOTHED_BROKEN_PARAMS["alpha2"],
        SMOOTHED_BROKEN_PARAMS["m1min"],
        SMOOTHED_BROKEN_PARAMS["mmax"],
        SMOOTHED_BROKEN_PARAMS["mbreak"],
    )
    assert_allclose(model._log_prob_m1(m1), expected, rtol=1e-10)


def test_two_peak_vanishes_below_the_primary_minimum():
    model = BrokenPowerlawTwoPeak(**{**TWO_PEAK_PARAMS, "m1min": 20.0})
    assert jnp.isneginf(model.log_prob(jnp.asarray([15.0, 0.9])))


def test_two_peak_validate_args_warns_outside_support():
    with warnings.catch_warnings():
        # __init__ evaluates the mass-ratio grid, which straddles the support boundary
        warnings.simplefilter("ignore", UserWarning)
        model = BrokenPowerlawTwoPeak(**TWO_PEAK_PARAMS, validate_args=True)
    with pytest.warns(UserWarning, match="Out-of-support values"):
        model.log_prob(jnp.asarray([1.0, 0.5]))


###############################################################################
# SmoothedBrokenPowerlawMassRatioPowerlaw specifics
###############################################################################


def test_smoothed_broken_primary_matches_the_broken_powerlaw_times_the_window():
    model = SmoothedBrokenPowerlawMassRatioPowerlaw(**SMOOTHED_BROKEN_PARAMS)
    m1 = jnp.asarray([10.0, 30.0, 70.0])
    expected = _broken_powerlaw_log_prob(
        m1,
        SMOOTHED_BROKEN_PARAMS["alpha1"],
        SMOOTHED_BROKEN_PARAMS["alpha2"],
        SMOOTHED_BROKEN_PARAMS["m1min"],
        SMOOTHED_BROKEN_PARAMS["mmax"],
        SMOOTHED_BROKEN_PARAMS["mbreak"],
    ) + log_planck_taper_window(
        (m1 - SMOOTHED_BROKEN_PARAMS["m1min"]) / SMOOTHED_BROKEN_PARAMS["delta_m1"]
    )
    assert_allclose(model._log_prob_m1(m1), expected, rtol=1e-10)


def test_smoothed_broken_mass_ratio_is_masked_by_the_support():
    model = SmoothedBrokenPowerlawMassRatioPowerlaw(**SMOOTHED_BROKEN_PARAMS)
    # m1 * q is far below m2min, so the mass-ratio factor must vanish
    value = jnp.asarray([10.0, 0.05])
    assert jnp.isneginf(model._log_prob_q(value))


def test_smoothed_broken_primary_is_monotonic_below_the_break():
    model = SmoothedBrokenPowerlawMassRatioPowerlaw(**{
        **SMOOTHED_BROKEN_PARAMS,
        "alpha1": 2.0,
        "delta_m1": 1e-3,
    })
    m1 = jnp.linspace(6.0, 29.0, 25)
    assert jnp.all(jnp.diff(model._log_prob_m1(m1)) < 0.0)
