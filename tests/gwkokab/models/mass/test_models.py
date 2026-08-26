# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.mass._models`."""

import warnings

import jax
import numpy as np
import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose
from numpyro.distributions import Normal

from gwkokab.models.mass import (
    GaussianPrimaryMassRatio,
    GenericSmoothedPowerlawMassRatio,
    PowerlawPrimaryMassRatio,
    SmoothedGaussianPrimaryMassRatio,
    SmoothedPowerlawPrimaryMassRatio,
    SmoothedTwoComponentPrimaryMassRatio,
    Wysocki2019MassModel,
)
from gwkokab.models.utils import (
    doubly_truncated_power_law_log_prob,
    DoublyTruncatedPowerLaw,
)
from gwkokab.utils.kernel import log_planck_taper_window


# Named parameter sets, so the individual tests below can refer to one model's
# parameters without indexing into the collections.
SMOOTHED_POWERLAW = dict(
    alpha=2.0, beta=1.0, delta_m1=3.0, delta_m2=3.0, mmax=100.0, m1min=5.0, m2min=5.0
)
SMOOTHED_GAUSSIAN = dict(
    loc=35.0,
    scale=5.0,
    beta=1.0,
    m1min=5.0,
    m2min=5.0,
    mmax=100.0,
    delta_m1=3.0,
    delta_m2=3.0,
)
TWO_COMPONENT = dict(
    alpha=2.0,
    beta=1.0,
    delta=3.0,
    lambda_peak=0.3,
    loc=35.0,
    mmax=100.0,
    mmin=5.0,
    scale=5.0,
)
POWERLAW = dict(alpha=-1.0, beta=1.0, mmin=10.0, mmax=50.0)
WYSOCKI = dict(alpha_m=-1.0, mmin=10.0, mmax=50.0)
GAUSSIAN = dict(loc=35.0, scale=5.0, beta=1.0, mmin=10.0, mmax=100.0)

# Models whose normalisation is precomputed on a fixed grid at construction time. They
# are only meaningful with a scalar (i.e. empty) batch shape; see the batching tests at
# the bottom of this module.
SPECIFIC_MODELS = [
    (SmoothedPowerlawPrimaryMassRatio, SMOOTHED_POWERLAW),
    (SmoothedGaussianPrimaryMassRatio, SMOOTHED_GAUSSIAN),
    (SmoothedTwoComponentPrimaryMassRatio, TWO_COMPONENT),
]

# Models that broadcast their parameters like any other NumPyro distribution.
GENERAL_MODELS = [
    (PowerlawPrimaryMassRatio, POWERLAW),
    (PowerlawPrimaryMassRatio, dict(alpha=2.0, beta=3.0, mmin=50.0, mmax=70.0)),
    (Wysocki2019MassModel, WYSOCKI),
    (Wysocki2019MassModel, dict(alpha_m=1.5, mmin=10.0, mmax=50.0)),
    (GaussianPrimaryMassRatio, GAUSSIAN),
]

ALL_MODELS = GENERAL_MODELS + SPECIFIC_MODELS


# Wysocki2019MassModel is the only model here whose event is (m1, m2); every other one
# uses (m1, q).
SECOND_COORDINATE_IS_MASS = (Wysocki2019MassModel,)


def _mass_bounds(kwargs) -> tuple[float, float]:
    low = kwargs.get("mmin", kwargs.get("m1min", kwargs.get("m2min")))
    return float(low), float(kwargs["mmax"])


def _inside(model_cls, kwargs, m1, q):
    """Build an in-support event from a primary mass and a mass ratio."""
    m1 = jnp.asarray(m1, dtype=float)
    q = jnp.asarray(q, dtype=float)
    second = m1 * q if model_cls in SECOND_COORDINATE_IS_MASS else q
    return jnp.stack(jnp.broadcast_arrays(m1, second), axis=-1)


###############################################################################
# shapes, supports and pytree behaviour
###############################################################################


@pytest.mark.parametrize("model_cls, kwargs", ALL_MODELS)
def test_shapes(model_cls, kwargs):
    model = model_cls(**kwargs)
    assert model.batch_shape == ()
    assert model.event_shape == (2,)
    assert model.support.event_dim == 1


@pytest.mark.parametrize("model_cls, kwargs", ALL_MODELS)
@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_log_prob_shape(model_cls, kwargs, sample_shape):
    model = model_cls(**kwargs)
    low, high = _mass_bounds(kwargs)
    value = jnp.broadcast_to(
        jnp.asarray([0.5 * (low + high), 0.7]), sample_shape + (2,)
    )
    assert model.log_prob(value).shape == sample_shape


@pytest.mark.parametrize("model_cls, kwargs", ALL_MODELS)
def test_pytree_roundtrip(model_cls, kwargs, pytree_roundtrip):
    model = model_cls(**kwargs)
    low, high = _mass_bounds(kwargs)
    value = _inside(model_cls, kwargs, 0.5 * (low + high), 0.6)
    assert_allclose(pytree_roundtrip(model).log_prob(value), model.log_prob(value))


@pytest.mark.parametrize("model_cls, kwargs", ALL_MODELS)
def test_log_prob_under_jit(model_cls, kwargs):
    low, high = _mass_bounds(kwargs)
    value = _inside(model_cls, kwargs, 0.5 * (low + high), 0.6)

    def log_prob(value):
        return model_cls(**kwargs).log_prob(value)

    assert_allclose(jax.jit(log_prob)(value), log_prob(value), rtol=1e-6)


@pytest.mark.parametrize("model_cls, kwargs", ALL_MODELS)
def test_log_prob_is_finite_inside_support(model_cls, kwargs):
    model = model_cls(**kwargs)
    low, high = _mass_bounds(kwargs)
    m1 = jnp.linspace(low + 0.5 * (high - low), high - 0.01 * (high - low), 17)
    q = jnp.linspace(0.8, 0.99, 17)
    grid_m1, grid_q = jnp.meshgrid(m1, q, indexing="ij")
    log_prob = model.log_prob(_inside(model_cls, kwargs, grid_m1, grid_q))
    assert jnp.all(jnp.isfinite(log_prob)), model_cls.__name__


@pytest.mark.parametrize("model_cls, kwargs", ALL_MODELS)
def test_out_of_support_is_rejected(model_cls, kwargs):
    model = model_cls(**kwargs)
    low, high = _mass_bounds(kwargs)
    # a primary mass below the lower edge and a mass ratio above one are both outside
    # the (m1, q) sandwich for every model in this module
    outside = jnp.asarray([
        [0.5 * low, 0.5],
        [0.5 * (low + high), 1.5],
        [2.0 * high, 0.5],
    ])
    assert not jnp.any(model.support.check(outside))


@pytest.mark.parametrize("model_cls, kwargs", ALL_MODELS)
def test_validate_args_warns_outside_support(model_cls, kwargs):
    with warnings.catch_warnings():
        # SmoothedTwoComponentPrimaryMassRatio validates the internal quadrature grid it
        # builds in __init__, which deliberately straddles the support boundary
        warnings.simplefilter("ignore", UserWarning)
        model = model_cls(**kwargs, validate_args=True)
    low, _ = _mass_bounds(kwargs)
    with pytest.warns(UserWarning, match="Out-of-support values"):
        model.log_prob(jnp.asarray([0.5 * low, 0.5]))


###############################################################################
# normalisation
###############################################################################


@pytest.mark.parametrize(
    "model_cls, kwargs, bounds",
    [
        (
            PowerlawPrimaryMassRatio,
            dict(alpha=-1.0, beta=1.0, mmin=10.0, mmax=50.0),
            [(10.0, 50.0), (1e-6, 1.0)],
        ),
        (
            PowerlawPrimaryMassRatio,
            dict(alpha=2.0, beta=3.0, mmin=50.0, mmax=70.0),
            [(50.0, 70.0), (1e-6, 1.0)],
        ),
        (
            PowerlawPrimaryMassRatio,
            dict(alpha=-1.4, beta=9.0, mmin=5.0, mmax=100.0),
            [(5.0, 100.0), (1e-6, 1.0)],
        ),
        (
            GaussianPrimaryMassRatio,
            dict(loc=35.0, scale=5.0, beta=1.0, mmin=10.0, mmax=100.0),
            [(10.0, 100.0), (1e-6, 1.0)],
        ),
        (
            SmoothedPowerlawPrimaryMassRatio,
            dict(
                alpha=2.0,
                beta=1.0,
                delta_m1=3.0,
                delta_m2=3.0,
                mmax=100.0,
                m1min=5.0,
                m2min=5.0,
            ),
            [(5.0, 100.0), (1e-6, 1.0)],
        ),
        (
            SmoothedGaussianPrimaryMassRatio,
            dict(
                loc=35.0,
                scale=5.0,
                beta=1.0,
                m1min=5.0,
                m2min=5.0,
                mmax=100.0,
                delta_m1=3.0,
                delta_m2=3.0,
            ),
            [(5.0, 100.0), (1e-6, 1.0)],
        ),
        (
            SmoothedTwoComponentPrimaryMassRatio,
            dict(
                alpha=2.0,
                beta=1.0,
                delta=3.0,
                lambda_peak=0.3,
                loc=35.0,
                mmax=100.0,
                mmin=5.0,
                scale=5.0,
            ),
            [(5.0, 100.0), (1e-6, 1.0)],
        ),
    ],
)
def test_normalisation(model_cls, kwargs, bounds, trapz_nd):
    assert_allclose(trapz_nd(model_cls(**kwargs), bounds, num=1201), 1.0, atol=2e-3)


def test_wysocki_normalisation(trapz_nd):
    model = Wysocki2019MassModel(alpha_m=-1.0, mmin=10.0, mmax=50.0)
    assert_allclose(
        trapz_nd(model, [(10.0, 50.0), (10.0, 50.0)], num=1201), 1.0, atol=5e-3
    )


###############################################################################
# closed forms
###############################################################################


@pytest.mark.parametrize("alpha", [-3.0, -1.0, 0.0, 1.7])
@pytest.mark.parametrize("beta", [-2.0, 0.0, 1.0])
def test_powerlaw_primary_mass_ratio_matches_closed_form(alpha, beta):
    mmin, mmax = 10.0, 50.0
    model = PowerlawPrimaryMassRatio(alpha=alpha, beta=beta, mmin=mmin, mmax=mmax)
    m1 = jnp.asarray([12.0, 25.0, 49.0])
    q = jnp.asarray([0.9, 0.6, 0.3])
    expected = doubly_truncated_power_law_log_prob(
        m1, -alpha, mmin, mmax
    ) + doubly_truncated_power_law_log_prob(q, beta, mmin / m1, 1.0)
    assert_allclose(model.log_prob(jnp.stack([m1, q], axis=-1)), expected, rtol=1e-12)


def test_powerlaw_primary_mass_ratio_is_minus_inf_at_the_lower_edge():
    model = PowerlawPrimaryMassRatio(alpha=1.0, beta=1.0, mmin=10.0, mmax=50.0)
    # at m1 == mmin the conditional mass-ratio law degenerates to a point mass
    assert jnp.isneginf(model.log_prob(jnp.asarray([10.0, 1.0])))


@pytest.mark.parametrize("alpha_m", [-2.0, -1.0, 0.0, 1.5])
def test_wysocki_matches_closed_form(alpha_m):
    mmin, mmax = 10.0, 50.0
    model = Wysocki2019MassModel(alpha_m=alpha_m, mmin=mmin, mmax=mmax)
    m1 = jnp.asarray([12.0, 25.0, 49.0])
    m2 = jnp.asarray([11.0, 20.0, 30.0])
    expected = doubly_truncated_power_law_log_prob(m1, -alpha_m, mmin, mmax) - jnp.log(
        m1 - mmin
    )
    assert_allclose(model.log_prob(jnp.stack([m1, m2], axis=-1)), expected, rtol=1e-12)


def test_wysocki_secondary_mass_is_uniform():
    model = Wysocki2019MassModel(alpha_m=-1.0, mmin=10.0, mmax=50.0)
    m2 = jnp.asarray([10.5, 15.0, 20.0, 29.9])
    value = jnp.stack([jnp.full_like(m2, 30.0), m2], axis=-1)
    log_prob = model.log_prob(value)
    assert_allclose(log_prob, jnp.full_like(log_prob, log_prob[0]), rtol=1e-12)


def test_generic_smoothed_powerlaw_matches_its_primary_distribution():
    primary = DoublyTruncatedPowerLaw(alpha=-2.0, low=5.0, high=100.0)
    model = GenericSmoothedPowerlawMassRatio(
        primary_mass_distribution=primary,
        beta=1.0,
        delta_m1=3.0,
        delta_m2=3.0,
        m2min=5.0,
    )
    m1 = jnp.asarray([20.0, 50.0, 90.0])
    expected = primary.log_prob(m1) + log_planck_taper_window((m1 - 5.0) / 3.0)
    assert_allclose(model._log_prob_m1_unnorm(m1), expected, rtol=1e-12)


def test_generic_smoothed_powerlaw_rejects_unbounded_primary():
    with pytest.raises(ValueError, match="interval support constraint"):
        GenericSmoothedPowerlawMassRatio(
            primary_mass_distribution=Normal(0.0, 1.0),
            beta=1.0,
            delta_m1=3.0,
            delta_m2=3.0,
            m2min=5.0,
        )


def test_two_component_reduces_to_the_powerlaw_when_the_peak_is_switched_off():
    shared = dict(alpha=2.0, beta=1.0, mmax=100.0)
    two_component = SmoothedTwoComponentPrimaryMassRatio(
        delta=3.0, lambda_peak=0.0, loc=35.0, mmin=5.0, scale=5.0, **shared
    )
    powerlaw = SmoothedPowerlawPrimaryMassRatio(
        delta_m1=3.0, delta_m2=3.0, m1min=5.0, m2min=5.0, **shared
    )
    value = jnp.asarray([[20.0, 0.4], [30.0, 0.5], [70.0, 0.9]])
    assert_allclose(two_component.log_prob(value), powerlaw.log_prob(value), rtol=1e-5)


def test_two_component_peak_only_is_a_smoothed_gaussian():
    model = SmoothedTwoComponentPrimaryMassRatio(
        alpha=2.0,
        beta=1.0,
        delta=3.0,
        lambda_peak=1.0,
        loc=35.0,
        mmax=100.0,
        mmin=5.0,
        scale=5.0,
    )
    m1 = jnp.asarray([20.0, 35.0, 60.0])
    expected = Normal(35.0, 5.0).log_prob(m1) + log_planck_taper_window(
        (m1 - 5.0) / 3.0
    )
    assert_allclose(model._log_prob_m1_unnorm(m1), expected, rtol=1e-10)


###############################################################################
# smoothing window
###############################################################################


@pytest.mark.parametrize(
    "model_cls, kwargs, zero_delta",
    [
        (SmoothedPowerlawPrimaryMassRatio, SMOOTHED_POWERLAW, "delta_m1"),
        (SmoothedPowerlawPrimaryMassRatio, SMOOTHED_POWERLAW, "delta_m2"),
        (SmoothedTwoComponentPrimaryMassRatio, TWO_COMPONENT, "delta"),
    ],
)
def test_non_positive_smoothing_scale_kills_the_density(model_cls, kwargs, zero_delta):
    model = model_cls(**{**kwargs, zero_delta: 0.0})
    assert jnp.isneginf(model.log_prob(jnp.asarray([30.0, 0.5])))


def test_smoothing_window_vanishes_below_the_lower_edge():
    model = SmoothedPowerlawPrimaryMassRatio(
        alpha=2.0,
        beta=1.0,
        delta_m1=3.0,
        delta_m2=3.0,
        mmax=100.0,
        m1min=20.0,
        m2min=5.0,
    )
    assert jnp.isneginf(model.log_prob(jnp.asarray([15.0, 0.5])))
    assert jnp.isneginf(model.log_prob(jnp.asarray([20.0, 0.9])))
    assert jnp.isfinite(model.log_prob(jnp.asarray([25.0, 0.9])))


def test_smoothing_window_is_monotonic_across_the_taper():
    model = SmoothedPowerlawPrimaryMassRatio(
        alpha=0.0,
        beta=0.0,
        delta_m1=10.0,
        delta_m2=1e-3,
        mmax=100.0,
        m1min=20.0,
        m2min=1.0,
    )
    m1 = jnp.linspace(20.5, 29.5, 25)
    log_prob_m1 = model._log_prob_m1_unnorm(m1)
    assert jnp.all(jnp.diff(log_prob_m1) > 0.0)


###############################################################################
# sampling
###############################################################################


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [(cls, kw) for cls, kw in GENERAL_MODELS if cls is not GaussianPrimaryMassRatio],
)
@pytest.mark.parametrize("sample_shape", [(), (5,), (2, 3)])
def test_sample_shape(model_cls, kwargs, sample_shape):
    model = model_cls(**kwargs)
    samples = model.sample(jrd.key(0), sample_shape)
    assert samples.shape == model.shape(sample_shape)


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [(cls, kw) for cls, kw in GENERAL_MODELS if cls is not GaussianPrimaryMassRatio],
)
def test_samples_lie_in_the_support(model_cls, kwargs):
    model = model_cls(**kwargs)
    samples = model.sample(jrd.key(7), (4096,))
    assert jnp.all(model.support.check(samples))


def test_powerlaw_primary_mass_samples_follow_the_marginal_cdf():
    alpha, mmin, mmax = 1.5, 10.0, 50.0
    model = PowerlawPrimaryMassRatio(alpha=alpha, beta=1.0, mmin=mmin, mmax=mmax)
    m1 = np.asarray(model.sample(jrd.key(3), (100_000,))[..., 0])
    grid = np.linspace(mmin, mmax, 25)
    empirical = np.mean(m1[:, None] <= grid[None, :], axis=0)
    exact = DoublyTruncatedPowerLaw(alpha=-alpha, low=mmin, high=mmax).cdf(grid)
    assert_allclose(empirical, exact, atol=8e-3)


def test_wysocki_secondary_mass_samples_are_uniform():
    model = Wysocki2019MassModel(alpha_m=-1.0, mmin=10.0, mmax=50.0)
    samples = model.sample(jrd.key(11), (100_000,))
    ratio = np.asarray((samples[..., 1] - 10.0) / (samples[..., 0] - 10.0))
    grid = np.linspace(0.0, 1.0, 21)
    empirical = np.mean(ratio[:, None] <= grid[None, :], axis=0)
    assert_allclose(empirical, grid, atol=8e-3)


###############################################################################
# batching
###############################################################################


@pytest.mark.parametrize(
    "model_cls, kwargs, batched",
    [
        (
            PowerlawPrimaryMassRatio,
            dict(alpha=-1.0, beta=1.0, mmin=10.0, mmax=50.0),
            dict(alpha=jnp.asarray([-1.0, 2.0])),
        ),
        (
            Wysocki2019MassModel,
            dict(alpha_m=-1.0, mmin=10.0, mmax=50.0),
            dict(alpha_m=jnp.asarray([-1.0, 2.0])),
        ),
        (
            GaussianPrimaryMassRatio,
            dict(loc=35.0, scale=5.0, beta=1.0, mmin=10.0, mmax=100.0),
            dict(loc=jnp.asarray([30.0, 40.0])),
        ),
    ],
)
def test_general_models_broadcast_their_parameters(model_cls, kwargs, batched):
    model = model_cls(**{**kwargs, **batched})
    assert model.batch_shape == (2,)
    assert model.event_shape == (2,)
    value = jnp.asarray([30.0, 0.5])
    log_prob = model.log_prob(value)
    assert log_prob.shape == (2,)
    for i in range(2):
        scalar_kwargs = {**kwargs, **{k: v[i] for k, v in batched.items()}}
        assert_allclose(
            log_prob[i], model_cls(**scalar_kwargs).log_prob(value), rtol=1e-12
        )


@pytest.mark.parametrize(
    "model_cls, kwargs, batched_key",
    [
        (SmoothedPowerlawPrimaryMassRatio, SMOOTHED_POWERLAW, "alpha"),
        (SmoothedGaussianPrimaryMassRatio, SMOOTHED_GAUSSIAN, "loc"),
        (SmoothedTwoComponentPrimaryMassRatio, TWO_COMPONENT, "alpha"),
    ],
)
def test_specific_models_require_a_scalar_batch_shape(model_cls, kwargs, batched_key):
    # these models tabulate their normalisation on a one-dimensional grid built from
    # their own parameters, which only exists when the batch shape is empty
    with pytest.raises(ValueError):
        model_cls(**{**kwargs, batched_key: jnp.asarray([1.0, 2.0])})


###############################################################################
# gradients
###############################################################################


@pytest.mark.parametrize(
    "model_cls, kwargs, argname",
    [
        (PowerlawPrimaryMassRatio, POWERLAW, "alpha"),
        (PowerlawPrimaryMassRatio, POWERLAW, "beta"),
        (Wysocki2019MassModel, WYSOCKI, "alpha_m"),
        (GaussianPrimaryMassRatio, GAUSSIAN, "loc"),
        (SmoothedPowerlawPrimaryMassRatio, SMOOTHED_POWERLAW, "alpha"),
        (SmoothedGaussianPrimaryMassRatio, SMOOTHED_GAUSSIAN, "loc"),
        (SmoothedTwoComponentPrimaryMassRatio, TWO_COMPONENT, "lambda_peak"),
    ],
)
def test_log_prob_gradient_matches_finite_differences(model_cls, kwargs, argname):
    value = _inside(model_cls, kwargs, 30.0, 0.6)

    def log_prob(x):
        return model_cls(**{**kwargs, argname: x}).log_prob(value)

    x0 = jnp.asarray(float(kwargs[argname]))
    step = 1e-5
    finite_difference = (log_prob(x0 + step) - log_prob(x0 - step)) / (2.0 * step)
    assert_allclose(jax.grad(log_prob)(x0), finite_difference, rtol=1e-4, atol=1e-6)
