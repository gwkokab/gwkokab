# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.redshift._models`."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose

from gwkokab.cosmology import default_cosmology
from gwkokab.models.redshift import (
    MadauDickinsonRedshiftModel,
    PowerlawRedshiftModel,
)
from gwkokab.models.redshift._models import _RedshiftModel


POWERLAW_PARAMS = [
    dict(z_max=1.0, kappa=0.0),
    dict(z_max=1.0, kappa=2.7),
    dict(z_max=2.3, kappa=1.0),
    dict(z_max=0.5, kappa=-1.5),
]

MADAU_DICKINSON_PARAMS = [
    dict(z_max=1.0, kappa=0.0, gamma=2.0, z_peak=0.1),
    dict(z_max=2.3, kappa=1.0, gamma=2.9, z_peak=0.5),
    dict(z_max=1.5, kappa=2.7, gamma=5.6, z_peak=1.9),
]

ALL_PARAMS = [(PowerlawRedshiftModel, p) for p in POWERLAW_PARAMS] + [
    (MadauDickinsonRedshiftModel, p) for p in MADAU_DICKINSON_PARAMS
]


###############################################################################
# shapes and support
###############################################################################


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_shapes_and_support(model_cls, params):
    model = model_cls(**params)
    assert model.batch_shape == ()
    assert model.event_shape == ()
    assert model.support.lower_bound == 0.0
    assert model.support.upper_bound == params["z_max"]


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_log_prob_shape(model_cls, params, sample_shape):
    model = model_cls(**params)
    value = jnp.full(sample_shape, 0.5 * params["z_max"])
    assert model.log_prob(value).shape == sample_shape


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_pytree_roundtrip(model_cls, params, pytree_roundtrip):
    model = model_cls(**params)
    value = jnp.asarray(0.5 * params["z_max"])
    assert_allclose(pytree_roundtrip(model).log_prob(value), model.log_prob(value))


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_under_jit(model_cls, params):
    value = jnp.asarray(0.5 * params["z_max"])

    def log_prob(value):
        return model_cls(**params).log_prob(value)

    assert_allclose(jax.jit(log_prob)(value), log_prob(value), rtol=1e-9)


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_validate_args_warns_outside_support(model_cls, params):
    model = model_cls(**params, validate_args=True)
    with pytest.warns(UserWarning, match="Out-of-support values"):
        model.log_prob(jnp.asarray(2.0 * params["z_max"]))


###############################################################################
# the density itself
###############################################################################


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_log_prob_is_the_differential_spacetime_volume(model_cls, params):
    model = model_cls(**params)
    z = jnp.linspace(1e-3, params["z_max"], 51)
    expected = -jnp.log1p(z) + default_cosmology().logdVcdz(z) + model.log_psi_of_z(z)
    assert_allclose(model.log_prob(z), expected, rtol=1e-12)


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_log_norm_is_the_integral_of_the_unnormalised_density(model_cls, params):
    model = model_cls(**params)
    z = jnp.linspace(0.0, params["z_max"], 40_001)
    integral = jnp.trapezoid(jnp.exp(model.log_prob(z)), z)
    assert_allclose(jnp.exp(model.log_norm()), integral, rtol=1e-4)


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_dividing_by_the_norm_gives_a_density(model_cls, params):
    model = model_cls(**params)
    z = jnp.linspace(0.0, params["z_max"], 40_001)
    normalised = jnp.exp(model.log_prob(z) - model.log_norm())
    assert_allclose(jnp.trapezoid(normalised, z), 1.0, rtol=1e-4)


@pytest.mark.parametrize("kappa", [-1.5, 0.0, 1.0, 2.7])
def test_powerlaw_psi(kappa):
    model = PowerlawRedshiftModel(z_max=2.0, kappa=kappa)
    z = jnp.linspace(0.0, 2.0, 11)
    assert_allclose(model.log_psi_of_z(z), kappa * jnp.log1p(z), rtol=1e-12)


@pytest.mark.parametrize("params", MADAU_DICKINSON_PARAMS)
def test_madau_dickinson_psi(params):
    model = MadauDickinsonRedshiftModel(**params)
    z = jnp.linspace(0.0, params["z_max"], 11)
    kappa, gamma, z_peak = params["kappa"], params["gamma"], params["z_peak"]
    expected = (
        kappa * jnp.log1p(z)
        + jnp.log1p((1.0 + z_peak) ** gamma)
        - jnp.log((1.0 + z_peak) ** gamma + (1.0 + z) ** gamma)
    )
    assert_allclose(model.log_psi_of_z(z), expected, rtol=1e-12)


def test_madau_dickinson_psi_is_normalised_at_zero_redshift():
    # psi(0) == 1 by construction, so log psi(0) == 0
    model = MadauDickinsonRedshiftModel(z_max=2.0, kappa=1.0, gamma=2.9, z_peak=0.5)
    assert_allclose(model.log_psi_of_z(jnp.asarray(0.0)), 0.0, atol=1e-12)


def test_madau_dickinson_with_no_high_redshift_suppression_is_a_powerlaw():
    # gamma == 0 removes the (1+z)^gamma turnover entirely
    madau = MadauDickinsonRedshiftModel(z_max=2.0, kappa=1.7, gamma=0.0, z_peak=0.5)
    powerlaw = PowerlawRedshiftModel(z_max=2.0, kappa=1.7)
    z = jnp.linspace(0.0, 2.0, 11)
    assert_allclose(madau.log_prob(z), powerlaw.log_prob(z), rtol=1e-12)


def test_madau_dickinson_turns_over_after_the_peak():
    model = MadauDickinsonRedshiftModel(z_max=4.0, kappa=2.7, gamma=5.6, z_peak=1.9)
    z = jnp.linspace(1e-3, 4.0, 400)
    log_psi = model.log_psi_of_z(z)
    peak = int(jnp.argmax(log_psi))
    assert jnp.all(jnp.diff(log_psi[:peak]) > 0.0)
    assert jnp.all(jnp.diff(log_psi[peak + 1 :]) < 0.0)


###############################################################################
# sampling
###############################################################################


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
@pytest.mark.parametrize("sample_shape", [(), (5,), (2, 3)])
def test_sample_shape(model_cls, params, sample_shape):
    model = model_cls(**params)
    assert model.sample(jrd.key(0), sample_shape).shape == model.shape(sample_shape)


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_samples_lie_in_the_support(model_cls, params):
    model = model_cls(**params)
    assert jnp.all(model.support.check(model.sample(jrd.key(1), (8192,))))


@pytest.mark.parametrize("model_cls, params", ALL_PARAMS)
def test_samples_follow_the_normalised_density(model_cls, params):
    model = model_cls(**params)
    samples = np.asarray(model.sample(jrd.key(2), (60_000,)))

    grid = np.linspace(0.0, params["z_max"], 4001)
    density = np.exp(np.asarray(model.log_prob(jnp.asarray(grid)) - model.log_norm()))
    exact_cdf = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid)),
    ])

    probe = np.linspace(0.0, params["z_max"], 21)
    empirical = np.mean(samples[:, None] <= probe[None, :], axis=0)
    assert_allclose(empirical, np.interp(probe, grid, exact_cdf), atol=1e-2)


###############################################################################
# batching
###############################################################################


@pytest.mark.parametrize(
    "model_cls, params, batched",
    [
        (
            PowerlawRedshiftModel,
            POWERLAW_PARAMS[1],
            dict(kappa=jnp.asarray([0.0, 2.0])),
        ),
        (
            MadauDickinsonRedshiftModel,
            MADAU_DICKINSON_PARAMS[1],
            dict(gamma=jnp.asarray([2.0, 4.0])),
        ),
    ],
)
def test_parameters_broadcast(model_cls, params, batched):
    model = model_cls(**{**params, **batched})
    assert model.batch_shape == (2,)
    value = jnp.asarray(0.4)
    log_prob = model.log_prob(value)
    assert log_prob.shape == (2,)
    for i in range(2):
        scalar = {**params, **{k: v[i] for k, v in batched.items()}}
        assert_allclose(log_prob[i], model_cls(**scalar).log_prob(value), rtol=1e-12)


@pytest.mark.parametrize(
    "model_cls, params, batched",
    [
        (
            PowerlawRedshiftModel,
            POWERLAW_PARAMS[1],
            dict(kappa=jnp.asarray([0.0, 2.0])),
        ),
        (
            MadauDickinsonRedshiftModel,
            MADAU_DICKINSON_PARAMS[1],
            dict(gamma=jnp.asarray([2.0, 4.0])),
        ),
    ],
)
def test_sampling_needs_a_scalar_batch_shape(model_cls, params, batched):
    # inverse-transform sampling interpolates on a single one-dimensional grid
    model = model_cls(**{**params, **batched})
    with pytest.raises(ValueError):
        model.sample(jrd.key(0))


###############################################################################
# the abstract base
###############################################################################


def test_base_model_requires_a_psi_implementation():
    model = _RedshiftModel(z_max=jnp.asarray(1.0))
    with pytest.raises(NotImplementedError, match="log_psi_of_z"):
        model.log_psi_of_z(jnp.asarray(0.5))


###############################################################################
# gradients
###############################################################################


@pytest.mark.parametrize(
    "model_cls, params, argname",
    [
        (PowerlawRedshiftModel, POWERLAW_PARAMS[1], "kappa"),
        (MadauDickinsonRedshiftModel, MADAU_DICKINSON_PARAMS[1], "kappa"),
        (MadauDickinsonRedshiftModel, MADAU_DICKINSON_PARAMS[1], "gamma"),
        (MadauDickinsonRedshiftModel, MADAU_DICKINSON_PARAMS[1], "z_peak"),
    ],
)
def test_gradients_match_finite_differences(model_cls, params, argname):
    value = jnp.asarray(0.4)

    def log_prob(x):
        return model_cls(**{**params, argname: x}).log_prob(value)

    x0 = jnp.asarray(float(params[argname]))
    step = 1e-6
    finite_difference = (log_prob(x0 + step) - log_prob(x0 - step)) / (2.0 * step)
    assert_allclose(jax.grad(log_prob)(x0), finite_difference, rtol=1e-5, atol=1e-8)
