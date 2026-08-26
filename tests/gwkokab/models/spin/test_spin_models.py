# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.spin._models`."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose
from numpyro.distributions import (
    Beta,
    Independent,
    MixtureGeneral,
    MultivariateNormal,
    Normal,
    TruncatedNormal,
)

from gwkokab.models.spin import (
    BetaFromMeanVar,
    GaussianSpinModel,
    GenericTiltModel,
    GWTC4EffectiveSpinSkewNormalModel,
)


###############################################################################
# GaussianSpinModel
###############################################################################


GAUSSIAN_SPIN_PARAMS = [
    dict(mu_eff=0.0, sigma_eff=0.2, mu_p=0.3, sigma_p=0.1, rho=0.0),
    dict(mu_eff=0.1, sigma_eff=0.2, mu_p=0.3, sigma_p=0.1, rho=0.5),
    dict(mu_eff=-0.3, sigma_eff=0.4, mu_p=0.2, sigma_p=0.3, rho=-0.8),
]


@pytest.mark.parametrize("params", GAUSSIAN_SPIN_PARAMS)
def test_gaussian_spin_model_is_a_bivariate_normal(params):
    model = GaussianSpinModel(**params)
    assert isinstance(model, MultivariateNormal)
    assert model.batch_shape == ()
    assert model.event_shape == (2,)


@pytest.mark.parametrize("params", GAUSSIAN_SPIN_PARAMS)
def test_gaussian_spin_model_moments(params):
    model = GaussianSpinModel(**params)
    sigma_eff, sigma_p, rho = params["sigma_eff"], params["sigma_p"], params["rho"]
    assert_allclose(model.mean, [params["mu_eff"], params["mu_p"]], rtol=1e-12)
    assert_allclose(
        model.covariance_matrix,
        [
            [sigma_eff**2, rho * sigma_eff * sigma_p],
            [rho * sigma_eff * sigma_p, sigma_p**2],
        ],
        rtol=1e-12,
    )


def test_gaussian_spin_model_correlation_is_recovered_from_samples():
    model = GaussianSpinModel(mu_eff=0.1, sigma_eff=0.2, mu_p=0.3, sigma_p=0.1, rho=0.5)
    samples = np.asarray(model.sample(jrd.key(0), (200_000,)))
    assert_allclose(samples.mean(axis=0), [0.1, 0.3], atol=5e-3)
    assert_allclose(np.corrcoef(samples.T)[0, 1], 0.5, atol=1e-2)


def test_gaussian_spin_model_with_zero_correlation_factorises():
    model = GaussianSpinModel(mu_eff=0.1, sigma_eff=0.2, mu_p=0.3, sigma_p=0.1, rho=0.0)
    value = jnp.asarray([0.25, 0.2])
    factorised = Normal(0.1, 0.2).log_prob(value[0]) + Normal(0.3, 0.1).log_prob(
        value[1]
    )
    assert_allclose(model.log_prob(value), factorised, rtol=1e-10)


def test_gaussian_spin_model_does_not_broadcast_its_parameters():
    # the factory stacks scalars into a 2-vector mean and a 2x2 covariance, which only
    # works for scalar inputs
    with pytest.raises(TypeError):
        GaussianSpinModel(
            mu_eff=jnp.asarray([0.1, 0.2]),
            sigma_eff=0.2,
            mu_p=0.3,
            sigma_p=0.1,
            rho=0.5,
        )


###############################################################################
# BetaFromMeanVar
###############################################################################


BETA_PARAMS = [(0.4, 0.02), (0.5, 0.05), (0.1, 0.005), (0.8, 0.01)]


@pytest.mark.parametrize("mean, variance", BETA_PARAMS)
def test_beta_from_mean_var_recovers_its_moments(mean, variance):
    model = BetaFromMeanVar(mean=mean, variance=variance)
    assert isinstance(model, Beta)
    assert_allclose(model.mean, mean, rtol=1e-12)
    assert_allclose(model.variance, variance, rtol=1e-12)


@pytest.mark.parametrize("mean, variance", BETA_PARAMS)
def test_beta_from_mean_var_concentrations(mean, variance):
    model = BetaFromMeanVar(mean=mean, variance=variance)
    common = mean * (1.0 - mean) / variance - 1.0
    assert_allclose(model.concentration1, mean * common, rtol=1e-10)
    assert_allclose(model.concentration0, (1.0 - mean) * common, rtol=1e-10)


def test_beta_from_mean_var_broadcasts():
    model = BetaFromMeanVar(
        mean=jnp.asarray([0.3, 0.4]), variance=jnp.asarray([0.02, 0.03])
    )
    assert model.batch_shape == (2,)
    assert_allclose(model.mean, [0.3, 0.4], rtol=1e-12)


def test_beta_from_mean_var_rejects_an_impossible_variance():
    # a Beta needs mean * (1 - mean) > variance; otherwise both concentrations are
    # non-positive
    with pytest.raises(ValueError):
        BetaFromMeanVar(mean=0.5, variance=0.5, validate_args=True)


@pytest.mark.parametrize("mean, variance", BETA_PARAMS)
def test_beta_from_mean_var_is_normalised(mean, variance, trapz_1d):
    model = BetaFromMeanVar(mean=mean, variance=variance)
    assert_allclose(trapz_1d(model, 1e-6, 1.0 - 1e-6), 1.0, atol=1e-3)


###############################################################################
# GenericTiltModel
###############################################################################


TILT_PARAMS = dict(zeta=0.4, loc1=1.0, loc2=1.0, scale1=0.5, scale2=0.7)


def test_generic_tilt_model_shapes():
    model = GenericTiltModel(**TILT_PARAMS)
    assert isinstance(model, MixtureGeneral)
    assert model.batch_shape == ()
    assert model.event_shape == (2,)
    assert model.mixture_size == 2


def test_generic_tilt_model_is_normalised(trapz_nd):
    model = GenericTiltModel(**TILT_PARAMS)
    assert_allclose(
        trapz_nd(model, [(-1.0, 1.0), (-1.0, 1.0)], num=801), 1.0, atol=1e-4
    )


def test_generic_tilt_model_without_alignment_is_isotropic():
    model = GenericTiltModel(**{**TILT_PARAMS, "zeta": 0.0})
    value = jnp.asarray([[0.3, -0.2], [-0.9, 0.9], [0.0, 0.0]])
    # the isotropic component is uniform on [-1, 1]^2, i.e. a density of 1/4
    assert_allclose(model.log_prob(value), jnp.full(3, jnp.log(0.25)), rtol=1e-12)


def test_generic_tilt_model_with_full_alignment_is_a_truncated_normal():
    model = GenericTiltModel(**{**TILT_PARAMS, "zeta": 1.0})
    gaussian = Independent(
        TruncatedNormal(
            loc=jnp.asarray([TILT_PARAMS["loc1"], TILT_PARAMS["loc2"]]),
            scale=jnp.asarray([TILT_PARAMS["scale1"], TILT_PARAMS["scale2"]]),
            low=jnp.asarray([-1.0, -1.0]),
            high=jnp.asarray([1.0, 1.0]),
        ),
        1,
    )
    value = jnp.asarray([[0.3, -0.2], [-0.9, 0.9]])
    assert_allclose(model.log_prob(value), gaussian.log_prob(value), rtol=1e-12)


def test_generic_tilt_model_mixes_its_two_components():
    value = jnp.asarray([[0.3, -0.2], [-0.9, 0.9]])
    isotropic = GenericTiltModel(**{**TILT_PARAMS, "zeta": 0.0}).log_prob(value)
    gaussian = GenericTiltModel(**{**TILT_PARAMS, "zeta": 1.0}).log_prob(value)
    zeta = TILT_PARAMS["zeta"]
    expected = jnp.log((1.0 - zeta) * jnp.exp(isotropic) + zeta * jnp.exp(gaussian))
    assert_allclose(
        GenericTiltModel(**TILT_PARAMS).log_prob(value), expected, rtol=1e-12
    )


def test_generic_tilt_model_honours_custom_bounds():
    model = GenericTiltModel(
        **{**TILT_PARAMS, "zeta": 0.0},
        low1=0.0,
        high1=1.0,
        low2=-0.5,
        high2=0.5,
    )
    # uniform on [0, 1] x [-0.5, 0.5] has unit density
    assert_allclose(model.log_prob(jnp.asarray([0.5, 0.0])), 0.0, atol=1e-12)


def test_generic_tilt_model_broadcasts():
    model = GenericTiltModel(**{**TILT_PARAMS, "zeta": jnp.asarray([0.2, 0.8])})
    assert model.batch_shape == (2,)
    assert model.event_shape == (2,)
    value = jnp.asarray([0.3, -0.2])
    log_prob = model.log_prob(value)
    assert log_prob.shape == (2,)
    for i, zeta in enumerate([0.2, 0.8]):
        expected = GenericTiltModel(**{**TILT_PARAMS, "zeta": zeta}).log_prob(value)
        assert_allclose(log_prob[i], expected, rtol=1e-12)


@pytest.mark.parametrize("sample_shape", [(), (5,), (2, 3)])
def test_generic_tilt_model_sample_shape(sample_shape):
    model = GenericTiltModel(**TILT_PARAMS)
    assert model.sample(jrd.key(0), sample_shape).shape == model.shape(sample_shape)


def test_generic_tilt_model_samples_lie_in_the_support():
    model = GenericTiltModel(**TILT_PARAMS)
    samples = model.sample(jrd.key(4), (8192,))
    assert jnp.all((samples >= -1.0) & (samples <= 1.0))


###############################################################################
# GWTC4EffectiveSpinSkewNormalModel
###############################################################################


SKEW_PARAMS = [
    dict(loc=0.0, scale=0.3, epsilon=0.0),
    dict(loc=0.1, scale=0.3, epsilon=0.4),
    dict(loc=-0.2, scale=0.5, epsilon=-0.6),
    dict(loc=0.0, scale=0.2, epsilon=0.9),
]


@pytest.mark.parametrize("params", SKEW_PARAMS)
def test_skew_normal_shapes_and_support(params):
    model = GWTC4EffectiveSpinSkewNormalModel(**params)
    assert model.batch_shape == ()
    assert model.event_shape == ()
    assert model.support.lower_bound == -1.0
    assert model.support.upper_bound == 1.0


@pytest.mark.parametrize("params", SKEW_PARAMS)
def test_skew_normal_is_normalised(params, trapz_1d):
    model = GWTC4EffectiveSpinSkewNormalModel(**params)
    assert_allclose(trapz_1d(model, -1.0 + 1e-9, 1.0 - 1e-9, 40_001), 1.0, atol=1e-4)


def test_skew_normal_without_skew_is_a_truncated_normal():
    model = GWTC4EffectiveSpinSkewNormalModel(loc=0.1, scale=0.3, epsilon=0.0)
    truncated = TruncatedNormal(loc=0.1, scale=0.3, low=-1.0, high=1.0)
    value = jnp.asarray([-0.5, 0.0, 0.1, 0.6])
    assert_allclose(model.log_prob(value), truncated.log_prob(value), rtol=1e-10)


@pytest.mark.parametrize("epsilon", [-0.6, -0.2, 0.2, 0.6])
def test_skew_normal_widths_switch_at_the_location(epsilon):
    loc, scale = 0.1, 0.3
    model = GWTC4EffectiveSpinSkewNormalModel(loc=loc, scale=scale, epsilon=epsilon)
    below, above = jnp.asarray(loc - 0.2), jnp.asarray(loc + 0.2)
    # the two halves are truncated normals of width scale * (1 +/- epsilon), rescaled by
    # the same factor and divided by a common normalisation
    left = TruncatedNormal(
        loc=loc, scale=scale * (1.0 + epsilon), low=-1.0, high=1.0
    ).log_prob(below) + jnp.log1p(epsilon)
    right = TruncatedNormal(
        loc=loc, scale=scale * (1.0 - epsilon), low=-1.0, high=1.0
    ).log_prob(above) + jnp.log1p(-epsilon)
    assert_allclose(
        model.log_prob(below) - model.log_prob(above), left - right, rtol=1e-10
    )


@pytest.mark.parametrize("epsilon", [-0.6, 0.6])
def test_skew_normal_is_skewed_in_the_expected_direction(epsilon):
    model = GWTC4EffectiveSpinSkewNormalModel(loc=0.0, scale=0.3, epsilon=epsilon)
    x = jnp.linspace(1e-3, 0.9, 200)
    left = jnp.trapezoid(jnp.exp(model.log_prob(-x)), x)
    right = jnp.trapezoid(jnp.exp(model.log_prob(x)), x)
    if epsilon > 0.0:
        assert left > right
    else:
        assert left < right


@pytest.mark.parametrize("params", SKEW_PARAMS)
@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_skew_normal_log_prob_shape(params, sample_shape):
    model = GWTC4EffectiveSpinSkewNormalModel(**params)
    assert model.log_prob(jnp.full(sample_shape, 0.2)).shape == sample_shape


def test_skew_normal_broadcasts():
    model = GWTC4EffectiveSpinSkewNormalModel(
        loc=jnp.asarray([0.0, 0.2]), scale=0.3, epsilon=0.4
    )
    assert model.batch_shape == (2,)
    value = jnp.asarray(0.1)
    log_prob = model.log_prob(value)
    for i, loc in enumerate([0.0, 0.2]):
        expected = GWTC4EffectiveSpinSkewNormalModel(
            loc=loc, scale=0.3, epsilon=0.4
        ).log_prob(value)
        assert_allclose(log_prob[i], expected, rtol=1e-12)


def test_skew_normal_pytree_roundtrip(pytree_roundtrip):
    model = GWTC4EffectiveSpinSkewNormalModel(loc=0.1, scale=0.3, epsilon=0.4)
    value = jnp.asarray(0.2)
    assert_allclose(pytree_roundtrip(model).log_prob(value), model.log_prob(value))


def test_skew_normal_warns_outside_its_support():
    model = GWTC4EffectiveSpinSkewNormalModel(
        loc=0.1, scale=0.3, epsilon=0.4, validate_args=True
    )
    with pytest.warns(UserWarning, match="Out-of-support values"):
        model.log_prob(jnp.asarray(2.0))


def test_skew_normal_rejects_an_out_of_range_skew():
    with pytest.raises(ValueError):
        GWTC4EffectiveSpinSkewNormalModel(
            loc=0.0, scale=0.3, epsilon=1.5, validate_args=True
        )


def test_skew_normal_does_not_implement_sampling():
    model = GWTC4EffectiveSpinSkewNormalModel(loc=0.0, scale=0.3, epsilon=0.4)
    with pytest.raises(NotImplementedError):
        model.sample(jrd.key(0))


@pytest.mark.parametrize("argname", ["loc", "scale", "epsilon"])
def test_skew_normal_gradients_match_finite_differences(argname):
    params = dict(loc=0.1, scale=0.3, epsilon=0.4)
    value = jnp.asarray(-0.2)

    def log_prob(x):
        return GWTC4EffectiveSpinSkewNormalModel(**{**params, argname: x}).log_prob(
            value
        )

    x0 = jnp.asarray(params[argname])
    step = 1e-6
    finite_difference = (log_prob(x0 + step) - log_prob(x0 - step)) / (2.0 * step)
    assert_allclose(jax.grad(log_prob)(x0), finite_difference, rtol=1e-5, atol=1e-7)
