# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.sundry._models`."""

import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose
from numpyro.distributions import (
    Independent,
    MixtureGeneral,
    TruncatedNormal,
    Uniform,
)

from gwkokab.models.sundry import (
    NDIsotropicAndTruncatedNormalMixture,
    NDTwoTruncatedNormalMixture,
    TwoTruncatedNormalMixture,
)


TWO_TRUNCATED_NORMAL = dict(
    comp1_high=1.0,
    comp1_loc=0.2,
    comp1_low=0.0,
    comp1_scale=0.3,
    comp2_high=1.0,
    comp2_loc=0.8,
    comp2_low=0.0,
    comp2_scale=0.1,
)

ND_TWO_TRUNCATED_NORMAL = dict(
    comp1_high=jnp.asarray([1.0, 1.0]),
    comp1_loc=jnp.asarray([0.2, 0.3]),
    comp1_low=jnp.asarray([0.0, 0.0]),
    comp1_scale=jnp.asarray([0.3, 0.2]),
    comp2_high=jnp.asarray([1.0, 1.0]),
    comp2_loc=jnp.asarray([0.8, 0.7]),
    comp2_low=jnp.asarray([0.0, 0.0]),
    comp2_scale=jnp.asarray([0.1, 0.15]),
)

ISOTROPIC_AND_NORMAL = dict(
    loc=jnp.asarray([1.0, 1.0]),
    scale=jnp.asarray([0.5, 0.6]),
    isotropic_low=jnp.asarray([-1.0, -1.0]),
    isotropic_high=jnp.asarray([1.0, 1.0]),
    gaussian_low=jnp.asarray([-1.0, -1.0]),
    gaussian_high=jnp.asarray([1.0, 1.0]),
)


###############################################################################
# TwoTruncatedNormalMixture
###############################################################################


def test_two_truncated_normal_mixture_shapes():
    model = TwoTruncatedNormalMixture(zeta=0.3, **TWO_TRUNCATED_NORMAL)
    assert isinstance(model, MixtureGeneral)
    assert model.batch_shape == ()
    assert model.event_shape == ()
    assert model.mixture_size == 2


@pytest.mark.parametrize("zeta", [0.0, 0.25, 0.5, 1.0])
def test_two_truncated_normal_mixture_weights(zeta):
    model = TwoTruncatedNormalMixture(zeta=zeta, **TWO_TRUNCATED_NORMAL)
    assert_allclose(model.mixing_distribution.probs, [1.0 - zeta, zeta], rtol=1e-12)


@pytest.mark.parametrize("zeta", [0.0, 0.3, 1.0])
def test_two_truncated_normal_mixture_is_normalised(zeta, trapz_1d):
    model = TwoTruncatedNormalMixture(zeta=zeta, **TWO_TRUNCATED_NORMAL)
    assert_allclose(trapz_1d(model, 0.0, 1.0), 1.0, atol=1e-6)


@pytest.mark.parametrize("zeta, component", [(0.0, 1), (1.0, 2)])
def test_two_truncated_normal_mixture_degenerates_to_one_component(zeta, component):
    model = TwoTruncatedNormalMixture(zeta=zeta, **TWO_TRUNCATED_NORMAL)
    single = TruncatedNormal(
        loc=TWO_TRUNCATED_NORMAL[f"comp{component}_loc"],
        scale=TWO_TRUNCATED_NORMAL[f"comp{component}_scale"],
        low=TWO_TRUNCATED_NORMAL[f"comp{component}_low"],
        high=TWO_TRUNCATED_NORMAL[f"comp{component}_high"],
    )
    value = jnp.asarray([0.1, 0.5, 0.9])
    assert_allclose(model.log_prob(value), single.log_prob(value), rtol=1e-12)


def test_two_truncated_normal_mixture_is_a_convex_combination():
    zeta = 0.3
    value = jnp.asarray([0.1, 0.5, 0.9])
    first = TwoTruncatedNormalMixture(zeta=0.0, **TWO_TRUNCATED_NORMAL).log_prob(value)
    second = TwoTruncatedNormalMixture(zeta=1.0, **TWO_TRUNCATED_NORMAL).log_prob(value)
    expected = jnp.log((1.0 - zeta) * jnp.exp(first) + zeta * jnp.exp(second))
    assert_allclose(
        TwoTruncatedNormalMixture(zeta=zeta, **TWO_TRUNCATED_NORMAL).log_prob(value),
        expected,
        rtol=1e-12,
    )


def test_two_truncated_normal_mixture_support_is_the_union_of_the_components():
    model = TwoTruncatedNormalMixture(
        zeta=0.3,
        **{
            **TWO_TRUNCATED_NORMAL,
            "comp1_low": 0.0,
            "comp1_high": 0.4,
            "comp2_low": 0.6,
            "comp2_high": 1.0,
        },
    )
    value = jnp.asarray([-0.1, 0.2, 0.5, 0.8, 1.1])
    assert_allclose(model.support.check(value), [False, True, False, True, False])


def test_two_truncated_normal_mixture_without_bounds_is_unbounded():
    model = TwoTruncatedNormalMixture(
        zeta=0.3,
        comp1_high=None,
        comp1_loc=0.2,
        comp1_low=None,
        comp1_scale=0.3,
        comp2_high=None,
        comp2_loc=0.8,
        comp2_low=None,
        comp2_scale=0.1,
    )
    value = jnp.asarray([-5.0, 0.0, 5.0])
    assert jnp.all(model.support.check(value))
    assert jnp.all(jnp.isfinite(model.log_prob(value)))


def test_two_truncated_normal_mixture_broadcasts():
    model = TwoTruncatedNormalMixture(
        zeta=jnp.asarray([0.2, 0.8]), **TWO_TRUNCATED_NORMAL
    )
    assert model.batch_shape == (2,)
    value = jnp.asarray(0.5)
    log_prob = model.log_prob(value)
    assert log_prob.shape == (2,)
    for i, zeta in enumerate([0.2, 0.8]):
        expected = TwoTruncatedNormalMixture(
            zeta=zeta, **TWO_TRUNCATED_NORMAL
        ).log_prob(value)
        assert_allclose(log_prob[i], expected, rtol=1e-12)


@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_two_truncated_normal_mixture_sample_shape(sample_shape):
    model = TwoTruncatedNormalMixture(zeta=0.3, **TWO_TRUNCATED_NORMAL)
    assert model.sample(jrd.key(0), sample_shape).shape == model.shape(sample_shape)


def test_two_truncated_normal_mixture_samples_lie_in_the_support():
    model = TwoTruncatedNormalMixture(zeta=0.3, **TWO_TRUNCATED_NORMAL)
    assert jnp.all(model.support.check(model.sample(jrd.key(5), (8192,))))


###############################################################################
# NDTwoTruncatedNormalMixture
###############################################################################


def test_nd_two_truncated_normal_mixture_shapes():
    model = NDTwoTruncatedNormalMixture(zeta=0.3, **ND_TWO_TRUNCATED_NORMAL)
    assert model.batch_shape == ()
    assert model.event_shape == (2,)
    assert model.mixture_size == 2


def test_nd_two_truncated_normal_mixture_is_normalised(trapz_nd):
    model = NDTwoTruncatedNormalMixture(zeta=0.3, **ND_TWO_TRUNCATED_NORMAL)
    assert_allclose(trapz_nd(model, [(0.0, 1.0), (0.0, 1.0)], num=801), 1.0, atol=1e-4)


@pytest.mark.parametrize("zeta, component", [(0.0, 1), (1.0, 2)])
def test_nd_two_truncated_normal_mixture_degenerates_to_one_component(zeta, component):
    model = NDTwoTruncatedNormalMixture(zeta=zeta, **ND_TWO_TRUNCATED_NORMAL)
    single = Independent(
        TruncatedNormal(
            loc=ND_TWO_TRUNCATED_NORMAL[f"comp{component}_loc"],
            scale=ND_TWO_TRUNCATED_NORMAL[f"comp{component}_scale"],
            low=ND_TWO_TRUNCATED_NORMAL[f"comp{component}_low"],
            high=ND_TWO_TRUNCATED_NORMAL[f"comp{component}_high"],
        ),
        1,
    )
    value = jnp.asarray([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]])
    assert_allclose(model.log_prob(value), single.log_prob(value), rtol=1e-12)


def test_nd_two_truncated_normal_mixture_factorises_across_dimensions():
    model = NDTwoTruncatedNormalMixture(zeta=0.0, **ND_TWO_TRUNCATED_NORMAL)
    value = jnp.asarray([0.3, 0.4])
    per_dimension = TruncatedNormal(
        loc=ND_TWO_TRUNCATED_NORMAL["comp1_loc"],
        scale=ND_TWO_TRUNCATED_NORMAL["comp1_scale"],
        low=ND_TWO_TRUNCATED_NORMAL["comp1_low"],
        high=ND_TWO_TRUNCATED_NORMAL["comp1_high"],
    ).log_prob(value)
    assert_allclose(model.log_prob(value), jnp.sum(per_dimension), rtol=1e-12)


def test_nd_two_truncated_normal_mixture_respects_batch_dim():
    shape = (2, 3)
    model = NDTwoTruncatedNormalMixture(
        comp1_high=jnp.ones(shape),
        comp1_loc=jnp.zeros(shape),
        comp1_low=-jnp.ones(shape),
        comp1_scale=jnp.ones(shape),
        comp2_high=jnp.ones(shape),
        comp2_loc=jnp.zeros(shape),
        comp2_low=-jnp.ones(shape),
        comp2_scale=jnp.ones(shape),
        zeta=0.3,
        batch_dim=2,
    )
    assert model.batch_shape == ()
    assert model.event_shape == shape
    assert model.log_prob(jnp.zeros(shape)).shape == ()


@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_nd_two_truncated_normal_mixture_sample_shape(sample_shape):
    model = NDTwoTruncatedNormalMixture(zeta=0.3, **ND_TWO_TRUNCATED_NORMAL)
    assert model.sample(jrd.key(0), sample_shape).shape == model.shape(sample_shape)


def test_nd_two_truncated_normal_mixture_samples_lie_in_the_support():
    model = NDTwoTruncatedNormalMixture(zeta=0.3, **ND_TWO_TRUNCATED_NORMAL)
    assert jnp.all(model.support.check(model.sample(jrd.key(6), (4096,))))


###############################################################################
# NDIsotropicAndTruncatedNormalMixture
###############################################################################


def test_isotropic_mixture_shapes():
    model = NDIsotropicAndTruncatedNormalMixture(zeta=0.4, **ISOTROPIC_AND_NORMAL)
    assert model.batch_shape == ()
    assert model.event_shape == (2,)
    assert model.mixture_size == 2


def test_isotropic_mixture_is_normalised(trapz_nd):
    model = NDIsotropicAndTruncatedNormalMixture(zeta=0.4, **ISOTROPIC_AND_NORMAL)
    assert_allclose(
        trapz_nd(model, [(-1.0, 1.0), (-1.0, 1.0)], num=801), 1.0, atol=1e-4
    )


def test_isotropic_mixture_without_alignment_is_uniform():
    model = NDIsotropicAndTruncatedNormalMixture(zeta=0.0, **ISOTROPIC_AND_NORMAL)
    value = jnp.asarray([[0.3, -0.2], [-0.9, 0.9], [0.0, 0.0]])
    assert_allclose(model.log_prob(value), jnp.full(3, jnp.log(0.25)), rtol=1e-12)


def test_isotropic_mixture_with_full_alignment_is_a_truncated_normal():
    model = NDIsotropicAndTruncatedNormalMixture(zeta=1.0, **ISOTROPIC_AND_NORMAL)
    gaussian = Independent(
        TruncatedNormal(
            loc=ISOTROPIC_AND_NORMAL["loc"],
            scale=ISOTROPIC_AND_NORMAL["scale"],
            low=ISOTROPIC_AND_NORMAL["gaussian_low"],
            high=ISOTROPIC_AND_NORMAL["gaussian_high"],
        ),
        1,
    )
    value = jnp.asarray([[0.3, -0.2], [-0.9, 0.9]])
    assert_allclose(model.log_prob(value), gaussian.log_prob(value), rtol=1e-12)


def test_isotropic_mixture_uniform_component_uses_the_isotropic_bounds():
    model = NDIsotropicAndTruncatedNormalMixture(
        zeta=0.0,
        **{
            **ISOTROPIC_AND_NORMAL,
            "isotropic_low": jnp.asarray([0.0, -0.5]),
            "isotropic_high": jnp.asarray([1.0, 0.5]),
        },
    )
    uniform = Independent(Uniform(jnp.asarray([0.0, -0.5]), jnp.asarray([1.0, 0.5])), 1)
    value = jnp.asarray([0.4, 0.1])
    assert_allclose(model.log_prob(value), uniform.log_prob(value), rtol=1e-12)


def test_isotropic_mixture_accepts_unbounded_gaussian_component():
    model = NDIsotropicAndTruncatedNormalMixture(
        zeta=0.0,
        **{**ISOTROPIC_AND_NORMAL, "gaussian_low": None, "gaussian_high": None},
    )
    assert_allclose(model.log_prob(jnp.zeros(2)), jnp.log(0.25), rtol=1e-12)


def test_isotropic_mixture_is_a_convex_combination():
    zeta = 0.4
    value = jnp.asarray([[0.3, -0.2], [-0.9, 0.9]])
    uniform = NDIsotropicAndTruncatedNormalMixture(
        zeta=0.0, **ISOTROPIC_AND_NORMAL
    ).log_prob(value)
    gaussian = NDIsotropicAndTruncatedNormalMixture(
        zeta=1.0, **ISOTROPIC_AND_NORMAL
    ).log_prob(value)
    expected = jnp.log((1.0 - zeta) * jnp.exp(uniform) + zeta * jnp.exp(gaussian))
    assert_allclose(
        NDIsotropicAndTruncatedNormalMixture(
            zeta=zeta, **ISOTROPIC_AND_NORMAL
        ).log_prob(value),
        expected,
        rtol=1e-12,
    )


@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_isotropic_mixture_sample_shape(sample_shape):
    model = NDIsotropicAndTruncatedNormalMixture(zeta=0.4, **ISOTROPIC_AND_NORMAL)
    assert model.sample(jrd.key(0), sample_shape).shape == model.shape(sample_shape)


def test_isotropic_mixture_samples_lie_in_the_support():
    model = NDIsotropicAndTruncatedNormalMixture(zeta=0.4, **ISOTROPIC_AND_NORMAL)
    assert jnp.all(model.support.check(model.sample(jrd.key(8), (4096,))))


def test_isotropic_mixture_respects_batch_dim():
    shape = (2, 3)
    model = NDIsotropicAndTruncatedNormalMixture(
        zeta=0.4,
        loc=jnp.zeros(shape),
        scale=jnp.ones(shape),
        isotropic_low=-jnp.ones(shape),
        isotropic_high=jnp.ones(shape),
        gaussian_low=-jnp.ones(shape),
        gaussian_high=jnp.ones(shape),
        batch_dim=2,
    )
    assert model.batch_shape == ()
    assert model.event_shape == shape
    assert model.log_prob(jnp.zeros(shape)).shape == ()
