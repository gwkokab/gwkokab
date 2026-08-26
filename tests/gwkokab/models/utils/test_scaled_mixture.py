# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.utils._scaledmixture`."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose
from numpyro.distributions import (
    constraints,
    MultivariateNormal,
    Normal,
    Uniform,
)

from gwkokab.models.constraints import any_constraint
from gwkokab.models.utils import ScaledMixture


LOG_SCALES = jnp.log(jnp.asarray([1.0, 2.0, 3.0]))


def _components():
    return [Normal(0.0, 1.0), Normal(-0.5, 0.3), Normal(0.6, 1.2)]


def _mixture(**kwargs):
    return ScaledMixture(LOG_SCALES, _components(), **kwargs)


###############################################################################
# construction
###############################################################################


def test_shapes_and_metadata():
    mixture = _mixture()
    assert mixture.batch_shape == ()
    assert mixture.event_shape == ()
    assert mixture.mixture_size == 3
    assert mixture.mixture_dim == -1
    assert mixture.is_discrete is False
    assert len(mixture.component_distributions) == 3


def test_support_defaults_to_the_first_component():
    mixture = _mixture()
    assert isinstance(mixture.support, constraints._Real)


def test_an_explicit_support_overrides_the_components():
    mixture = ScaledMixture(
        LOG_SCALES, _components(), support=constraints.interval(-1.0, 1.0)
    )
    assert mixture.support == constraints.interval(-1.0, 1.0)


def test_an_explicit_support_masks_each_component_by_its_own_support():
    # passing a support switches on per-component masking, which is what lets a mixture
    # of disjointly supported components behave correctly
    components = [Uniform(0.0, 1.0), Uniform(2.0, 3.0)]
    mixture = ScaledMixture(
        jnp.log(jnp.asarray([1.0, 1.0])),
        components,
        support=any_constraint(tuple(d.support for d in components)),
    )
    value = jnp.asarray([0.5, 1.5, 2.5])
    assert_allclose(mixture.log_prob(value), [0.0, -jnp.inf, 0.0])


def test_without_an_explicit_support_components_are_not_masked():
    # numpyro's Uniform returns its constant log density everywhere, so the gap between
    # the two components is filled in unless a support is supplied
    mixture = ScaledMixture(
        jnp.log(jnp.asarray([1.0, 1.0])), [Uniform(0.0, 1.0), Uniform(2.0, 3.0)]
    )
    assert_allclose(mixture.log_prob(jnp.asarray(1.5)), jnp.log(2.0), rtol=1e-12)
    assert mixture.support == constraints.interval(0.0, 1.0)


def test_multi_dimensional_events():
    components = [MultivariateNormal(jnp.zeros(2), jnp.eye(2)) for _ in range(3)]
    mixture = ScaledMixture(LOG_SCALES, components)
    assert mixture.event_shape == (2,)
    assert mixture.mixture_dim == -2
    assert mixture.log_prob(jnp.zeros(2)).shape == ()
    assert mixture.sample(jrd.key(0), (4,)).shape == (4, 2)


def test_component_batch_shapes_are_broadcast():
    mixture = ScaledMixture(
        jnp.log(jnp.asarray([[1.0, 2.0], [3.0, 4.0]])),
        [Normal(jnp.zeros(2), 1.0), Normal(jnp.ones(2), 1.0)],
    )
    assert mixture.batch_shape == (2,)
    assert mixture.log_prob(jnp.asarray(0.5)).shape == (2,)
    assert mixture.sample(jrd.key(0), (5,)).shape == (5, 2)


###############################################################################
# construction errors
###############################################################################


def test_rejects_a_non_iterable_component_argument():
    with pytest.raises(ValueError, match="must be a list of Distribution objects"):
        ScaledMixture(LOG_SCALES, 5)


def test_rejects_non_distribution_components():
    with pytest.raises(ValueError, match="must be instances of"):
        ScaledMixture(LOG_SCALES, [Normal(0.0, 1.0), Normal(0.0, 1.0), 5])


def test_rejects_a_component_count_that_does_not_match_the_scales():
    with pytest.raises(ValueError, match="must match the mixture size"):
        ScaledMixture(LOG_SCALES, _components()[:2])


def test_rejects_components_with_different_supports():
    with pytest.raises(ValueError, match="same support"):
        ScaledMixture(
            LOG_SCALES, [Normal(0.0, 1.0), Normal(0.0, 1.0), Uniform(0.0, 1.0)]
        )


def test_rejects_components_with_different_event_shapes():
    # an explicit support gets past the support check, so the event-shape check fires
    with pytest.raises(ValueError, match="same event shape"):
        ScaledMixture(
            jnp.log(jnp.asarray([1.0, 2.0])),
            [Normal(0.0, 1.0), MultivariateNormal(jnp.zeros(2), jnp.eye(2))],
            support=constraints.real,
        )


def test_rejects_a_non_constraint_support():
    with pytest.raises(AssertionError, match="support must be a Constraint"):
        ScaledMixture(LOG_SCALES, _components(), support="real")


###############################################################################
# the density
###############################################################################


def test_log_prob_is_the_scaled_log_sum_exp():
    mixture = _mixture()
    value = jnp.asarray([-1.0, 0.0, 0.3, 2.0])
    stacked = jnp.stack([c.log_prob(value) for c in _components()], axis=-1)
    expected = jax.nn.logsumexp(LOG_SCALES + stacked, axis=-1)
    assert_allclose(mixture.log_prob(value), expected, rtol=1e-12)


def test_component_log_probs_carry_the_scales():
    mixture = _mixture()
    value = jnp.asarray(0.3)
    expected = LOG_SCALES + jnp.stack([c.log_prob(value) for c in _components()])
    assert_allclose(mixture.component_log_probs(value), expected, rtol=1e-12)


def test_the_density_integrates_to_the_total_rate(trapz_1d):
    mixture = _mixture()
    total_rate = float(jnp.sum(jnp.exp(LOG_SCALES)))
    assert_allclose(trapz_1d(mixture, -30.0, 30.0, 60_001), total_rate, rtol=1e-6)


def test_unit_scales_reduce_to_an_ordinary_mixture(trapz_1d):
    mixture = ScaledMixture(jnp.log(jnp.asarray([0.25, 0.25, 0.5])), _components())
    assert_allclose(trapz_1d(mixture, -30.0, 30.0, 60_001), 1.0, rtol=1e-6)


def test_a_zero_scale_removes_a_component():
    scales = jnp.asarray([0.0, -jnp.inf, jnp.log(3.0)])
    mixture = ScaledMixture(scales, _components())
    value = jnp.asarray([-1.0, 0.3, 2.0])
    survivors = [_components()[0], _components()[2]]
    expected = jax.nn.logsumexp(
        jnp.stack(
            [
                jnp.log(1.0) + survivors[0].log_prob(value),
                jnp.log(3.0) + survivors[1].log_prob(value),
            ],
            axis=-1,
        ),
        axis=-1,
    )
    assert_allclose(mixture.log_prob(value), expected, rtol=1e-12)


def test_all_scales_zero_gives_no_density():
    mixture = ScaledMixture(jnp.full(3, -jnp.inf), _components())
    assert jnp.isneginf(mixture.log_prob(jnp.asarray(0.3)))


@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_log_prob_shape(sample_shape):
    mixture = _mixture()
    assert mixture.log_prob(jnp.zeros(sample_shape)).shape == sample_shape


###############################################################################
# moments and the cdf
###############################################################################


def test_component_moments_are_stacked_along_the_mixture_axis():
    mixture = _mixture()
    assert_allclose(mixture.component_mean, [c.mean for c in _components()], rtol=1e-12)
    assert_allclose(
        mixture.component_variance, [c.variance for c in _components()], rtol=1e-12
    )


def test_mean_is_the_scale_weighted_component_mean():
    mixture = _mixture()
    expected = sum(
        float(jnp.exp(scale)) * c.mean for scale, c in zip(LOG_SCALES, _components())
    )
    assert_allclose(mixture.mean, expected, rtol=1e-12)


def test_variance_is_the_law_of_total_variance():
    mixture = _mixture()
    probs = jnp.exp(LOG_SCALES)
    means = jnp.asarray([c.mean for c in _components()])
    variances = jnp.asarray([c.variance for c in _components()])
    expected = jnp.sum(probs * variances) + jnp.sum(
        probs * jnp.square(means - mixture.mean)
    )
    assert_allclose(mixture.variance, expected, rtol=1e-12)


def test_cdf_is_the_scale_weighted_component_cdf():
    mixture = _mixture()
    value = jnp.asarray(0.4)
    expected = sum(
        float(jnp.exp(scale)) * c.cdf(value)
        for scale, c in zip(LOG_SCALES, _components())
    )
    assert_allclose(mixture.cdf(value), expected, rtol=1e-12)
    assert_allclose(
        mixture.component_cdf(value), [c.cdf(value) for c in _components()], rtol=1e-12
    )


###############################################################################
# sampling
###############################################################################


@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_sample_shape(sample_shape):
    mixture = _mixture()
    assert mixture.sample(jrd.key(0), sample_shape).shape == mixture.shape(sample_shape)


def test_component_sample_shape():
    mixture = _mixture()
    assert mixture.component_sample(jrd.key(0), (4,)).shape == (4, 3)


def test_sample_with_intermediates_returns_the_component_indices():
    mixture = _mixture()
    samples, (indices,) = mixture.sample_with_intermediates(jrd.key(0), (16,))
    assert samples.shape == (16,)
    assert indices.shape == (16,)
    assert jnp.all((indices >= 0) & (indices < 3))


def test_component_choice_follows_the_softmax_of_the_scales():
    mixture = _mixture()
    _, (indices,) = mixture.sample_with_intermediates(jrd.key(3), (200_000,))
    counts = np.bincount(np.asarray(indices), minlength=3) / indices.size
    assert_allclose(counts, np.asarray(jax.nn.softmax(LOG_SCALES)), atol=5e-3)


def test_samples_follow_the_normalised_density():
    mixture = _mixture()
    samples = np.asarray(mixture.sample(jrd.key(4), (200_000,)))
    total_rate = float(jnp.sum(jnp.exp(LOG_SCALES)))
    probe = np.linspace(-4.0, 4.0, 17)
    empirical = np.mean(samples[:, None] <= probe[None, :], axis=0)
    exact = np.asarray(mixture.cdf(jnp.asarray(probe))) / total_rate
    assert_allclose(empirical, exact, atol=5e-3)


def test_sample_rejects_a_plain_integer_key():
    with pytest.raises(AssertionError):
        _mixture().sample(0)


###############################################################################
# jax integration
###############################################################################


def test_pytree_roundtrip(pytree_roundtrip):
    mixture = _mixture()
    value = jnp.asarray(0.3)
    assert_allclose(pytree_roundtrip(mixture).log_prob(value), mixture.log_prob(value))


def test_under_jit_and_grad():
    value = jnp.asarray(0.3)

    def log_prob(log_scales):
        return ScaledMixture(log_scales, _components()).log_prob(value)

    assert_allclose(jax.jit(log_prob)(LOG_SCALES), log_prob(LOG_SCALES), rtol=1e-12)

    gradient = jax.grad(log_prob)(LOG_SCALES)
    # d/d(log s_k) log sum_j s_j p_j = s_k p_k / sum_j s_j p_j, which sums to one
    assert_allclose(jnp.sum(gradient), 1.0, rtol=1e-10)


def test_under_vmap():
    values = jnp.asarray([-1.0, 0.0, 1.0])
    mixture = _mixture()
    assert_allclose(
        jax.vmap(mixture.log_prob)(values), mixture.log_prob(values), rtol=1e-12
    )


def test_validate_args_warns_outside_an_explicit_support():
    mixture = ScaledMixture(
        LOG_SCALES,
        _components(),
        support=constraints.interval(-1.0, 1.0),
        validate_args=True,
    )
    with pytest.warns(UserWarning, match="Out-of-support values"):
        mixture.log_prob(jnp.asarray(5.0))


def test_scales_that_are_not_an_array_are_rejected():
    # the mixture size is read off log_scales.shape, so a plain list has no shape
    with pytest.raises(AttributeError):
        ScaledMixture([0.0, 0.0, 0.0], _components())
