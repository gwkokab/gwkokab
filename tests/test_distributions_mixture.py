# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrd
import numpyro.distributions as dist
import pytest
from jaxtyping import Array
from numpyro.distributions import Distribution, HalfNormal, MultivariateNormal, Normal

from gwkokab.models.utils import ScaledMixture


rng_key = jax.random.PRNGKey(42)


def get_normal(batch_shape: Tuple[int, ...]) -> Normal:
    """Get parameterized Normal with given batch shape."""
    loc = jnp.zeros(batch_shape)
    scale = jnp.ones(batch_shape)
    normal = Normal(loc=loc, scale=scale)
    return normal


def get_half_normal(batch_shape: Tuple[int, ...]) -> HalfNormal:
    """Get parameterized HalfNormal with given batch shape."""
    scale = jnp.ones(batch_shape)
    half_normal = HalfNormal(scale=scale)
    return half_normal


def get_mvn(batch_shape: Tuple[int, ...]) -> MultivariateNormal:
    """Get parameterized MultivariateNormal with given batch shape."""
    dimensions = 2
    loc = jnp.zeros((*batch_shape, dimensions))
    cov_matrix = jnp.eye(dimensions, dimensions)
    for i, s in enumerate(batch_shape):
        loc = jnp.repeat(jnp.expand_dims(loc, i), s, axis=i)
        cov_matrix = jnp.repeat(jnp.expand_dims(cov_matrix, i), s, axis=i)
    mvn = MultivariateNormal(loc=loc, covariance_matrix=cov_matrix)
    return mvn


@pytest.mark.parametrize("jax_dist_getter", [get_normal, get_mvn])
@pytest.mark.parametrize("nb_mixtures", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (1,), (7,), (2, 5)])
def test_mixture_same_batch_shape(jax_dist_getter, nb_mixtures, batch_shape):
    log_scales = jrd.normal(rng_key, (nb_mixtures,)) * jnp.pi
    for i, s in enumerate(batch_shape):
        log_scales = jnp.repeat(jnp.expand_dims(log_scales, i), s, axis=i)
    component_distribution = [jax_dist_getter(batch_shape) for _ in range(nb_mixtures)]
    _test_mixture(log_scales, component_distribution)


@pytest.mark.parametrize("jax_dist_getter", [get_normal, get_mvn])
@pytest.mark.parametrize("nb_mixtures", [3])
@pytest.mark.parametrize("mixing_batch_shape, component_batch_shape", [[(2,), (7, 2)]])
def test_mixture_broadcast_batch_shape(
    jax_dist_getter, nb_mixtures, mixing_batch_shape, component_batch_shape
):
    # Create mixture
    log_scales = jrd.normal(rng_key, (nb_mixtures,)) * jnp.pi
    for i, s in enumerate(mixing_batch_shape):
        log_scales = jnp.repeat(jnp.expand_dims(log_scales, i), s, axis=i)
    component_distribution = [
        jax_dist_getter(component_batch_shape) for _ in range(nb_mixtures)
    ]
    _test_mixture(log_scales, component_distribution)


@pytest.mark.parametrize("batch_shape", [(), (1,), (7,), (2, 5)])
@pytest.mark.filterwarnings(
    "ignore:Out-of-support values provided to log prob method."
    " The value argument should be within the support.:UserWarning"
)
def test_mixture_with_different_support(batch_shape):
    log_scales = jrd.normal(rng_key, (2,)) * jnp.pi
    component_distribution = [
        get_normal(batch_shape),
        get_half_normal(batch_shape),
    ]
    mixture = ScaledMixture(
        log_scales=log_scales,
        component_distributions=component_distribution,
        support=dist.constraints.real,
    )
    assert mixture.batch_shape == batch_shape
    sample_shape = (11,)
    component_distribution[0]._validate_args = True
    component_distribution[1]._validate_args = True
    xx = component_distribution[0].sample(rng_key, sample_shape)
    log_prob_0 = component_distribution[0].log_prob(xx)
    log_prob_1 = component_distribution[1].log_prob(xx)
    expected_log_prob = jax.scipy.special.logsumexp(
        jnp.stack([log_prob_0 + log_scales[0], log_prob_1 + log_scales[1]], axis=-1),
        axis=-1,
    )
    result = mixture.log_prob(xx)
    assert jnp.allclose(result, expected_log_prob)


def _test_mixture(log_scales: Array, component_distribution: Sequence[Distribution]):
    # Create mixture
    mixture = ScaledMixture(
        log_scales=log_scales,
        component_distributions=component_distribution,
    )
    assert mixture.mixture_size == log_scales.shape[-1], (
        "Mixture size needs to be the size of the probability vector"
    )

    if isinstance(component_distribution, dist.Distribution):
        assert mixture.batch_shape == component_distribution.batch_shape[:-1], (
            "Mixture batch shape needs to be the component batch shape without the mixture dimension."
        )
    else:
        assert mixture.batch_shape == component_distribution[0].batch_shape, (
            "Mixture batch shape needs to be the component batch shape."
        )
    # Test samples
    sample_shape = (11,)
    # Samples from component distribution(s)
    component_samples = mixture.component_sample(rng_key, sample_shape)
    assert component_samples.shape == (
        *sample_shape,
        *mixture.batch_shape,
        mixture.mixture_size,
        *mixture.event_shape,
    )
    # Samples from mixture
    samples = mixture.sample(rng_key, sample_shape=sample_shape)
    assert samples.shape == (*sample_shape, *mixture.batch_shape, *mixture.event_shape)
    # Check log_prob
    lp = mixture.log_prob(samples)
    nb_value_dims = len(samples.shape) - mixture.event_dim
    expected_shape = samples.shape[:nb_value_dims]
    assert lp.shape == expected_shape
    # Samples with indices
    samples_, [indices] = mixture.sample_with_intermediates(
        rng_key, sample_shape=sample_shape
    )
    assert samples_.shape == samples.shape
    assert indices.shape == (*sample_shape, *mixture.batch_shape)
    assert jnp.issubdtype(indices.dtype, jnp.integer)
    assert (indices >= 0).all() and (indices < mixture.mixture_size).all()
    # Check mean
    mean = mixture.mean
    assert mean.shape == mixture.shape()
    # Check variance
    var = mixture.variance
    assert var.shape == mixture.shape()
    # Check cdf
    if mixture.event_shape == ():
        cdf = mixture.cdf(samples)
        assert cdf.shape == (*sample_shape, *mixture.shape())
