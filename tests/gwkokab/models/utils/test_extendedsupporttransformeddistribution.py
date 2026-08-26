# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for
:mod:`gwkokab.models.utils._extendedsupporttransformeddistribution`.
"""

import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose
from numpyro.distributions import TransformedDistribution, Uniform
from numpyro.distributions.transforms import AffineTransform, ExpTransform

from gwkokab.models.constraints import transform_constraint
from gwkokab.models.mass import PowerlawPrimaryMassRatio
from gwkokab.models.transformations import (
    PrimaryMassAndMassRatioToComponentMassesTransform,
)
from gwkokab.models.utils import ExtendedSupportTransformedDistribution


def _mass_model(**kwargs):
    base = PowerlawPrimaryMassRatio(alpha=1.0, beta=1.0, mmin=10.0, mmax=50.0)
    transform = PrimaryMassAndMassRatioToComponentMassesTransform()
    return (
        ExtendedSupportTransformedDistribution(
            base_distribution=base, transforms=transform, **kwargs
        ),
        TransformedDistribution(base, transform, **kwargs),
    )


def test_it_is_a_transformed_distribution():
    extended, plain = _mass_model()
    assert isinstance(extended, TransformedDistribution)
    assert extended.batch_shape == plain.batch_shape
    assert extended.event_shape == plain.event_shape


def test_the_density_is_unchanged():
    extended, plain = _mass_model()
    value = jnp.asarray([[30.0, 20.0], [45.0, 12.0], [11.0, 10.5]])
    assert_allclose(extended.log_prob(value), plain.log_prob(value), rtol=1e-12)


def test_the_support_is_the_pushforward_of_the_base_support():
    extended, _ = _mass_model()
    base = PowerlawPrimaryMassRatio(alpha=1.0, beta=1.0, mmin=10.0, mmax=50.0)
    transform = PrimaryMassAndMassRatioToComponentMassesTransform()
    assert extended.support == transform_constraint(base.support, [transform])


def test_the_support_is_tighter_than_the_plain_transformed_one():
    extended, plain = _mass_model()
    # (5, 2) is a decreasing pair of positive masses -- which is all the codomain of the
    # transform knows -- but its primary mass is below mmin
    value = jnp.asarray([[30.0, 20.0], [30.0, 40.0], [5.0, 2.0]])
    assert_allclose(extended.support.check(value), [True, False, False])
    assert_allclose(plain.support.check(value), [True, False, True])


def test_samples_lie_in_the_extended_support():
    extended, _ = _mass_model()
    samples = extended.sample(jrd.key(0), (4096,))
    assert jnp.all(extended.support.check(samples))


@pytest.mark.parametrize("sample_shape", [(), (5,), (2, 3)])
def test_sample_shape(sample_shape):
    extended, _ = _mass_model()
    assert extended.sample(jrd.key(0), sample_shape).shape == extended.shape(
        sample_shape
    )


def test_a_bounded_base_keeps_its_bounds_through_a_monotone_transform():
    base = Uniform(0.0, 1.0)
    extended = ExtendedSupportTransformedDistribution(
        base_distribution=base, transforms=[ExpTransform()]
    )
    plain = TransformedDistribution(base, [ExpTransform()])
    value = jnp.asarray([0.5, 1.5, 3.0])
    # exp maps [0, 1] onto [1, e]; the plain support only records "positive"
    assert_allclose(extended.support.check(value), [False, True, False])
    assert jnp.all(plain.support.check(value))


def test_a_chain_of_transforms_is_composed():
    base = Uniform(0.0, 1.0)
    transforms = [AffineTransform(1.0, 2.0), ExpTransform()]
    extended = ExtendedSupportTransformedDistribution(
        base_distribution=base, transforms=transforms
    )
    # the chain maps [0, 1] to [e, e^3]
    value = jnp.asarray([jnp.e * 0.5, jnp.e * 1.5, jnp.e**4])
    assert_allclose(extended.support.check(value), [False, True, False])


def test_pytree_roundtrip(pytree_roundtrip):
    extended, _ = _mass_model()
    value = jnp.asarray([30.0, 20.0])
    assert_allclose(
        pytree_roundtrip(extended).log_prob(value), extended.log_prob(value)
    )
