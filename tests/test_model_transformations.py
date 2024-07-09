# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import jax.numpy as jnp
import pytest
from jax import jacfwd, random, vmap
from numpyro.distributions.transforms import biject_to

from gwkokab.models import constraints
from gwkokab.models.transformations import (
    DeltaToSymmetricMassRatio,
)


@pytest.mark.parametrize(
    "transform, shape",
    [
        (DeltaToSymmetricMassRatio(), ()),
    ],
)
def test_bijective_transforms(transform, shape):
    if isinstance(transform, type):
        pytest.skip()
    # Get a sample from the support of the distribution.
    batch_shape = (13,)
    unconstrained = random.normal(random.key(17), batch_shape + shape)
    x1 = biject_to(transform.domain)(unconstrained)

    # Transform forward and backward, checking shapes, values, and Jacobian shape.
    y = transform(x1)
    assert y.shape == transform.forward_shape(x1.shape)

    x2 = transform.inv(y)
    assert x2.shape == transform.inverse_shape(y.shape)
    # Some transforms are a bit less stable; we give them larger tolerances.
    atol = 1e-6
    assert jnp.allclose(x1, x2, atol=atol, equal_nan=True)

    log_abs_det_jacobian = transform.log_abs_det_jacobian(x1, y)
    assert log_abs_det_jacobian.shape == batch_shape

    # Also check the Jacobian numerically for transforms with the same input and output
    # size, unless they are explicitly excluded. E.g., the upper triangular of the
    # CholeskyTransform is zero, giving rise to a singular Jacobian.
    size_x = int(x1.size / math.prod(batch_shape))
    size_y = int(y.size / math.prod(batch_shape))
    if size_x == size_y:
        jac = (
            vmap(jacfwd(transform))(x1)
            .reshape((-1,) + x1.shape[len(batch_shape) :])
            .reshape(batch_shape + (size_y, size_x))
        )
        slogdet = jnp.linalg.slogdet(jac)
        assert jnp.allclose(log_abs_det_jacobian, slogdet.logabsdet, atol=atol)


@pytest.mark.parametrize(
    "constraint, shape",
    [
        (constraints.unique_intervals((-10, -8, -6, -4), (4, 6, 8, 10)), (4,)),
        (constraints.positive_decreasing_vector, (5,)),
        (constraints.decreasing_vector, (5,)),
        (constraints.strictly_decreasing_vector, (5,)),
        (constraints.positive_increasing_vector, (5,)),
        (constraints.increasing_vector, (5,)),
        (constraints.strictly_increasing_vector, (5,)),
    ],
    ids=str,
)
def test_biject_to(constraint, shape):
    batch_shape = (13, 19)
    unconstrained = random.normal(random.key(93), batch_shape + shape)
    constrained = biject_to(constraint)(unconstrained)
    passed = constraint.check(constrained)
    expected_shape = constrained.shape[: constrained.ndim - constraint.event_dim]
    assert passed.shape == expected_shape
    assert jnp.all(passed)
