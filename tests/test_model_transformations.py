# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest
from jax import jacfwd, jit, random, vmap
from numpyro.distributions.transforms import biject_to

from gwkokab.models import constraints
from gwkokab.models.transformations import (
    ComponentMassesAndRedshiftToDetectedMassAndRedshift,
    ComponentMassesToChirpMassAndDelta,
    ComponentMassesToChirpMassAndSymmetricMassRatio,
    ComponentMassesToMassRatioAndSecondaryMass,
    ComponentMassesToPrimaryMassAndMassRatio,
    ComponentMassesToTotalMassAndMassRatio,
    DeltaToSymmetricMassRatio,
    PrimaryMassAndMassRatioToComponentMassesTransform,
    SourceMassAndRedshiftToDetectedMassAndRedshift,
)


class T(namedtuple("TestCase", ["transform_cls", "params", "kwargs"])):
    pass


TRANSFORMS = {
    "component_masses_and_redshift_to_detected_mass_and_redshift": T(
        ComponentMassesAndRedshiftToDetectedMassAndRedshift, (), dict()
    ),
    "component_masses_to_chirp_mass_and_delta": T(
        ComponentMassesToChirpMassAndDelta, (), dict()
    ),
    "component_masses_to_chirp_mass_and_symmetric_mass_ratio": T(
        ComponentMassesToChirpMassAndSymmetricMassRatio, (), dict()
    ),
    "component_masses_to_mass_ratio_and_secondary_mass": T(
        ComponentMassesToMassRatioAndSecondaryMass, (), dict()
    ),
    "component_masses_to_primary_mass_and_mass_ratio": T(
        ComponentMassesToPrimaryMassAndMassRatio, (), dict()
    ),
    "component_masses_to_total_mass_and_mass_ratio": T(
        ComponentMassesToTotalMassAndMassRatio, (), dict()
    ),
    "delta_to_symmetric_mass_ratio": T(DeltaToSymmetricMassRatio, (), dict()),
    "primary_mass_and_mass_ratio_to_component_masses_transform": T(
        PrimaryMassAndMassRatioToComponentMassesTransform, (), dict()
    ),
    "source_mass_and_redshift_to_detected_mass_and_redshift": T(
        SourceMassAndRedshiftToDetectedMassAndRedshift, (), dict()
    ),
}


@pytest.mark.parametrize(
    "cls, transform_args, transform_kwargs",
    TRANSFORMS.values(),
    ids=TRANSFORMS.keys(),
)
def test_parametrized_transform_pytree(cls, transform_args, transform_kwargs):
    transform = cls(*transform_args, **transform_kwargs)

    # test that singleton transforms objects can be used as pytrees
    def in_t(transform, x):
        return x**2

    def out_t(transform, x):
        return transform

    jitted_in_t = jit(in_t)
    jitted_out_t = jit(out_t)

    assert jitted_in_t(transform, 1.0) == 1.0
    assert jitted_out_t(transform, 1.0) == transform

    assert jitted_out_t(transform.inv, 1.0) == transform.inv

    assert jnp.allclose(
        vmap(in_t, in_axes=(None, 0), out_axes=0)(transform, jnp.ones(3)),
        jnp.ones(3),
    )

    assert (
        vmap(out_t, in_axes=(None, 0), out_axes=None)(transform, jnp.ones(3))
        == transform
    )

    if len(transform_args) > 0:
        # test creating and manipulating vmapped constraints
        # this test assumes jittable args, and non-jittable kwargs, which is
        # not suited for all transforms, see InverseAutoregressiveTransform.
        # TODO: split among jittable and non-jittable args/kwargs instead.
        vmapped_transform_args = jax.tree.map(lambda x: x[None], transform_args)

        vmapped_transform = jit(
            vmap(lambda args: cls(*args, **transform_kwargs), in_axes=(0,))
        )(vmapped_transform_args)
        assert vmap(lambda x: x == transform, in_axes=0)(vmapped_transform).all()

        twice_vmapped_transform_args = jax.tree.map(
            lambda x: x[None], vmapped_transform_args
        )

        vmapped_transform = jit(
            vmap(
                vmap(lambda args: cls(*args, **transform_kwargs), in_axes=(0,)),
                in_axes=(0,),
            )
        )(twice_vmapped_transform_args)
        assert vmap(vmap(lambda x: x == transform, in_axes=0), in_axes=0)(
            vmapped_transform
        ).all()


@pytest.mark.parametrize(
    "cls, transform_args, transform_kwargs",
    TRANSFORMS.values(),
    ids=TRANSFORMS.keys(),
)
def test_parametrized_transform_eq(cls, transform_args, transform_kwargs):
    transform = cls(*transform_args, **transform_kwargs)
    transform2 = cls(*transform_args, **transform_kwargs)
    assert transform == transform2
    assert transform != 1.0

    # check that equality checks are robust to transforms parametrized
    # by abstract values
    @jit
    def check_transforms(t1, t2):
        return t1 == t2

    assert check_transforms(transform, transform2)


@pytest.mark.parametrize(
    "transform, shape",
    [
        (ComponentMassesAndRedshiftToDetectedMassAndRedshift(), (3,)),
        (ComponentMassesToChirpMassAndDelta(), (2,)),
        (ComponentMassesToChirpMassAndSymmetricMassRatio(), (2,)),
        (ComponentMassesToMassRatioAndSecondaryMass(), (2,)),
        (ComponentMassesToPrimaryMassAndMassRatio(), (2,)),
        (ComponentMassesToTotalMassAndMassRatio(), (2,)),
        (DeltaToSymmetricMassRatio(), ()),
        (PrimaryMassAndMassRatioToComponentMassesTransform(), (2,)),
        (SourceMassAndRedshiftToDetectedMassAndRedshift(), (2,)),
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
        (constraints.positive_decreasing_vector, (5,)),
        (constraints.decreasing_vector, (5,)),
        (constraints.strictly_decreasing_vector, (5,)),
        (constraints.positive_increasing_vector, (5,)),
        (constraints.increasing_vector, (5,)),
        (constraints.strictly_increasing_vector, (5,)),
        (constraints.mass_sandwich(10.0, 50.0), (2,)),
        (constraints.mass_ratio_mass_sandwich(10.0, 50.0), (2,)),
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
