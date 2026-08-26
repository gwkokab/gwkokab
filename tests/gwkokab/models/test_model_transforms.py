# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Value semantics of the transforms in :mod:`gwkokab.models.transformations`.

``tests/test_model_transformations.py`` checks that each transform is bijective and
behaves as a pytree; this module checks that it computes the coordinate change it
advertises, that its log Jacobian matches automatic differentiation, and that its domain
and codomain agree with the maps themselves.
"""

import jax
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose
from numpyro.distributions.transforms import (
    AffineTransform,
    ExpTransform,
    OrderedTransform,
)

from gwkokab.cosmology import Cosmology, default_cosmology
from gwkokab.models.transformations import (
    BlockTransform,
    ComponentMassesAndRedshiftToDetectedMassAndRedshift,
    ComponentMassesToChirpMassAndDelta,
    ComponentMassesToChirpMassAndSymmetricMassRatio,
    ComponentMassesToMassRatioAndSecondaryMass,
    ComponentMassesToPrimaryMassAndMassRatio,
    ComponentMassesToTotalMassAndMassRatio,
    DeltaToSymmetricMassRatio,
    PrimaryMassAndMassRatioToComponentMassesTransform,
    RedshiftToLuminosityDistance,
    SourceMassAndRedshiftToDetectedMassAndRedshift,
)
from gwkokab.utils.transformations import (
    chirp_mass,
    delta_m,
    mass_ratio,
    symmetric_mass_ratio,
    total_mass,
)


COMPONENT_MASSES = jnp.asarray([[30.0, 20.0], [45.0, 12.0], [11.0, 10.5], [8.0, 8.0]])
REDSHIFTS = jnp.asarray([0.05, 0.4, 1.2, 2.5])


def _autodiff_log_det(transform, x):
    """The log absolute determinant of the Jacobian, straight from ``jax.jacfwd``."""
    if x.ndim == 0:
        return jnp.log(jnp.abs(jax.grad(transform)(x)))
    jacobian = jax.jacfwd(transform)(x)
    return jnp.log(jnp.abs(jnp.linalg.det(jacobian)))


###############################################################################
# the mass coordinate changes
###############################################################################


def test_primary_mass_and_mass_ratio_to_component_masses():
    transform = PrimaryMassAndMassRatioToComponentMassesTransform()
    x = jnp.asarray([[30.0, 0.5], [45.0, 0.9]])
    assert_allclose(transform(x), [[30.0, 15.0], [45.0, 40.5]], rtol=1e-12)
    assert_allclose(transform.inv(transform(x)), x, rtol=1e-12)


def test_component_masses_to_primary_mass_and_mass_ratio():
    transform = ComponentMassesToPrimaryMassAndMassRatio()
    m1, m2 = jnp.unstack(COMPONENT_MASSES, axis=-1)
    assert_allclose(
        transform(COMPONENT_MASSES),
        jnp.stack([m1, mass_ratio(m1=m1, m2=m2)], axis=-1),
        rtol=1e-12,
    )
    assert_allclose(
        transform.inv(transform(COMPONENT_MASSES)), COMPONENT_MASSES, rtol=1e-12
    )


def test_the_two_primary_mass_and_mass_ratio_transforms_are_inverses():
    forward = PrimaryMassAndMassRatioToComponentMassesTransform()
    backward = ComponentMassesToPrimaryMassAndMassRatio()
    assert_allclose(forward(backward(COMPONENT_MASSES)), COMPONENT_MASSES, rtol=1e-12)


def test_component_masses_to_chirp_mass_and_symmetric_mass_ratio():
    transform = ComponentMassesToChirpMassAndSymmetricMassRatio()
    m1, m2 = jnp.unstack(COMPONENT_MASSES, axis=-1)
    expected = jnp.stack(
        [chirp_mass(m1=m1, m2=m2), symmetric_mass_ratio(m1=m1, m2=m2)], axis=-1
    )
    assert_allclose(transform(COMPONENT_MASSES), expected, rtol=1e-12)
    assert_allclose(
        transform.inv(transform(COMPONENT_MASSES)), COMPONENT_MASSES, rtol=1e-8
    )


def test_symmetric_mass_ratio_peaks_at_equal_masses():
    transform = ComponentMassesToChirpMassAndSymmetricMassRatio()
    eta = transform(COMPONENT_MASSES)[..., 1]
    assert jnp.all(eta <= 0.25)
    assert_allclose(eta[-1], 0.25, rtol=1e-12)


def test_component_masses_to_chirp_mass_and_delta():
    transform = ComponentMassesToChirpMassAndDelta()
    m1, m2 = jnp.unstack(COMPONENT_MASSES, axis=-1)
    expected = jnp.stack([chirp_mass(m1=m1, m2=m2), delta_m(m1=m1, m2=m2)], axis=-1)
    assert_allclose(transform(COMPONENT_MASSES), expected, rtol=1e-12)
    assert_allclose(
        transform.inv(transform(COMPONENT_MASSES)), COMPONENT_MASSES, rtol=1e-8
    )


@pytest.mark.parametrize("delta", [0.0, 0.25, 0.5, 0.99])
def test_delta_to_symmetric_mass_ratio(delta):
    transform = DeltaToSymmetricMassRatio()
    x = jnp.asarray(delta)
    assert_allclose(transform(x), 0.25 * (1.0 - delta**2), rtol=1e-12)
    assert_allclose(transform.inv(transform(x)), x, atol=1e-8)


def test_delta_to_symmetric_mass_ratio_endpoints():
    transform = DeltaToSymmetricMassRatio()
    # equal masses (delta = 0) give eta = 1/4; an extreme ratio (delta -> 1) gives 0
    assert_allclose(transform(jnp.asarray(0.0)), 0.25, rtol=1e-12)
    assert_allclose(transform(jnp.asarray(1.0)), 0.0, atol=1e-12)


def test_component_masses_to_mass_ratio_and_secondary_mass():
    transform = ComponentMassesToMassRatioAndSecondaryMass()
    m1, m2 = jnp.unstack(COMPONENT_MASSES, axis=-1)
    expected = jnp.stack([mass_ratio(m1=m1, m2=m2), m2], axis=-1)
    assert_allclose(transform(COMPONENT_MASSES), expected, rtol=1e-12)
    assert_allclose(
        transform.inv(transform(COMPONENT_MASSES)), COMPONENT_MASSES, rtol=1e-12
    )


def test_component_masses_to_total_mass_and_mass_ratio():
    transform = ComponentMassesToTotalMassAndMassRatio()
    m1, m2 = jnp.unstack(COMPONENT_MASSES, axis=-1)
    expected = jnp.stack([total_mass(m1=m1, m2=m2), mass_ratio(m1=m1, m2=m2)], axis=-1)
    assert_allclose(transform(COMPONENT_MASSES), expected, rtol=1e-12)
    assert_allclose(
        transform.inv(transform(COMPONENT_MASSES)), COMPONENT_MASSES, rtol=1e-12
    )


def test_total_mass_and_mass_ratio_inverse_guards_the_degenerate_ratio():
    # q == -1 would divide by zero, so the primary mass is sent to infinity instead
    transform = ComponentMassesToTotalMassAndMassRatio()
    recovered = transform.inv(jnp.asarray([50.0, -1.0]))
    assert jnp.isposinf(recovered[0])


###############################################################################
# redshifting
###############################################################################


def test_source_mass_and_redshift_to_detected_mass_and_redshift():
    transform = SourceMassAndRedshiftToDetectedMassAndRedshift()
    x = jnp.stack([jnp.asarray([30.0, 45.0, 11.0, 8.0]), REDSHIFTS], axis=-1)
    detected = transform(x)
    assert_allclose(detected[..., 0], x[..., 0] * (1.0 + REDSHIFTS), rtol=1e-12)
    assert_allclose(detected[..., 1], REDSHIFTS, rtol=1e-12)
    assert_allclose(transform.inv(detected), x, rtol=1e-12)


def test_component_masses_and_redshift_to_detected_masses_and_redshift():
    transform = ComponentMassesAndRedshiftToDetectedMassAndRedshift()
    x = jnp.concatenate([COMPONENT_MASSES, REDSHIFTS[:, None]], axis=-1)
    detected = transform(x)
    assert_allclose(
        detected[..., :2], COMPONENT_MASSES * (1.0 + REDSHIFTS[:, None]), rtol=1e-12
    )
    assert_allclose(detected[..., 2], REDSHIFTS, rtol=1e-12)
    assert_allclose(transform.inv(detected), x, rtol=1e-12)


def test_a_zero_redshift_leaves_the_masses_alone():
    transform = ComponentMassesAndRedshiftToDetectedMassAndRedshift()
    x = jnp.asarray([30.0, 20.0, 0.0])
    assert_allclose(transform(x), x, rtol=1e-12)


def test_redshift_to_luminosity_distance_round_trips():
    cosmology = default_cosmology()
    transform = RedshiftToLuminosityDistance(cosmology)
    z = jnp.asarray([0.05, 0.4, 1.2])
    assert_allclose(transform(z), cosmology.z_to_DL(z), rtol=1e-12)
    assert_allclose(transform.inv(transform(z)), z, rtol=1e-4)


def test_redshift_to_luminosity_distance_is_increasing():
    transform = RedshiftToLuminosityDistance(default_cosmology())
    distances = transform(jnp.linspace(0.01, 3.0, 50))
    assert jnp.all(jnp.diff(distances) > 0.0)


def test_redshift_to_luminosity_distance_keeps_its_cosmology_in_the_pytree():
    cosmology = default_cosmology()
    transform = RedshiftToLuminosityDistance(cosmology)
    leaves, treedef = jax.tree_util.tree_flatten(transform)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt.cosmology, Cosmology)
    z = jnp.asarray(0.4)
    assert_allclose(rebuilt(z), transform(z), rtol=1e-12)


###############################################################################
# log Jacobians against automatic differentiation
###############################################################################


@pytest.mark.parametrize(
    "transform, x",
    [
        (PrimaryMassAndMassRatioToComponentMassesTransform(), [30.0, 0.5]),
        (ComponentMassesToPrimaryMassAndMassRatio(), [30.0, 20.0]),
        (ComponentMassesToChirpMassAndSymmetricMassRatio(), [30.0, 20.0]),
        (ComponentMassesToChirpMassAndDelta(), [30.0, 20.0]),
        (ComponentMassesToMassRatioAndSecondaryMass(), [30.0, 20.0]),
        (ComponentMassesToTotalMassAndMassRatio(), [30.0, 20.0]),
        (SourceMassAndRedshiftToDetectedMassAndRedshift(), [30.0, 0.4]),
        (ComponentMassesAndRedshiftToDetectedMassAndRedshift(), [30.0, 20.0, 0.4]),
    ],
)
def test_log_abs_det_jacobian_matches_autodiff(transform, x):
    x = jnp.asarray(x)
    y = transform(x)
    assert_allclose(
        transform.log_abs_det_jacobian(x, y), _autodiff_log_det(transform, x), rtol=1e-8
    )


def test_delta_to_symmetric_mass_ratio_log_jacobian_matches_autodiff():
    transform = DeltaToSymmetricMassRatio()
    x = jnp.asarray(0.4)
    assert_allclose(
        transform.log_abs_det_jacobian(x, transform(x)),
        _autodiff_log_det(transform, x),
        rtol=1e-8,
    )


def test_redshift_to_luminosity_distance_log_jacobian_matches_autodiff():
    transform = RedshiftToLuminosityDistance(default_cosmology())
    x = jnp.asarray(0.4)
    assert_allclose(
        transform.log_abs_det_jacobian(x, transform(x)),
        _autodiff_log_det(transform, x),
        rtol=1e-4,
    )


###############################################################################
# domains and codomains
###############################################################################


@pytest.mark.parametrize(
    "transform, inside, outside",
    [
        (
            PrimaryMassAndMassRatioToComponentMassesTransform(),
            [30.0, 0.5],
            [30.0, 1.5],
        ),
        (ComponentMassesToPrimaryMassAndMassRatio(), [30.0, 20.0], [20.0, 30.0]),
        (ComponentMassesToChirpMassAndDelta(), [30.0, 20.0], [-30.0, 20.0]),
        (SourceMassAndRedshiftToDetectedMassAndRedshift(), [30.0, 0.4], [-30.0, 0.4]),
    ],
)
def test_domains_accept_only_valid_inputs(transform, inside, outside):
    assert bool(transform.domain.check(jnp.asarray(inside)))
    assert not bool(transform.domain.check(jnp.asarray(outside)))


@pytest.mark.parametrize(
    "transform, x",
    [
        (PrimaryMassAndMassRatioToComponentMassesTransform(), [30.0, 0.5]),
        (ComponentMassesToPrimaryMassAndMassRatio(), [30.0, 20.0]),
        (ComponentMassesToChirpMassAndSymmetricMassRatio(), [30.0, 20.0]),
        (ComponentMassesToChirpMassAndDelta(), [30.0, 20.0]),
        (ComponentMassesToMassRatioAndSecondaryMass(), [30.0, 20.0]),
        (ComponentMassesToTotalMassAndMassRatio(), [30.0, 20.0]),
        (SourceMassAndRedshiftToDetectedMassAndRedshift(), [30.0, 0.4]),
        (ComponentMassesAndRedshiftToDetectedMassAndRedshift(), [30.0, 20.0, 0.4]),
    ],
)
def test_the_image_of_a_valid_input_lands_in_the_codomain(transform, x):
    x = jnp.asarray(x)
    assert bool(transform.domain.check(x))
    assert bool(transform.codomain.check(transform(x)))


###############################################################################
# BlockTransform
###############################################################################


def _block():
    return BlockTransform(
        AffineTransform(loc=1.0, scale=2.0),
        ExpTransform(),
        event_slices=[(0, 3), (3, 4)],
    )


def test_block_transform_applies_each_transform_to_its_own_slice():
    transform = _block()
    x = jnp.asarray([1.0, 2.0, 3.0, 0.5])
    assert_allclose(transform(x), [3.0, 5.0, 7.0, jnp.exp(0.5)], rtol=1e-12)


def test_block_transform_round_trips():
    transform = _block()
    x = jnp.asarray([1.0, 2.0, 3.0, 0.5])
    assert_allclose(transform.inv(transform(x)), x, rtol=1e-12)


def test_block_transform_log_jacobian_is_the_sum_of_the_blocks():
    transform = _block()
    x = jnp.asarray([1.0, 2.0, 3.0, 0.5])
    y = transform(x)
    expected = jnp.sum(
        AffineTransform(loc=1.0, scale=2.0).log_abs_det_jacobian(x[:3], y[:3])
    ) + jnp.sum(ExpTransform().log_abs_det_jacobian(x[3:], y[3:]))
    assert_allclose(transform.log_abs_det_jacobian(x, y), expected, rtol=1e-12)


@pytest.mark.parametrize(
    "transform, x",
    [
        (_block(), [1.0, 2.0, 3.0, 0.5]),
        (
            BlockTransform(ExpTransform(), ExpTransform(), event_slices=[0, 1]),
            [0.5, 1.5],
        ),
        (
            BlockTransform(
                ExpTransform(), AffineTransform(1.0, 3.0), event_slices=[0, (1, 3)]
            ),
            [0.5, 1.0, 2.0],
        ),
        (
            BlockTransform(
                OrderedTransform(), ExpTransform(), event_slices=[(0, 3), 3]
            ),
            [1.0, 0.5, 0.2, 0.7],
        ),
    ],
)
def test_block_transform_log_jacobian_matches_autodiff(transform, x):
    # covers a sub-transform that reduces its own block (OrderedTransform has an event
    # dimension) alongside elementwise ones that do not
    x = jnp.asarray(x)
    assert_allclose(
        transform.log_abs_det_jacobian(x, transform(x)),
        _autodiff_log_det(transform, x),
        rtol=1e-10,
    )


def test_block_transform_is_batched_over_leading_axes():
    transform = _block()
    x = jnp.asarray([[1.0, 2.0, 3.0, 0.5], [2.0, 3.0, 4.0, 1.5]])
    y = transform(x)
    assert y.shape == x.shape
    assert_allclose(transform.inv(y), x, rtol=1e-12)
    assert_allclose(
        transform.log_abs_det_jacobian(x, y),
        [_autodiff_log_det(transform, row) for row in x],
        rtol=1e-10,
    )


def test_block_transform_mixes_indexed_and_sliced_blocks():
    transform = BlockTransform(
        ExpTransform(), AffineTransform(1.0, 3.0), event_slices=[0, (1, 3)]
    )
    x = jnp.asarray([0.5, 1.0, 2.0])
    assert_allclose(transform(x), [jnp.exp(0.5), 4.0, 7.0], rtol=1e-12)
    assert_allclose(transform.inv(transform(x)), x, rtol=1e-12)


def test_block_transform_accepts_scalar_slices():
    transform = BlockTransform(ExpTransform(), ExpTransform(), event_slices=[0, 1])
    x = jnp.asarray([0.5, 1.5])
    assert_allclose(transform(x), jnp.exp(x), rtol=1e-12)
    assert_allclose(transform.inv(transform(x)), x, rtol=1e-12)


def test_block_transform_domain_and_codomain_are_the_blockwise_conjunction():
    transform = BlockTransform(
        ExpTransform(), AffineTransform(loc=0.0, scale=1.0), event_slices=[0, 1]
    )
    x = jnp.asarray([0.5, -1.0])
    assert bool(transform.domain.check(x))
    assert bool(transform.codomain.check(transform(x)))
    # the codomain of exp excludes non-positive values
    assert not bool(transform.codomain.check(jnp.asarray([-1.0, -1.0])))


def test_block_transform_rejects_a_slice_count_mismatch():
    with pytest.raises(AssertionError, match="Number of event slices must match"):
        BlockTransform(ExpTransform(), event_slices=[(0, 1), (1, 2)])


def test_block_transform_equality():
    assert _block().eq(_block())
    assert not _block().eq(
        BlockTransform(
            AffineTransform(loc=1.0, scale=2.0),
            ExpTransform(),
            event_slices=[(0, 2), (2, 4)],
        )
    )
    assert not _block().eq(ExpTransform())


def test_block_transform_is_a_pytree():
    transform = _block()
    leaves, treedef = jax.tree_util.tree_flatten(transform)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    x = jnp.asarray([1.0, 2.0, 3.0, 0.5])
    assert_allclose(rebuilt(x), transform(x), rtol=1e-12)
