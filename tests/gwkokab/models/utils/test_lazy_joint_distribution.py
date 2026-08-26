# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.utils._lazyjointdistribution`."""

import jax
import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose
from numpyro.distributions import (
    Exponential,
    HalfNormal,
    MultivariateNormal,
    Normal,
    Uniform,
)

from gwkokab.models.utils import LazyJointDistribution
from gwkokab.models.utils._lazyjointdistribution import _LazyConstraint
from gwkokab.utils.exceptions import LoggedUserWarning, LoggedValueError


def _normal_given_loc():
    return jax.tree_util.Partial(lambda loc: Normal(loc, 1.0))


def _normal_given_loc_and_scale():
    return jax.tree_util.Partial(lambda loc, scale: Normal(loc, scale))


def _simple():
    """``x0 ~ N(0, 1)`` and ``x1 | x0 ~ N(x0, 1)``."""
    return LazyJointDistribution(
        Normal(0.0, 1.0),
        _normal_given_loc(),
        dependencies={1: {"loc": 0}},
        partial_order=[1],
    )


###############################################################################
# construction
###############################################################################


def test_shapes_and_layout():
    model = _simple()
    assert model.batch_shape == ()
    assert model.event_shape == (2,)
    assert model.shaped_values == (0, 1)
    assert model.partial_order == [1]
    assert model.dependencies == {1: {"loc": 0}}


def test_event_slices_account_for_vector_marginals():
    lazy = jax.tree_util.Partial(
        lambda loc: MultivariateNormal(jnp.stack([loc, loc], axis=-1), jnp.eye(2))
    )
    model = LazyJointDistribution(
        Normal(0.0, 1.0),
        lazy,
        dependencies={1: {"loc": 0}},
        partial_order=[1],
        dependencies_event_shape=[(), (2,)],
    )
    assert model.event_shape == (3,)
    assert model.shaped_values == (0, (1, 3))


def test_several_independent_marginals_and_one_lazy_one():
    model = LazyJointDistribution(
        Normal(0.0, 1.0),
        Uniform(0.0, 1.0),
        _normal_given_loc_and_scale(),
        dependencies={2: {"loc": 0, "scale": 1}},
        partial_order=[2],
    )
    assert model.event_shape == (3,)
    value = jnp.asarray([0.5, 0.25, 1.5])
    expected = (
        Normal(0.0, 1.0).log_prob(0.5)
        + Uniform(0.0, 1.0).log_prob(0.25)
        + Normal(0.5, 0.25).log_prob(1.5)
    )
    assert_allclose(model.log_prob(value), expected, rtol=1e-12)


def test_a_chain_of_lazy_marginals():
    model = LazyJointDistribution(
        Normal(0.0, 1.0),
        _normal_given_loc(),
        _normal_given_loc(),
        dependencies={1: {"loc": 0}, 2: {"loc": 1}},
        partial_order=[1, 2],
    )
    value = jnp.asarray([0.5, 1.5, -0.5])
    expected = (
        Normal(0.0, 1.0).log_prob(0.5)
        + Normal(0.5, 1.0).log_prob(1.5)
        + Normal(1.5, 1.0).log_prob(-0.5)
    )
    assert_allclose(model.log_prob(value), expected, rtol=1e-12)
    assert model.sample(jrd.key(0), (7,)).shape == (7, 3)


###############################################################################
# construction errors
###############################################################################


def test_rejects_no_marginals():
    with pytest.raises(LoggedValueError, match="At least one marginal distribution"):
        LazyJointDistribution(dependencies={}, partial_order=[])


def test_rejects_an_empty_partial_order():
    with pytest.raises(LoggedValueError, match="`partial_order` must be provided"):
        LazyJointDistribution(Normal(0.0, 1.0), dependencies={}, partial_order=[])


def test_rejects_missing_dependencies():
    with pytest.raises(LoggedValueError, match="`dependencies` must be provided"):
        LazyJointDistribution(
            Normal(0.0, 1.0),
            _normal_given_loc(),
            dependencies=None,
            partial_order=[1],
        )


def test_rejects_a_partial_order_that_does_not_match_the_dependencies():
    with pytest.raises(LoggedValueError, match="must have the same length"):
        LazyJointDistribution(
            Normal(0.0, 1.0),
            _normal_given_loc(),
            _normal_given_loc(),
            dependencies={1: {"loc": 0}},
            partial_order=[1, 2],
        )


def test_rejects_a_marginal_that_is_neither_a_distribution_nor_a_partial():
    with pytest.raises(LoggedValueError, match="All marginals must be instances of"):
        LazyJointDistribution(
            Normal(0.0, 1.0),
            "not a distribution",
            dependencies={1: {"loc": 0}},
            partial_order=[1],
        )


def test_flatten_method_is_ignored_with_a_warning():
    with pytest.warns(LoggedUserWarning, match="not used"):
        LazyJointDistribution(
            Normal(0.0, 1.0),
            _normal_given_loc(),
            dependencies={1: {"loc": 0}},
            partial_order=[1],
            flatten_method="deep",
        )


###############################################################################
# the density
###############################################################################


def test_log_prob_factorises_along_the_dependency_graph():
    model = _simple()
    value = jnp.asarray([0.5, 1.5])
    expected = Normal(0.0, 1.0).log_prob(0.5) + Normal(0.5, 1.0).log_prob(1.5)
    assert_allclose(model.log_prob(value), expected, rtol=1e-12)


@pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
def test_log_prob_shape(sample_shape):
    model = _simple()
    value = jnp.broadcast_to(jnp.asarray([0.5, 1.5]), sample_shape + (2,))
    assert model.log_prob(value).shape == sample_shape


def test_log_prob_of_a_vector_valued_lazy_marginal():
    lazy = jax.tree_util.Partial(
        lambda loc: MultivariateNormal(jnp.stack([loc, loc], axis=-1), jnp.eye(2))
    )
    model = LazyJointDistribution(
        Normal(0.0, 1.0),
        lazy,
        dependencies={1: {"loc": 0}},
        partial_order=[1],
        dependencies_event_shape=[(), (2,)],
    )
    value = jnp.asarray([0.5, 1.0, 2.0])
    expected = Normal(0.0, 1.0).log_prob(0.5) + MultivariateNormal(
        jnp.asarray([0.5, 0.5]), jnp.eye(2)
    ).log_prob(jnp.asarray([1.0, 2.0]))
    assert_allclose(model.log_prob(value), expected, rtol=1e-12)


def test_under_jit_and_grad():
    model = _simple()
    value = jnp.asarray([0.5, 1.5])
    assert_allclose(jax.jit(model.log_prob)(value), model.log_prob(value), rtol=1e-12)
    # d/dx0 [log N(x0 | 0, 1) + log N(x1 | x0, 1)] = -x0 + (x1 - x0)
    gradient = jax.grad(model.log_prob)(value)
    assert_allclose(gradient, [-0.5 + 1.0, -1.0], rtol=1e-12)


###############################################################################
# the support
###############################################################################


def test_the_support_is_the_intersection_of_the_marginal_supports():
    model = LazyJointDistribution(
        HalfNormal(1.0),
        jax.tree_util.Partial(lambda low: Uniform(low, low + 1.0)),
        dependencies={1: {"low": 0}},
        partial_order=[1],
    )
    assert isinstance(model.support, _LazyConstraint)
    assert_allclose(
        model.support.check(jnp.asarray([[1.0, 1.5], [1.0, 3.0], [-1.0, 0.5]])),
        [True, False, False],
    )


def test_an_explicit_support_is_used_verbatim():
    from numpyro.distributions import constraints

    model = LazyJointDistribution(
        Normal(0.0, 1.0),
        _normal_given_loc(),
        dependencies={1: {"loc": 0}},
        partial_order=[1],
        support=constraints.real_vector,
    )
    assert model.support is constraints.real_vector


def test_the_lazy_constraint_compares_its_parts():
    marginals = (Normal(0.0, 1.0), _normal_given_loc())
    first = _LazyConstraint(
        *marginals, dependencies={1: {"loc": 0}}, event_slices=(0, 1)
    )
    second = _LazyConstraint(
        *marginals, dependencies={1: {"loc": 0}}, event_slices=(0, 1)
    )
    different = _LazyConstraint(
        *marginals, dependencies={1: {"loc": 0}}, event_slices=(1, 0)
    )
    assert first == second
    assert first != different
    assert first != "not a constraint"


def test_lazy_constraints_of_a_different_length_are_not_equal():
    # a length mismatch must fall out as inequality, not as an error from the zip
    marginals = (Normal(0.0, 1.0), _normal_given_loc())
    longer = _LazyConstraint(
        *marginals, dependencies={1: {"loc": 0}}, event_slices=(0, 1)
    )
    shorter = _LazyConstraint(
        marginals[0], dependencies={1: {"loc": 0}}, event_slices=(0,)
    )
    assert longer != shorter
    assert shorter != longer


def test_two_independently_built_lazy_constraints_are_not_equal():
    # marginals are compared by identity, so structurally identical constraints built
    # from freshly constructed distributions still differ
    assert _simple().support != _simple().support


def test_the_lazy_constraint_is_a_pytree():
    constraint = _simple().support
    leaves, treedef = jax.tree_util.tree_flatten(constraint)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    value = jnp.asarray([0.5, 1.5])
    assert_allclose(rebuilt.check(value), constraint.check(value))


def test_validate_args_warns_outside_the_support():
    model = LazyJointDistribution(
        HalfNormal(1.0),
        jax.tree_util.Partial(lambda low: Uniform(low, low + 1.0)),
        dependencies={1: {"low": 0}},
        partial_order=[1],
        validate_args=True,
    )
    with pytest.warns(UserWarning, match="Out-of-support values"):
        model.log_prob(jnp.asarray([1.0, 5.0]))


###############################################################################
# sampling
###############################################################################


@pytest.mark.parametrize("sample_shape", [(), (1,), (7,)])
def test_sample_shape(sample_shape):
    model = _simple()
    assert model.sample(jrd.key(0), sample_shape).shape == sample_shape + (2,)


def test_samples_respect_the_conditional_structure():
    model = _simple()
    samples = model.sample(jrd.key(0), (100_000,))
    # x1 - x0 is a standard normal independent of x0
    residual = samples[..., 1] - samples[..., 0]
    assert_allclose(float(jnp.mean(samples[..., 0])), 0.0, atol=2e-2)
    assert_allclose(float(jnp.std(samples[..., 0])), 1.0, atol=2e-2)
    assert_allclose(float(jnp.mean(residual)), 0.0, atol=2e-2)
    assert_allclose(float(jnp.std(residual)), 1.0, atol=2e-2)


def test_independent_marginals_are_sampled_too():
    model = LazyJointDistribution(
        Exponential(2.0),
        Uniform(0.0, 1.0),
        _normal_given_loc(),
        dependencies={2: {"loc": 0}},
        partial_order=[2],
    )
    samples = model.sample(jrd.key(1), (10_000,))
    assert samples.shape == (10_000, 3)
    assert jnp.all(samples[..., 0] > 0.0)
    assert jnp.all((samples[..., 1] >= 0.0) & (samples[..., 1] <= 1.0))


def test_sample_rejects_a_plain_integer_key():
    with pytest.raises(AssertionError):
        _simple().sample(0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LazyJointDistribution.sample slices sample_shape by the marginal's event_dim, "
        "so a multi-dimensional sample_shape draws the wrong number of variates"
    ),
)
def test_multi_dimensional_sample_shape():
    model = _simple()
    assert model.sample(jrd.key(0), (2, 3)).shape == (2, 3, 2)
