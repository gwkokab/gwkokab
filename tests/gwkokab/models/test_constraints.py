# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.constraints`.

The first half pins what each constraint actually accepts and what
:meth:`feasible_like` hands back; the second half pins their pytree and equality
behaviour.
"""

from collections import namedtuple

import jax
import pytest
from jax import jit, numpy as jnp, vmap
from numpy.testing import assert_allclose
from numpyro.distributions import constraints as npc
from numpyro.distributions.transforms import AffineTransform, ExpTransform

from gwkokab.models.constraints import (
    all_constraint,
    any_constraint,
    decreasing_vector,
    increasing_vector,
    mass_ratio_mass_sandwich,
    mass_sandwich,
    positive_decreasing_vector,
    positive_increasing_vector,
    strictly_decreasing_vector,
    strictly_increasing_vector,
    transform_constraint,
)


###############################################################################
# mass sandwiches
###############################################################################


def test_mass_sandwich_accepts_only_ordered_masses_inside_the_window():
    constraint = mass_sandwich(10.0, 50.0)
    value = jnp.asarray([
        [30.0, 20.0],  # ordered and inside
        [30.0, 60.0],  # secondary above the primary
        [5.0, 4.0],  # both below mmin
        [20.0, 30.0],  # secondary heavier than the primary
        [50.0, 10.0],  # exactly on both edges
        [60.0, 20.0],  # primary above mmax
    ])
    assert_allclose(constraint.check(value), [True, False, False, False, True, False])


def test_mass_sandwich_rejects_everything_when_mmin_is_not_positive():
    # the constraint explicitly requires 0 < mmin
    constraint = mass_sandwich(0.0, 50.0)
    assert not jnp.any(constraint.check(jnp.asarray([[30.0, 20.0], [1.0, 0.5]])))


def test_mass_sandwich_metadata_and_feasible_point():
    constraint = mass_sandwich(10.0, 50.0)
    assert constraint.event_dim == 1
    feasible = constraint.feasible_like(jnp.zeros((3, 2)))
    assert feasible.shape == (3, 2)
    assert_allclose(feasible, 30.0)
    assert jnp.all(constraint.check(feasible))


def test_mass_sandwich_equality():
    assert mass_sandwich(10.0, 50.0) == mass_sandwich(10.0, 50.0)
    assert mass_sandwich(10.0, 50.0) != mass_sandwich(10.0, 60.0)
    assert mass_sandwich(10.0, 50.0) != increasing_vector


def test_mass_ratio_mass_sandwich_accepts_only_reachable_mass_ratios():
    constraint = mass_ratio_mass_sandwich(10.0, 50.0)
    value = jnp.asarray([
        [30.0, 0.5],  # m2 = 15, inside
        [30.0, 0.1],  # m2 = 3, below mmin
        [60.0, 0.9],  # primary above mmax
        [30.0, 1.5],  # secondary heavier than the primary
        [30.0, 1.0],  # equal masses are allowed
    ])
    assert_allclose(constraint.check(value), [True, False, False, False, True])


def test_mass_ratio_mass_sandwich_feasible_point():
    constraint = mass_ratio_mass_sandwich(10.0, 50.0)
    feasible = constraint.feasible_like(jnp.zeros((4, 2)))
    assert feasible.shape == (4, 2)
    assert jnp.all(constraint.check(feasible))


@pytest.mark.parametrize("prototype", [jnp.zeros(()), jnp.zeros(3), jnp.zeros((2, 3))])
def test_mass_ratio_mass_sandwich_feasible_like_checks_its_prototype(prototype):
    with pytest.raises(AssertionError):
        mass_ratio_mass_sandwich(10.0, 50.0).feasible_like(prototype)


def test_mass_ratio_mass_sandwich_equality():
    assert mass_ratio_mass_sandwich(10.0, 50.0) == mass_ratio_mass_sandwich(10.0, 50.0)
    assert mass_ratio_mass_sandwich(10.0, 50.0) != mass_ratio_mass_sandwich(9.0, 50.0)
    assert mass_ratio_mass_sandwich(10.0, 50.0) != mass_sandwich(10.0, 50.0)


###############################################################################
# ordered vectors
###############################################################################


ORDERING_CASES = jnp.asarray([
    [1.0, 2.0, 3.0],  # strictly increasing
    [3.0, 2.0, 1.0],  # strictly decreasing
    [1.0, 1.0, 1.0],  # constant
    [1.0, 3.0, 2.0],  # unordered
    [-1.0, 0.0, 1.0],  # increasing but not positive
])


@pytest.mark.parametrize(
    "constraint, expected",
    [
        (increasing_vector, [True, False, True, False, True]),
        (decreasing_vector, [False, True, True, False, False]),
        (strictly_increasing_vector, [True, False, False, False, True]),
        (strictly_decreasing_vector, [False, True, False, False, False]),
        (positive_increasing_vector, [True, False, True, False, False]),
        (positive_decreasing_vector, [False, True, True, False, False]),
    ],
)
def test_vector_orderings(constraint, expected):
    assert_allclose(constraint.check(ORDERING_CASES), expected)


@pytest.mark.parametrize(
    "constraint",
    [
        increasing_vector,
        decreasing_vector,
        strictly_increasing_vector,
        strictly_decreasing_vector,
        positive_increasing_vector,
        positive_decreasing_vector,
    ],
)
def test_vector_orderings_are_event_shaped(constraint):
    assert constraint.event_dim == 1
    assert constraint.check(jnp.zeros((4, 5, 3))).shape == (4, 5)


@pytest.mark.parametrize(
    "constraint",
    [
        increasing_vector,
        decreasing_vector,
        strictly_increasing_vector,
        strictly_decreasing_vector,
        positive_increasing_vector,
        positive_decreasing_vector,
    ],
)
def test_vector_ordering_feasible_points_are_feasible(constraint):
    feasible = constraint.feasible_like(jnp.zeros(4))
    assert feasible.shape == (4,)
    assert bool(constraint.check(feasible))


@pytest.mark.parametrize(
    "constraint",
    [increasing_vector, decreasing_vector, strictly_increasing_vector],
)
def test_a_single_element_vector_is_trivially_ordered(constraint):
    assert bool(constraint.check(jnp.asarray([5.0])))


###############################################################################
# all_constraint
###############################################################################


def _all():
    return all_constraint(
        [npc.positive, npc.unit_interval, mass_sandwich(10.0, 50.0)], [0, 1, (2, 4)]
    )


def test_all_constraint_requires_every_part_to_hold():
    constraint = _all()
    assert bool(constraint.check(jnp.asarray([1.0, 0.5, 30.0, 20.0])))
    assert not bool(constraint.check(jnp.asarray([-1.0, 0.5, 30.0, 20.0])))
    assert not bool(constraint.check(jnp.asarray([1.0, 1.5, 30.0, 20.0])))
    assert not bool(constraint.check(jnp.asarray([1.0, 0.5, 20.0, 30.0])))


def test_all_constraint_is_batched_over_leading_axes():
    constraint = _all()
    value = jnp.asarray([[1.0, 0.5, 30.0, 20.0], [-1.0, 0.5, 30.0, 20.0]])
    assert_allclose(constraint.check(value), [True, False])


def test_all_constraint_feasible_point_is_feasible():
    constraint = _all()
    feasible = constraint.feasible_like(jnp.zeros(4))
    assert feasible.shape == (4,)
    assert bool(constraint.check(feasible))


def test_all_constraint_rejects_mismatched_slice_counts():
    with pytest.raises(AssertionError, match="must match the number of"):
        all_constraint([npc.positive], [0, 1])


def test_all_constraint_equality():
    assert _all() == _all()
    assert _all() != all_constraint(
        [npc.unit_interval, npc.unit_interval, mass_sandwich(10.0, 50.0)],
        [0, 1, (2, 4)],
    )
    assert _all() != npc.positive


def test_all_constraint_of_a_different_length_is_not_equal():
    # agreeing on a prefix is not enough: the two must have the same number of parts
    assert _all() != all_constraint([npc.positive], [0])
    assert all_constraint([npc.positive], [0]) != _all()


###############################################################################
# any_constraint
###############################################################################


def _any():
    return any_constraint((npc.interval(0.0, 1.0), npc.interval(2.0, 3.0)))


def test_any_constraint_is_the_union_of_its_parts():
    assert_allclose(
        _any().check(jnp.asarray([0.5, 1.5, 2.5, -1.0])),
        [True, False, True, False],
    )


def test_any_constraint_feasible_point_comes_from_the_first_part():
    feasible = _any().feasible_like(jnp.zeros(3))
    assert_allclose(feasible, 0.5)
    assert jnp.all(_any().check(feasible))


def test_a_single_part_any_constraint_is_that_part():
    constraint = any_constraint((npc.interval(0.0, 1.0),))
    value = jnp.asarray([0.5, 1.5])
    assert_allclose(constraint.check(value), npc.interval(0.0, 1.0).check(value))


def test_any_constraint_equality():
    assert _any() == _any()
    assert _any() != any_constraint((npc.interval(0.0, 1.0), npc.positive))
    assert _any() != _all()


def test_any_constraint_of_a_different_length_is_not_equal():
    assert _any() != any_constraint((npc.interval(0.0, 1.0),))
    assert any_constraint((npc.interval(0.0, 1.0),)) != _any()


###############################################################################
# transform_constraint
###############################################################################


def test_transform_constraint_pulls_the_value_back_through_the_chain():
    constraint = transform_constraint(npc.unit_interval, ExpTransform())
    # exp maps [0, 1] onto [1, e]
    assert_allclose(
        constraint.check(jnp.asarray([0.5, 1.5, 3.0])), [False, True, False]
    )


def test_transform_constraint_accepts_a_single_transform_or_a_sequence():
    single = transform_constraint(npc.unit_interval, ExpTransform())
    sequence = transform_constraint(npc.unit_interval, [ExpTransform()])
    assert single.transforms == sequence.transforms
    value = jnp.asarray([0.5, 1.5, 3.0])
    assert_allclose(single.check(value), sequence.check(value))


def test_transform_constraint_composes_a_chain_in_order():
    constraint = transform_constraint(
        npc.unit_interval, [AffineTransform(1.0, 2.0), ExpTransform()]
    )
    # the chain maps [0, 1] to [e, e^3]
    value = jnp.asarray([jnp.e * 0.5, jnp.e * 1.5, jnp.e**4])
    assert_allclose(constraint.check(value), [False, True, False])


def test_transform_constraint_also_enforces_each_codomain():
    constraint = transform_constraint(npc.real, ExpTransform())
    # the codomain of exp is the positive half line
    assert_allclose(constraint.check(jnp.asarray([-1.0, 1.0])), [False, True])


def test_transform_constraint_feasible_point_is_pushed_forward():
    constraint = transform_constraint(npc.unit_interval, ExpTransform())
    feasible = constraint.feasible_like(jnp.zeros(3))
    assert jnp.all(constraint.check(feasible))


def test_transform_constraint_equality():
    first = transform_constraint(npc.unit_interval, ExpTransform())
    assert first == transform_constraint(npc.unit_interval, ExpTransform())
    assert first != transform_constraint(npc.real, ExpTransform())
    assert first != transform_constraint(
        npc.unit_interval, [ExpTransform(), ExpTransform()]
    )
    assert first != npc.unit_interval


###############################################################################
# jax integration
###############################################################################


@pytest.mark.parametrize(
    "constraint, value",
    [
        (mass_sandwich(10.0, 50.0), [30.0, 20.0]),
        (mass_ratio_mass_sandwich(10.0, 50.0), [30.0, 0.5]),
        (increasing_vector, [1.0, 2.0, 3.0]),
        (_all(), [0.5, 0.5, 30.0, 20.0]),
        (_any(), [0.5, 2.5, 1.5]),
        (transform_constraint(npc.unit_interval, ExpTransform()), [0.5, 1.5, 3.0]),
    ],
)
def test_check_works_under_jit(constraint, value):
    value = jnp.asarray(value)
    assert_allclose(jax.jit(constraint.check)(value), constraint.check(value))


###############################################################################
# pytree and equality behaviour
#
# Adapted from the Pyro project (Copyright Contributors to the Pyro project).
###############################################################################


_a = jnp.asarray


class T(namedtuple("TestCase", ["constraint_cls", "params", "kwargs"])):
    pass


SINGLETON_CONSTRAINTS = {
    "decreasing_vector": decreasing_vector,
    "increasing_vector": increasing_vector,
    "positive_decreasing_vector": positive_decreasing_vector,
    "positive_increasing_vector": positive_increasing_vector,
    "strictly_decreasing_vector": strictly_decreasing_vector,
    "strictly_increasing_vector": strictly_increasing_vector,
}
PARAMETRIZED_CONSTRAINTS = {
    "mass_sandwich": T(mass_sandwich, (_a(10.0), _a(30.0)), dict()),
    "mass_ratio_mass_sandwich": T(
        mass_ratio_mass_sandwich, (_a(10.0), _a(30.0)), dict()
    ),
    "all_constraint": T(
        all_constraint,
        (),
        {
            "constraints": [
                decreasing_vector,
                positive_decreasing_vector,
            ],
            "event_slices": [0, 1],
        },
    ),
    "any_constraint": T(
        any_constraint,
        (),
        {
            "constraints": [
                decreasing_vector,
                positive_decreasing_vector,
            ]
        },
    ),
}


@pytest.mark.parametrize(
    "constraint", SINGLETON_CONSTRAINTS.values(), ids=SINGLETON_CONSTRAINTS.keys()
)
def test_singleton_constraint_pytree(constraint):
    # test that singleton constraints objects can be used as pytrees
    def in_cst(constraint, x):
        return x**2

    def out_cst(constraint, x):
        return constraint

    jitted_in_cst = jit(in_cst)
    jitted_out_cst = jit(out_cst)

    assert jitted_in_cst(constraint, 1.0) == 1.0
    assert jitted_out_cst(constraint, 1.0) == constraint

    assert jnp.allclose(
        vmap(in_cst, in_axes=(None, 0), out_axes=0)(constraint, jnp.ones(3)),
        jnp.ones(3),
    )

    assert (
        vmap(out_cst, in_axes=(None, 0), out_axes=None)(constraint, jnp.ones(3))
        is constraint
    )


@pytest.mark.parametrize(
    "cls, cst_args, cst_kwargs",
    PARAMETRIZED_CONSTRAINTS.values(),
    ids=PARAMETRIZED_CONSTRAINTS.keys(),
)
def test_parametrized_constraint_pytree(cls, cst_args, cst_kwargs):
    constraint = cls(*cst_args, **cst_kwargs)

    # test that singleton constraints objects can be used as pytrees
    def in_cst(constraint, x):
        return x**2

    def out_cst(constraint, x):
        return constraint

    jitted_in_cst = jit(in_cst)
    jitted_out_cst = jit(out_cst)

    assert jitted_in_cst(constraint, 1.0) == 1.0
    assert jitted_out_cst(constraint, 1.0) == constraint

    assert jnp.allclose(
        vmap(in_cst, in_axes=(None, 0), out_axes=0)(constraint, jnp.ones(3)),
        jnp.ones(3),
    )

    assert (
        vmap(out_cst, in_axes=(None, 0), out_axes=None)(constraint, jnp.ones(3))
        == constraint
    )

    if len(cst_args) > 0:
        # test creating and manipulating vmapped constraints
        vmapped_cst_args = jax.tree.map(lambda x: x[None], cst_args)

        vmapped_csts = jit(vmap(lambda args: cls(*args, **cst_kwargs), in_axes=(0,)))(
            vmapped_cst_args
        )
        assert vmap(lambda x: x.eq(constraint), in_axes=0)(vmapped_csts).all()

        twice_vmapped_cst_args = jax.tree.map(lambda x: x[None], vmapped_cst_args)

        vmapped_csts = jit(
            vmap(
                vmap(lambda args: cls(*args, **cst_kwargs), in_axes=(0,)),
                in_axes=(0,),
            ),
        )(twice_vmapped_cst_args)
        assert vmap(vmap(lambda x: x.eq(constraint), in_axes=0), in_axes=0)(
            vmapped_csts
        ).all()


@pytest.mark.parametrize(
    "constraint", SINGLETON_CONSTRAINTS.values(), ids=SINGLETON_CONSTRAINTS.keys()
)
def test_singleton_constraint_eq(constraint):
    assert constraint == constraint
    assert constraint != 1

    # check that equality checks are robust to constraints parametrized
    # by abstract values
    @jit
    def check_constraints(c1, c2):
        return c1.eq(c2)

    assert check_constraints(constraint, constraint)


@pytest.mark.parametrize(
    "cls, cst_args, cst_kwargs",
    PARAMETRIZED_CONSTRAINTS.values(),
    ids=PARAMETRIZED_CONSTRAINTS.keys(),
)
def test_parametrized_constraint_eq(cls, cst_args, cst_kwargs):
    constraint = cls(*cst_args, **cst_kwargs)
    constraint2 = cls(*cst_args, **cst_kwargs)
    assert constraint == constraint2
    assert constraint != 1

    # check that equality checks are robust to constraints parametrized
    # by abstract values
    @jit
    def check_constraints(c1, c2):
        return c1.eq(c2)

    assert check_constraints(constraint, constraint2)
