# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Beta-distribution concentration guard.

``check_min_concentration_for_beta_dist`` is an algebraic rearrangement: it answers
"does this (mean, variance) pair correspond to a Beta whose concentrations both clear
their floors?" without ever forming the concentrations. The tests therefore form them
the direct way and check the two agree.
"""

import chex
import numpy as np
import pytest
from jax import numpy as jnp

from gwkokab.analysis.utils.checks import check_min_concentration_for_beta_dist


def _concentrations(loc: np.ndarray, var: np.ndarray):
    r""":math:`(\alpha, \beta)` of the Beta distribution with mean `loc` and variance
    `var`.
    """
    nu = loc * (1.0 - loc) / var - 1.0
    return loc * nu, (1.0 - loc) * nu


@pytest.mark.parametrize("alpha_min", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("beta_min", [1.0, 0.5, 2.0])
def test_matches_the_direct_concentrations(alpha_min, beta_min):
    """The inequality must agree with ``alpha > alpha_min and beta > beta_min``."""
    loc, var = np.meshgrid(
        np.linspace(0.05, 0.95, 19), np.linspace(1e-4, 0.2, 23), indexing="ij"
    )
    alpha, beta = _concentrations(loc, var)
    expected = (alpha > alpha_min) & (beta > beta_min)

    valid = check_min_concentration_for_beta_dist(
        jnp.asarray(loc), jnp.asarray(var), alpha_min=alpha_min, beta_min=beta_min
    )

    chex.assert_trees_all_equal(np.asarray(valid), expected)


def test_defaults_are_unit_floors():
    """The default call is the ``alpha_min = beta_min = 1`` case, i.e. unimodality."""
    loc = jnp.linspace(0.1, 0.9, 9)
    var = jnp.full_like(loc, 0.01)

    chex.assert_trees_all_equal(
        check_min_concentration_for_beta_dist(loc, var),
        check_min_concentration_for_beta_dist(loc, var, alpha_min=1.0, beta_min=1.0),
    )


def test_large_variance_is_rejected():
    """A variance at or above the Bernoulli bound cannot come from any Beta."""
    loc = jnp.asarray([0.2, 0.5, 0.8])
    var = loc * (1.0 - loc)

    valid = check_min_concentration_for_beta_dist(loc, var)

    assert not jnp.any(valid)


def test_small_variance_is_accepted():
    """Shrinking the variance sends both concentrations to infinity, so all pass."""
    loc = jnp.linspace(0.05, 0.95, 19)
    var = jnp.full_like(loc, 1e-8)

    assert jnp.all(check_min_concentration_for_beta_dist(loc, var))


def test_raising_the_floors_can_only_reject_more():
    """The accepted set is monotone decreasing in ``alpha_min`` and ``beta_min``."""
    loc = jnp.linspace(0.05, 0.95, 19)[:, None]
    var = jnp.linspace(1e-4, 0.2, 23)[None, :]

    lenient = check_min_concentration_for_beta_dist(
        loc, var, alpha_min=0.5, beta_min=0.5
    )
    strict = check_min_concentration_for_beta_dist(
        loc, var, alpha_min=3.0, beta_min=3.0
    )

    assert jnp.all(jnp.logical_or(lenient, jnp.logical_not(strict)))
    assert jnp.any(lenient) and not jnp.all(strict)


def test_broadcasts_over_per_element_floors():
    """The floors may themselves be arrays, one per element."""
    loc = jnp.asarray([0.5, 0.5])
    var = jnp.asarray([0.02, 0.02])
    alpha_min = jnp.asarray([1.0, 100.0])

    valid = check_min_concentration_for_beta_dist(loc, var, alpha_min=alpha_min)

    chex.assert_trees_all_equal(np.asarray(valid), np.asarray([True, False]))


def test_shape_is_preserved():
    loc = jnp.full((4, 3), 0.4)
    var = jnp.full((4, 3), 0.01)

    chex.assert_shape(check_min_concentration_for_beta_dist(loc, var), (4, 3))
