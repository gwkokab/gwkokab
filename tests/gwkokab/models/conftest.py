# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Sequence

import jax
import pytest
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import Distribution


@pytest.fixture
def pytree_roundtrip() -> Callable[[Distribution], Distribution]:
    """Flatten a distribution to its leaves and rebuild it.

    NumPyro distributions are registered pytrees, so every model must survive this
    round trip unchanged: it is exactly what happens when a model crosses a
    :func:`jax.jit` or :func:`jax.vmap` boundary.
    """

    def _roundtrip(distribution: Distribution) -> Distribution:
        leaves, treedef = jax.tree_util.tree_flatten(distribution)
        return jax.tree_util.tree_unflatten(treedef, leaves)

    return _roundtrip


@pytest.fixture
def trapz_1d() -> Callable[[Distribution, float, float, int], Array]:
    r"""Integrate :math:`\exp(\log p)` of a scalar-event distribution over an
    interval.
    """

    def _integrate(
        distribution: Distribution, low: float, high: float, num: int = 20_001
    ) -> Array:
        x = jnp.linspace(low, high, num)
        return jnp.trapezoid(jnp.exp(distribution.log_prob(x)), x)

    return _integrate


@pytest.fixture
def trapz_nd() -> Callable[..., Array]:
    r"""Integrate :math:`\exp(\log p)` of a vector-event distribution over a box.

    Values outside the support are masked out first: the models deliberately return the
    unmasked functional form there, so a naive quadrature over the bounding box would
    pick up density that the support forbids.
    """

    def _integrate(
        distribution: Distribution,
        bounds: Sequence[tuple[float, float]],
        num: int = 401,
    ) -> Array:
        axes = [jnp.linspace(low, high, num) for low, high in bounds]
        grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        log_prob = jnp.where(
            distribution.support.check(grid), distribution.log_prob(grid), -jnp.inf
        )
        integral = jnp.nan_to_num(jnp.exp(log_prob), nan=0.0, posinf=0.0)
        for axis in reversed(axes):
            integral = jnp.trapezoid(integral, axis, axis=-1)
        return integral

    return _integrate
