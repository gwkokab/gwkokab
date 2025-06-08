# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""Provides implementation of various constraints using
:class:`~numpyro.distributions.constraints.Constraint`.
"""

from collections.abc import Sequence

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions.constraints import (
    _SingletonConstraint,
    Constraint,
    independent,
    positive,
)


__all__ = [
    "decreasing_vector",
    "increasing_vector",
    "mass_ratio_mass_sandwich",
    "mass_sandwich",
    "positive_decreasing_vector",
    "positive_increasing_vector",
    "strictly_decreasing_vector",
    "strictly_increasing_vector",
    "all_constraint",
    "any_constraint",
]


class _MassSandwichConstraint(Constraint):
    r"""Constrain mass values to lie within a sandwiched interval.

    .. math::
        m_{\text{min}} \leq m_2 \leq m_1 \leq m_{\text{max}}
    """

    event_dim = 1

    def __init__(self, mmin: float, mmax: float):
        """
        Parameters
        ----------
        mmin : float
            Minimum mass.
        mmax : float
            Maximum mass.
        """
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, x: Array) -> Array:
        m1, m2 = jnp.unstack(x, axis=-1)
        mask = jnp.logical_and(jnp.less(0.0, self.mmin), jnp.less_equal(self.mmin, m2))
        mask = jnp.logical_and(mask, jnp.less_equal(m2, m1))
        mask = jnp.logical_and(mask, jnp.less_equal(m1, self.mmax))
        return jnp.asarray(mask, dtype=bool)

    def feasible_like(self, prototype: Array) -> Array:
        return jnp.full(prototype.shape[:-1], (self.mmin + self.mmax) * 0.5)

    def tree_flatten(self):
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MassSandwichConstraint):
            return False
        return jnp.array_equal(self.mmin, other.mmin) & jnp.array_equal(
            self.mmax, other.mmax
        )


class _MassRatioMassSandwichConstraint(Constraint):
    r"""Constrain primary mass to lie within a sandwiched interval and the mass ratio to
    lie within a given interval. This is a transformed version of the
    :class:`_MassSandwichConstraint`.

    .. math::
        \begin{align*}
            m_{\text{min}}             & \leq m_1 \leq m_{\max} \\
            \frac{m_{\text{min}}}{m_1} & \leq q   \leq 1
        \end{align*}
    """

    event_dim = 1

    def __init__(self, mmin: float, mmax: float):
        """
        Parameters
        ----------
        mmin : float
            Minimum mass.
        mmax : float
            Maximum mass.
        """
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, x: Array) -> Array:
        m1, q = jnp.unstack(x, axis=-1)
        m2 = jnp.multiply(m1, q)
        mask = jnp.logical_and(jnp.less(0.0, self.mmin), jnp.less_equal(self.mmin, m2))
        mask = jnp.logical_and(mask, jnp.less_equal(m2, m1))
        mask = jnp.logical_and(mask, jnp.less_equal(m1, self.mmax))
        return jnp.asarray(mask, dtype=bool)

    def feasible_like(self, prototype: Array) -> Array:
        shape = prototype.shape[:-1]
        m1 = (self.mmin + self.mmax) * 0.5
        q = jnp.clip(self.mmin / m1, 0.0, 1.0)
        m1 = jnp.broadcast_to(m1, shape)
        q = jnp.broadcast_to(q, shape)
        return jnp.stack((m1, q), axis=-1)

    def tree_flatten(self):
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MassRatioMassSandwichConstraint):
            return False
        return jnp.array_equal(self.mmin, other.mmin) & jnp.array_equal(
            self.mmax, other.mmax
        )


class _IncreasingVector(_SingletonConstraint):
    r"""Constrain values to be increasing, i.e. :math:`\forall i<j,x_i\leq x_j`."""

    event_dim = 1

    def __call__(self, x):
        return jnp.all(x[..., 1:] >= x[..., :-1], axis=-1)

    def feasible_like(self, prototype):
        return jnp.ones(prototype.shape, dtype=prototype.dtype)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _IncreasingVector)


class _DecreasingVector(_SingletonConstraint):
    r"""Constrain values to be decreasing, i.e. :math:`\forall i<j, x_i \geq x_j`."""

    event_dim = 1

    def __call__(self, x):
        return jnp.all(x[..., 1:] <= x[..., :-1], axis=-1)

    def feasible_like(self, prototype):
        return jnp.ones(prototype.shape, dtype=prototype.dtype)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _DecreasingVector)


class _StrictlyIncreasingVector(_SingletonConstraint):
    r"""Constrain values to be strictly increasing, i.e. :math:`\forall i<j, x_i <
    x_j`.
    """

    event_dim = 1

    def __call__(self, x):
        return jnp.all(x[..., 1:] > x[..., :-1], axis=-1)

    def feasible_like(self, prototype):
        return jnp.ones(prototype.shape, dtype=prototype.dtype) * jnp.arange(
            1, prototype.shape[-1] + 1, dtype=prototype.dtype
        )

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _StrictlyIncreasingVector)


class _StrictlyDecreasingVector(_SingletonConstraint):
    r"""Constrain values to be strictly decreasing, i.e. :math:`\forall i<j,x_i >
    x_j`.
    """

    event_dim = 1

    def __call__(self, x):
        return jnp.all(x[..., 1:] < x[..., :-1], axis=-1)

    def feasible_like(self, prototype):
        return jnp.ones(prototype.shape, dtype=prototype.dtype) * jnp.arange(
            prototype.shape[-1], 0, -1, dtype=prototype.dtype
        )

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _StrictlyDecreasingVector)


class _PositiveIncreasingVector(_SingletonConstraint):
    r"""Constrain values to be positive and increasing, i.e. :math:`\forall i<j, x_i
    \leq x_j`.
    """

    event_dim = 1

    def __call__(self, x):
        return increasing_vector.check(x) & independent(positive, 1).check(x)

    def feasible_like(self, prototype):
        return jnp.ones(prototype.shape, dtype=prototype.dtype)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _PositiveIncreasingVector)


class _PositiveDecreasingVector(_SingletonConstraint):
    r"""Constrain values to be positive and decreasing, i.e. :math:`\forall i<j, x_i
    \geq x_j`.
    """

    event_dim = 1

    def __call__(self, x):
        return decreasing_vector.check(x) & independent(positive, 1).check(x)

    def feasible_like(self, prototype):
        return jnp.ones(prototype.shape, dtype=prototype.dtype)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _PositiveDecreasingVector)


class _AllConstraint(Constraint):
    r"""Constrain values to satisfy multiple constraints."""

    def __init__(
        self, constraints: Sequence[Constraint], event_slices: Sequence[int | slice]
    ):
        assert len(constraints) == len(event_slices), (
            f"Number of constraints ({len(constraints)}) must match the number of "
            f"event slices ({len(event_slices)})"
        )
        self.constraints = constraints
        self.event_slices = event_slices

    def __call__(self, x):
        mask = self.constraints[0].check(x[..., self.event_slices[0]])
        for constraint, event_slice in zip(self.constraints[1:], self.event_slices[1:]):
            mask &= constraint.check(x[..., event_slice])
        return mask

    def feasible_like(self, prototype: Array) -> Array:
        feasible_values = []
        for constraint, event_slice in zip(self.constraints, self.event_slices):
            feasible_values.append(
                constraint.feasible_like(prototype[..., event_slice])
            )
        max_ndim = max([feasible_value.ndim for feasible_value in feasible_values])
        feasible_values = [
            jnp.expand_dims(
                feasible_value, axis=tuple(range(feasible_value.ndim, max_ndim))
            )
            if feasible_value.ndim < max_ndim
            else feasible_value
            for feasible_value in feasible_values
        ]
        feasible_value = jnp.concatenate(feasible_values, axis=-1)
        return feasible_value

    def tree_flatten(self):
        return (self.constraints, self.event_slices), (
            ("constraints", "event_slices"),
            dict(),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _AllConstraint):
            return False
        return all(
            constraint == other_constraint
            for constraint, other_constraint in zip(self.constraints, other.constraints)
        )


class _AnyConstraint(Constraint):
    r"""Constrain values to satisfy at least one of the constraints."""

    def __init__(self, constraints: Sequence[Constraint]):
        self.constraints = constraints

    def __call__(self, x):
        mask = self.constraints[0].check(x)
        for constraint in self.constraints[1:]:
            mask |= constraint.check(x)
        return mask

    def feasible_like(self, prototype: Array) -> Array:
        return self.constraints[0].feasible_like(prototype)

    def tree_flatten(self):
        return (self.constraints,), (("constraints",), dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _AnyConstraint):
            return False
        return all(
            constraint == other_constraint
            for constraint, other_constraint in zip(self.constraints, other.constraints)
        )


mass_sandwich = _MassSandwichConstraint
mass_ratio_mass_sandwich = _MassRatioMassSandwichConstraint
increasing_vector = _IncreasingVector()
decreasing_vector = _DecreasingVector()
strictly_increasing_vector = _StrictlyIncreasingVector()
strictly_decreasing_vector = _StrictlyDecreasingVector()
positive_increasing_vector = _PositiveIncreasingVector()
positive_decreasing_vector = _PositiveDecreasingVector()
all_constraint = _AllConstraint
any_constraint = _AnyConstraint
