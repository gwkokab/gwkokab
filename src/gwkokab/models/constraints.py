# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
r"""Provides implementation of various constraints using
:class:`~numpyro.distributions.constraints.Constraint`.

Two families live here. The *ordering* constraints (increasing, decreasing, their
strict and positive variants) apply to a whole vector and are what keep an ordered
set of hyper-parameters -- successive Gaussian means, the break points of a broken
power law -- from swapping places during sampling. The *mass* constraints encode the
sandwich :math:`m_\min \leq m_2 \leq m_1 \leq m_\max` in either mass coordinates.

:data:`all_constraint` and :data:`any_constraint` combine constraints, the former
slice-wise across the event axis, which is how
:class:`~gwkokab.models.utils.JointDistribution` derives its support from those of
its marginals. :data:`transform_constraint` pushes a support through a chain of
transforms, which is what
:class:`~gwkokab.models.utils.ExtendedSupportTransformedDistribution` uses.

Public names are lower-case aliases of the private classes: the parameterised ones
(:data:`mass_sandwich`, :data:`all_constraint`, ...) alias the class itself and must
be called, while the parameter-free ones (:data:`increasing_vector`, ...) alias a
singleton instance.
"""

from collections.abc import Sequence
from typing import Tuple, Union

from jax import lax, numpy as jnp
from jaxtyping import Array
from numpyro.distributions.constraints import (
    _SingletonConstraint,
    Constraint,
    independent,
    positive,
)
from numpyro.distributions.transforms import Transform


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

    Expects the last axis of the checked value to be :math:`(m_1, m_2)`.

    Parameters
    ----------
    mmin : float
        Minimum mass :math:`m_\min`.
    mmax : float
        Maximum mass :math:`m_\max`.
    """

    event_dim = 1

    def __init__(self, mmin: float, mmax: float):
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, x: Array) -> Array:
        r"""Check the sandwich constraint.

        Parameters
        ----------
        x : Array
            Values whose last axis is :math:`(m_1, m_2)`.

        Returns
        -------
        Array
            Boolean mask, true where
            :math:`0 < m_\min \leq m_2 \leq m_1 \leq m_\max`.
        """
        m1, m2 = jnp.unstack(x, axis=-1)
        mask = jnp.logical_and(jnp.less(0.0, self.mmin), jnp.less_equal(self.mmin, m2))
        mask = jnp.logical_and(mask, jnp.less_equal(m2, m1))
        mask = jnp.logical_and(mask, jnp.less_equal(m1, self.mmax))
        return jnp.asarray(mask, dtype=bool)

    def feasible_like(self, prototype: Array) -> Array:
        r"""Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            An array filled with the midpoint
            :math:`(m_\min + m_\max)/2`, which satisfies the sandwich with
            :math:`m_1 = m_2`.
        """
        return jnp.full(prototype.shape, (self.mmin + self.mmax) * 0.5)

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        Returns
        -------
        tuple
            The children and the auxiliary metadata, in the layout NumPyro's constraint
            registry expects.
        """
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())

    def eq(self, other: object, static: bool = False) -> bool:
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is a :class:`_MassSandwichConstraint` with the same bounds.
        """
        if not isinstance(other, _MassSandwichConstraint):
            return False
        return jnp.array_equal(self.mmin, other.mmin) & jnp.array_equal(
            self.mmax, other.mmax
        )


class _MassRatioMassSandwichConstraint(Constraint):
    r"""Constrain the primary mass and mass ratio to a sandwiched region.

    This is a transformed version of the :class:`_MassSandwichConstraint`, expressed in
    :math:`(m_1, q)` coordinates.

    .. math::
        \begin{align*}
            m_{\text{min}}             & \leq m_1 \leq m_{\max} \\
            \frac{m_{\text{min}}}{m_1} & \leq q   \leq 1
        \end{align*}

    Expects the last axis of the checked value to be :math:`(m_1, q)`.

    Parameters
    ----------
    mmin : float
        Minimum mass :math:`m_\min`.
    mmax : float
        Maximum mass :math:`m_\max`.
    """

    event_dim = 1

    def __init__(self, mmin: float, mmax: float):
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, x: Array) -> Array:
        r"""Check the sandwich constraint in mass-ratio coordinates.

        Parameters
        ----------
        x : Array
            Values whose last axis is :math:`(m_1, q)`.

        Returns
        -------
        Array
            Boolean mask, true where :math:`m_2 = m_1 q` satisfies
            :math:`0 < m_\min \leq m_2 \leq m_1 \leq m_\max`.
        """
        m1, q = jnp.unstack(x, axis=-1)
        m2 = jnp.multiply(m1, q)
        mask = jnp.logical_and(jnp.less(0.0, self.mmin), jnp.less_equal(self.mmin, m2))
        mask = jnp.logical_and(mask, jnp.less_equal(m2, m1))
        mask = jnp.logical_and(mask, jnp.less_equal(m1, self.mmax))
        return jnp.asarray(mask, dtype=bool)

    def feasible_like(self, prototype: Array) -> Array:
        r"""Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose last axis has size 2,
            holding :math:`(m_1, q)`.

        Returns
        -------
        Array
            An array with :math:`m_1` at the midpoint :math:`(m_\min + m_\max)/2` and
            :math:`q = m_\min/m_1`, the smallest admissible ratio.

        Raises
        ------
        AssertionError
            If ``prototype`` is scalar or its last axis is not of size 2.
        """
        assert prototype.ndim >= 1, "Prototype must have at least one dimension."
        assert prototype.shape[-1] == 2, (
            "Prototype must have last dimension of size 2 for mass and mass ratio."
        )
        shape = prototype.shape[:-1]
        m1 = (self.mmin + self.mmax) * 0.5
        q = jnp.clip(self.mmin / m1, 0.0, 1.0)
        m1 = jnp.broadcast_to(m1, shape)
        q = jnp.broadcast_to(q, shape)
        return jnp.stack((m1, q), axis=-1)

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        Returns
        -------
        tuple
            The children and the auxiliary metadata, in the layout NumPyro's constraint
            registry expects.
        """
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())

    def eq(self, other: object, static: bool = False) -> bool:
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is a :class:`_MassRatioMassSandwichConstraint` with the same bounds.
        """
        if not isinstance(other, _MassRatioMassSandwichConstraint):
            return False
        return jnp.array_equal(self.mmin, other.mmin) & jnp.array_equal(
            self.mmax, other.mmax
        )


class _IncreasingVector(_SingletonConstraint):
    r"""Constrain values to be increasing, i.e. :math:`\forall i<j,x_i\leq x_j`."""

    event_dim = 1

    def __call__(self, x):
        """Check that the last axis is increasing.

        Parameters
        ----------
        x : Array
            Values to check; the last axis is the vector.

        Returns
        -------
        Array
            Boolean mask of shape ``x.shape[:-1]``.
        """
        return jnp.all(x[..., 1:] >= x[..., :-1], axis=-1)

    def feasible_like(self, prototype):
        """Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            An array of ones, which is (weakly) increasing.
        """
        return jnp.ones(prototype.shape, dtype=prototype.dtype)

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        The constraint is a stateless
        :class:`~numpyro.distributions.constraints._SingletonConstraint`, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`_IncreasingVector`.
        """
        return isinstance(other, _IncreasingVector)


class _DecreasingVector(_SingletonConstraint):
    r"""Constrain values to be decreasing, i.e. :math:`\forall i<j, x_i \geq x_j`."""

    event_dim = 1

    def __call__(self, x):
        """Check that the last axis is decreasing.

        Parameters
        ----------
        x : Array
            Values to check; the last axis is the vector.

        Returns
        -------
        Array
            Boolean mask of shape ``x.shape[:-1]``.
        """
        return jnp.all(x[..., 1:] <= x[..., :-1], axis=-1)

    def feasible_like(self, prototype):
        """Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            An array of ones, which is (weakly) decreasing.
        """
        return jnp.ones(prototype.shape, dtype=prototype.dtype)

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        The constraint is a stateless
        :class:`~numpyro.distributions.constraints._SingletonConstraint`, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`_DecreasingVector`.
        """
        return isinstance(other, _DecreasingVector)


class _StrictlyIncreasingVector(_SingletonConstraint):
    r"""Constrain values to be strictly increasing, i.e. :math:`\forall i<j, x_i <
    x_j`.
    """

    event_dim = 1

    def __call__(self, x):
        """Check that the last axis is strictly increasing.

        Parameters
        ----------
        x : Array
            Values to check; the last axis is the vector.

        Returns
        -------
        Array
            Boolean mask of shape ``x.shape[:-1]``.
        """
        return jnp.all(x[..., 1:] > x[..., :-1], axis=-1)

    def feasible_like(self, prototype):
        """Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            An array of ``1, 2, ..., n``.
        """
        return jnp.ones(prototype.shape, dtype=prototype.dtype) * jnp.arange(
            1, prototype.shape[-1] + 1, dtype=prototype.dtype
        )

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        The constraint is a stateless
        :class:`~numpyro.distributions.constraints._SingletonConstraint`, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`_StrictlyIncreasingVector`.
        """
        return isinstance(other, _StrictlyIncreasingVector)


class _StrictlyDecreasingVector(_SingletonConstraint):
    r"""Constrain values to be strictly decreasing, i.e. :math:`\forall i<j,x_i >
    x_j`.
    """

    event_dim = 1

    def __call__(self, x):
        """Check that the last axis is strictly decreasing.

        Parameters
        ----------
        x : Array
            Values to check; the last axis is the vector.

        Returns
        -------
        Array
            Boolean mask of shape ``x.shape[:-1]``.
        """
        return jnp.all(x[..., 1:] < x[..., :-1], axis=-1)

    def feasible_like(self, prototype):
        """Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            An array of ``n, n-1, ..., 1``.
        """
        return jnp.ones(prototype.shape, dtype=prototype.dtype) * jnp.arange(
            prototype.shape[-1], 0, -1, dtype=prototype.dtype
        )

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        The constraint is a stateless
        :class:`~numpyro.distributions.constraints._SingletonConstraint`, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`_StrictlyDecreasingVector`.
        """
        return isinstance(other, _StrictlyDecreasingVector)


class _PositiveIncreasingVector(_SingletonConstraint):
    r"""Constrain values to be positive and increasing, i.e. :math:`\forall i<j,\ 0 <
    x_i \leq x_j`.
    """

    event_dim = 1

    def __call__(self, x):
        """Check that the last axis is positive and increasing.

        Parameters
        ----------
        x : Array
            Values to check; the last axis is the vector.

        Returns
        -------
        Array
            Boolean mask of shape ``x.shape[:-1]``.
        """
        return increasing_vector.check(x) & independent(positive, 1).check(x)

    def feasible_like(self, prototype):
        """Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            An array of ones, which is positive and (weakly) increasing.
        """
        return jnp.ones(prototype.shape, dtype=prototype.dtype)

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        The constraint is a stateless
        :class:`~numpyro.distributions.constraints._SingletonConstraint`, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`_PositiveIncreasingVector`.
        """
        return isinstance(other, _PositiveIncreasingVector)


class _PositiveDecreasingVector(_SingletonConstraint):
    r"""Constrain values to be positive and decreasing, i.e. :math:`\forall i<j,\ x_i
    \geq x_j > 0`.
    """

    event_dim = 1

    def __call__(self, x):
        """Check that the last axis is positive and decreasing.

        Parameters
        ----------
        x : Array
            Values to check; the last axis is the vector.

        Returns
        -------
        Array
            Boolean mask of shape ``x.shape[:-1]``.
        """
        return decreasing_vector.check(x) & independent(positive, 1).check(x)

    def feasible_like(self, prototype):
        """Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            An array of ones, which is positive and (weakly) decreasing.
        """
        return jnp.ones(prototype.shape, dtype=prototype.dtype)

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        The constraint is a stateless
        :class:`~numpyro.distributions.constraints._SingletonConstraint`, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`_PositiveDecreasingVector`.
        """
        return isinstance(other, _PositiveDecreasingVector)


class _AllConstraint(Constraint):
    """Constrain disjoint slices of the event axis to satisfy their own constraints.

    Each constraint is paired with the slice of the event axis it governs, and a value
    is admissible only if every slice satisfies its own constraint. This is how
    :class:`~gwkokab.models.utils.JointDistribution` builds its support out of the
    supports of its marginals.

    Parameters
    ----------
    constraints : Sequence[Constraint]
        One constraint per slice.
    event_slices : Sequence[int | Tuple[int, int]]
        Position of each slice on the event axis: an ``int`` index for a scalar slice,
        a ``(start, stop)`` pair for a contiguous block. Must be the same length as
        ``constraints``.

    Raises
    ------
    AssertionError
        If ``constraints`` and ``event_slices`` have different lengths.
    """

    event_dim = -1

    def __init__(
        self,
        constraints: Sequence[Constraint],
        event_slices: Sequence[int | Tuple[int, int]],
    ):
        assert len(constraints) == len(event_slices), (
            f"Number of constraints ({len(constraints)}) must match the number of "
            f"event slices ({len(event_slices)})"
        )
        self.constraints = constraints
        self.event_slices = event_slices

    def __call__(self, x):
        """Check every slice against its own constraint.

        Parameters
        ----------
        x : Array
            Values whose last axis is the event axis.

        Returns
        -------
        Array
            Boolean mask, true where all slices satisfy their constraints.
        """
        mask = None
        for constraint, event_slice in zip(self.constraints, self.event_slices):
            if isinstance(event_slice, int):
                x_slice = lax.dynamic_index_in_dim(
                    x, event_slice, axis=self.event_dim, keepdims=False
                )
            else:
                x_slice = lax.dynamic_slice_in_dim(
                    x,
                    event_slice[0],
                    event_slice[1] - event_slice[0],
                    axis=self.event_dim,
                )
            if mask is None:
                mask = constraint.check(x_slice)
            else:
                mask = jnp.logical_and(mask, constraint.check(x_slice))
        return mask

    def feasible_like(self, prototype: Array) -> Array:
        """Produce a value satisfying every constraint, shaped like ``prototype``.

        Each slice's own ``feasible_like`` is evaluated and the results are concatenated
        back along the event axis, broadcasting scalar slices up to the prototype's rank.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            A feasible value with the same event layout as ``prototype``.
        """
        feasible_values = []
        for constraint, event_slice in zip(self.constraints, self.event_slices):
            if isinstance(event_slice, int):
                prototype_slice = lax.dynamic_index_in_dim(
                    prototype, event_slice, axis=self.event_dim, keepdims=False
                )
            else:
                prototype_slice = lax.dynamic_slice_in_dim(
                    prototype,
                    event_slice[0],
                    event_slice[1] - event_slice[0],
                    axis=self.event_dim,
                )
            feasible_values.append(constraint.feasible_like(prototype_slice))
        max_ndim = prototype.ndim
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
        """Flatten the constraint into a JAX pytree.

        Returns
        -------
        tuple
            The children and the auxiliary metadata, in the layout NumPyro's constraint
            registry expects.
        """
        return (self.constraints,), (
            ("constraints",),
            {"event_slices": self.event_slices},
        )

    def eq(self, other: object, static: bool = False) -> bool:
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is an :class:`_AllConstraint` over an equal sequence of constraints.
        """
        if not isinstance(other, _AllConstraint):
            return False
        if len(self.constraints) != len(other.constraints):
            return False
        return all(
            constraint == other_constraint
            for constraint, other_constraint in zip(
                self.constraints, other.constraints, strict=True
            )
        )


class _AnyConstraint(Constraint):
    """Constrain values to satisfy at least one of the constraints.

    Unlike :class:`_AllConstraint`, every constraint is applied to the whole value
    rather than to a slice of it, and the masks are combined with a logical *or*.

    Parameters
    ----------
    constraints : Sequence[Constraint]
        The alternatives; at least one must be satisfied. Must be non-empty.
    """

    def __init__(self, constraints: Sequence[Constraint]):
        self.constraints = constraints

    def __call__(self, x):
        """Check the value against each alternative in turn.

        Parameters
        ----------
        x : Array
            Values to check.

        Returns
        -------
        Array
            Boolean mask, true where at least one constraint is satisfied.
        """
        mask = self.constraints[0].check(x)
        for constraint in self.constraints[1:]:
            mask |= constraint.check(x)
        return mask

    def feasible_like(self, prototype: Array) -> Array:
        """Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            A value feasible under the *first* alternative, which suffices
            because only one alternative has to hold.
        """
        return self.constraints[0].feasible_like(prototype)

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        Returns
        -------
        tuple
            The children and the auxiliary metadata, in the layout NumPyro's constraint
            registry expects.
        """
        return (self.constraints,), (("constraints",), dict())

    def eq(self, other: object, static: bool = False) -> bool:
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is an :class:`_AnyConstraint` over an equal sequence of constraints.
        """
        if not isinstance(other, _AnyConstraint):
            return False
        if len(self.constraints) != len(other.constraints):
            return False
        return all(
            constraint == other_constraint
            for constraint, other_constraint in zip(
                self.constraints, other.constraints, strict=True
            )
        )


class _TransformedConstraint(Constraint):
    """The image of a constraint under a chain of transforms.

    A value is admissible if unwinding the transforms lands it in ``base_support``, and
    if every intermediate value lies in the corresponding transform's codomain. This is
    narrower than the codomain of the final transform alone, which is what NumPyro would
    otherwise report for a transformed distribution.

    Parameters
    ----------
    base_support : Constraint
        The constraint satisfied before any transform is applied.
    transforms : Union[Transform, Sequence[Transform]]
        The transform, or chain of transforms, applied to the base space in order.

    See Also
    --------
    gwkokab.models.utils.ExtendedSupportTransformedDistribution
    """

    def __init__(
        self,
        base_support: Constraint,
        transforms: Union[Transform, Sequence[Transform]],
    ):
        self.base_support = base_support
        if isinstance(transforms, Transform):
            self.transforms = (transforms,)
        else:
            self.transforms = tuple(transforms)

    def __call__(self, x: Array) -> Array:
        """Check values by unwinding the transform chain.

        Parameters
        ----------
        x : Array
            Values in the transformed space.

        Returns
        -------
        Array
            Boolean mask, true where ``x`` lies in every transform's codomain and
            its preimage lies in ``base_support``.
        """
        y = x
        mask = jnp.ones(x.shape[:-1], dtype=bool)
        for transform in reversed(self.transforms):
            mask = jnp.logical_and(mask, transform.codomain.check(y))
            y = transform.inv(y)
        mask = jnp.logical_and(mask, self.base_support.check(y))
        return mask

    def feasible_like(self, prototype: Array) -> Array:
        """Produce a value satisfying the constraint, shaped like ``prototype``.

        Parameters
        ----------
        prototype : Array
            Array whose shape and dtype to match.

        Returns
        -------
        Array
            A feasible value of ``base_support``, pushed forward through the
            transform chain.
        """
        fl = self.base_support.feasible_like(prototype)
        for transform in self.transforms:
            fl = transform(fl)
        return fl

    def tree_flatten(self):
        """Flatten the constraint into a JAX pytree.

        Returns
        -------
        tuple
            The children and the auxiliary metadata, in the layout NumPyro's constraint
            registry expects.
        """
        return (self.base_support, self.transforms), (
            ("base_support", "transforms"),
            dict(),
        )

    def eq(self, other: object, static: bool = False) -> bool:
        """Compare two constraints structurally.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.constraints.Constraint.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is a :class:`_TransformedConstraint` with an equal base support and
            transform chain.
        """
        if not isinstance(other, _TransformedConstraint):
            return False
        if len(self.transforms) != len(other.transforms):
            return False
        return self.base_support == other.base_support and all(
            transform == other_transform
            for transform, other_transform in zip(self.transforms, other.transforms)
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
transform_constraint = _TransformedConstraint
