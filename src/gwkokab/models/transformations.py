# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
r"""Provides implementation of various transformations using
:class:`~numpyro.distributions.transforms.Transform`.

Most of these are *changes of mass coordinates*: a population model is naturally
written in one parameterisation -- :math:`(m_1, q)`, say -- while the data or the
selection function live in another, such as :math:`(m_1, m_2)` or
:math:`(M_c, \eta)`. Each transform supplies the forward and inverse maps together
with the log Jacobian determinant that keeps the density normalised across the
change. :class:`RedshiftToLuminosityDistance` does the same for the
redshift/distance pair, and :class:`BlockTransform` composes several transforms
that act on disjoint slices of one event vector.

The module also registers ``biject_to`` rules for the custom constraints defined in
:mod:`gwkokab.models.constraints`, so that samplers working in an unconstrained
space know how to reach an ordered vector or a mass sandwich.
"""

from typing import Sequence, Tuple, Union

import jax
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import constraints
from numpyro.distributions.transforms import (
    AbsTransform,
    AffineTransform,
    biject_to,
    ComposeTransform,
    OrderedTransform,
    PowerTransform,
    SigmoidTransform,
    Transform,
)

from ..cosmology import Cosmology
from ..utils.transformations import (
    chirp_mass,
    delta_m,
    delta_m_to_symmetric_mass_ratio,
    m1_q_to_m2,
    m2_q_to_m1,
    mass_ratio,
    Mc_eta_to_m1_m2,
    symmetric_mass_ratio,
    total_mass,
)
from .constraints import (
    all_constraint,
    decreasing_vector,
    increasing_vector,
    mass_ratio_mass_sandwich,
    mass_sandwich,
    positive_decreasing_vector,
    positive_increasing_vector,
    strictly_decreasing_vector,
    strictly_increasing_vector,
)


__all__ = [
    "BlockTransform",
    "ComponentMassesAndRedshiftToDetectedMassAndRedshift",
    "ComponentMassesToChirpMassAndDelta",
    "ComponentMassesToChirpMassAndSymmetricMassRatio",
    "ComponentMassesToMassRatioAndSecondaryMass",
    "ComponentMassesToPrimaryMassAndMassRatio",
    "ComponentMassesToTotalMassAndMassRatio",
    "DeltaToSymmetricMassRatio",
    "PrimaryMassAndMassRatioToComponentMassesTransform",
    "RedshiftToLuminosityDistance",
    "SourceMassAndRedshiftToDetectedMassAndRedshift",
]


class BlockTransform(Transform):
    r"""A transform that applies multiple sub-transforms to disjoint slices of the event
    dimension.

    This class implements a block-separable transformation of the form

    .. math::

        T(x)
        = \big( T_1(x_{S_1}), \; T_2(x_{S_2}), \; \dots, \; T_K(x_{S_K}) \big),

    where each :math:`T_i` is a ``Transform`` and :math:`S_i` is a slice of the
    event dimension specified by ``event_slices``.  The slices must be
    pairwise disjoint so that no parameters or coordinates are shared between
    sub-transforms.

    Because each sub-transform acts independently on its own coordinate block,
    the Jacobian matrix has block-diagonal structure:

    .. math::

        J_T(x)
        =
        \begin{pmatrix}
            J_{T_1}(x_{S_1}) & 0 & \cdots & 0 \\
            0 & J_{T_2}(x_{S_2}) & \cdots & 0 \\
            \vdots & \vdots & \ddots & \vdots \\
            0 & 0 & \cdots & J_{T_K}(x_{S_K})
        \end{pmatrix}.

    Consequently, the log absolute determinant of the Jacobian factorizes:

    .. math::

        \log \left| \det J_T(x) \right|
        =
        \sum_{i=1}^K
        \log \left| \det J_{T_i}(x_{S_i}) \right|.

    Parameters
    ----------
    *transforms : Transform
        A sequence of sub-transforms :math:`T_1, \dots, T_K`. Each transform is
        applied independently to its corresponding slice of the input.
    event_slices : Sequence[Union[int, Tuple[int, int]]]
        A sequence specifying the slices :math:`S_i` of the event dimension.
        Each entry is either:
        - an integer ``j`` (interpreted as selecting ``x[..., j]``), or
        - a tuple ``(start, end)`` denoting the half-open interval
        :math:`[\mathrm{start}, \mathrm{end})`.

    Notes
    -----
    - The overall transformation is equivalent to a product of independent
      transforms acting on different subspaces.
    - No checks are performed to ensure that slices fully cover the event
      dimension or that the resulting concatenation is contiguous.

    Warning
    -------
    - ``event_slices`` **must be non-overlapping**. Overlapping slices violate
      the independence assumption and produce incorrect Jacobians.
    - Each sub-transform must be dimensionally compatible with the slice it
      receives.
    - If a slice misses part of the event dimension or overlaps with another,
      the forward and inverse mappings may not be valid.

    Examples
    --------
    >>> t1 = AffineTransform(loc=0.0, scale=1.0)
    >>> t2 = ExpTransform()
    >>> bt = BlockTransform(t1, t2, event_slices=[(0, 3), (3, 4)])
    >>> x = jnp.array([1.0, 2.0, 3.0, 0.5])
    >>> y = bt(x)
    >>> x_recovered = bt.inv(y)
    """

    def __init__(
        self,
        *transforms: Transform,
        event_slices: Sequence[Union[int, Tuple[int, int]]],
    ):
        self.event_slices = tuple(event_slices)
        self.transforms = transforms
        assert len(self.event_slices) == len(self.transforms), (
            "Number of event slices must match number of transforms."
            f"Got {len(self.event_slices)} slices and {len(self.transforms)} transforms."
        )

    @property
    def domain(self) -> constraints.Constraint:
        """Domain of the block transform.

        Returns
        -------
        constraints.Constraint
            An :func:`~gwkokab.models.constraints.all_constraint` requiring each slice to
            lie in the domain of its own sub-transform.
        """
        return all_constraint([t.domain for t in self.transforms], self.event_slices)

    @property
    def codomain(self) -> constraints.Constraint:
        """Codomain of the block transform.

        Returns
        -------
        constraints.Constraint
            An :func:`~gwkokab.models.constraints.all_constraint` requiring each slice to
            lie in the codomain of its own sub-transform.
        """
        return all_constraint([t.codomain for t in self.transforms], self.event_slices)

    @staticmethod
    def _block(x: Array, event_slice: Union[int, Tuple[int, int]]) -> Array:
        """Take the block of the event dimension that ``event_slice`` selects.

        Parameters
        ----------
        x : Array
            Array whose last axis is the event axis.
        event_slice : Union[int, Tuple[int, int]]
            A bare index, which drops the event axis, or a half-open ``(start, end)`` pair,
            which keeps it.

        Returns
        -------
        Array
            The selected block.
        """
        if isinstance(event_slice, int):
            return jax.lax.dynamic_index_in_dim(x, event_slice, axis=-1, keepdims=False)
        return jax.lax.dynamic_slice_in_dim(
            x, event_slice[0], event_slice[1] - event_slice[0], axis=-1
        )

    def _join(self, blocks: Sequence[Array]) -> Array:
        """Reassemble transformed blocks into a single event.

        A block selected by a bare index lost its event axis on the way in, so it is given
        back before the blocks are concatenated.

        Parameters
        ----------
        blocks : Sequence[Array]
            The transformed blocks, in the same order as ``event_slices``.

        Returns
        -------
        Array
            The blocks concatenated along the event axis.
        """
        return jnp.concatenate(
            [
                jnp.expand_dims(block, axis=-1)
                if isinstance(event_slice, int)
                else block
                for block, event_slice in zip(blocks, self.event_slices)
            ],
            axis=-1,
        )

    def __call__(self, x: Array) -> Array:
        """Apply each sub-transform to its own slice.

        Parameters
        ----------
        x : Array
            Values whose last axis is the event axis.

        Returns
        -------
        Array
            The transformed value, with the blocks reassembled.
        """
        return self._join([
            transform(self._block(x, event_slice))
            for transform, event_slice in zip(self.transforms, self.event_slices)
        ])

    def _inverse(self, y: Array) -> Array:
        """Apply the inverse of each sub-transform to its own slice.

        Parameters
        ----------
        y : Array
            Values in the codomain.

        Returns
        -------
        Array
            The preimage of ``y``.
        """
        return self._join([
            transform.inv(self._block(y, event_slice))
            for transform, event_slice in zip(self.transforms, self.event_slices)
        ])

    def log_abs_det_jacobian(self, x: Array, y: Array, intermediates=None):
        r"""Log absolute determinant of the block-diagonal Jacobian.

        Because the blocks are independent, the determinant factorises and the log
        determinant is the sum of the sub-transforms' contributions:

        .. math::
            \log\left|\det J_T(x)\right| =
            \sum_{i=1}^{K} \log\left|\det J_{T_i}(x_{S_i})\right|.

        An elementwise sub-transform reports one term per coordinate while one with an
        event dimension has already reduced its block, so any axes beyond the batch shape
        are summed away before the terms are added.

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Passed through to each sub-transform. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant, of the batch shape of ``x``.
        """
        batch_ndim = jnp.ndim(x) - 1
        log_detJ = 0.0
        for transform, event_slice in zip(self.transforms, self.event_slices):
            log_detJ_slice = transform.log_abs_det_jacobian(
                self._block(x, event_slice), self._block(y, event_slice), intermediates
            )
            # an elementwise sub-transform reports one term per coordinate, whereas one
            # with an event dimension has already reduced its block; sum away whatever
            # axes are left over so that every block contributes a batch-shaped term
            extra_ndim = jnp.ndim(log_detJ_slice) - batch_ndim
            if extra_ndim > 0:
                log_detJ_slice = jnp.sum(
                    log_detJ_slice,
                    axis=tuple(range(batch_ndim, jnp.ndim(log_detJ_slice))),
                )
            log_detJ += log_detJ_slice
        return log_detJ

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        Returns
        -------
        tuple
            The sub-transforms as children, with ``event_slices`` carried as static
            metadata.
        """
        return (self.transforms,), (
            ("transforms",),
            {"event_slices": self.event_slices},
        )

    def eq(self, value, static: bool = False):
        """Compare two block transforms structurally.

        Parameters
        ----------
        value : object
            The object to compare against.
        static : bool, optional
            Forwarded to each sub-transform's ``eq``. Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``value`` is a :class:`BlockTransform` with the same event
            slices and pairwise equal sub-transforms.
        """
        if not isinstance(value, BlockTransform):
            return False
        return self.event_slices == value.event_slices and all(
            t1.eq(t2, static=static)
            for t1, t2 in zip(self.transforms, value.transforms)
        )


class PrimaryMassAndMassRatioToComponentMassesTransform(Transform):
    r"""Transforms a primary mass and mass ratio to component masses.

    .. math::
        f: (m_1, q)\to (m_1, m_1q)

    .. math::
        f^{-1}: (m_1, m_2)\to (m_1, m_2/m_1)
    """

    domain = constraints.independent(
        constraints.interval(
            jnp.zeros((2,)), jnp.array([jnp.finfo(jnp.result_type(float)).max, 1.0])
        ),
        1,
    )
    r""":math:`\mathcal{D}(f) = \mathbb{R}^2_+\times[0, 1]`"""
    codomain = positive_decreasing_vector
    r""":math:`\mathcal{C}(f)=\{(m_1, m_2)\in\mathbb{R}^2_+\mid m_1\geq m_2>0\}`"""

    def __call__(self, x: Array):
        """Map the primary mass and mass ratio :math:`(m_1, q)` to the component masses
        :math:`(m_1, m_1 q)`.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        m1, q = jnp.unstack(x, axis=-1)
        m2 = jnp.multiply(m1, q)
        m1m2 = jnp.stack((m1, m2), axis=-1)
        return m1m2

    def _inverse(self, y: Array):
        """Map the component masses :math:`(m_1, m_2)` back to the primary mass and mass
        ratio :math:`(m_1, m_2/m_1)`.

        Parameters
        ----------
        y : Array
            Values in the codomain, stacked along the last axis.

        Returns
        -------
        Array
            The preimage of ``y``, stacked along the last axis.
        """
        m1, m2 = jnp.unstack(y, axis=-1)
        q = mass_ratio(m2=m2, m1=m1)
        m1q = jnp.stack((m1, q), axis=-1)
        return m1q

    def log_abs_det_jacobian(self, x: Array, y: Array, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(|m_1|)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        m1 = x[..., 0]
        return jnp.log(jnp.abs(m1))

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`PrimaryMassAndMassRatioToComponentMassesTransform`. The transform is
            stateless, so type identity is enough.
        """
        if not isinstance(other, PrimaryMassAndMassRatioToComponentMassesTransform):
            return False
        return self.domain == other.domain


class ComponentMassesToChirpMassAndSymmetricMassRatio(Transform):
    r"""Transforms component masses to chirp mass and symmetric mass ratio.

    .. math::
        f: (m_1, m_2)\to \left(\frac{(m_1m_2)^{3/5}}{(m_1+m_2)^{1/5}}, \frac{m_1m_2}{(m_1+m_2)^{2}}\right)
    """

    domain = positive_decreasing_vector
    r""":math:`\mathcal{D}(f)=\{(m_1,m_2)\in\mathbb{R}^2_+\mid m_1\geq m_2>0\}`"""

    codomain = constraints.independent(
        constraints.interval(
            jnp.zeros((2,)), jnp.array([jnp.finfo(jnp.result_type(float)).max, 0.25])
        ),
        1,
    )
    r""":math:`\mathcal{C}(f) = \mathbb{R}^2_+\times[0, 0.25]`"""

    def __call__(self, x):
        r"""Map the component masses :math:`(m_1, m_2)` to the chirp mass and symmetric
        mass ratio :math:`(M_c, \eta)`.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        m1, m2 = jnp.unstack(x, axis=-1)
        Mc = chirp_mass(m1=m1, m2=m2)
        eta = symmetric_mass_ratio(m1=m1, m2=m2)
        return jnp.stack((Mc, eta), axis=-1)

    def _inverse(self, y):
        r"""Map the chirp mass and symmetric mass ratio :math:`(M_c, \eta)` back to the
        component masses :math:`(m_1, m_2)`.

        Parameters
        ----------
        y : Array
            Values in the codomain, stacked along the last axis.

        Returns
        -------
        Array
            The preimage of ``y``, stacked along the last axis.
        """
        Mc, eta = jnp.unstack(y, axis=-1)
        m1, m2 = Mc_eta_to_m1_m2(Mc=Mc, eta=eta)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right)=\frac{6}{5}\ln(\eta)+\frac{1}{2}\ln(1-4\eta)-\ln(M_c)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        Mc, eta = jnp.unstack(y, axis=-1)
        log_detJ = 1.2 * jnp.log(eta) + 0.5 * jnp.log1p(-4.0 * eta) - jnp.log(Mc)
        return log_detJ

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`. The transform is
            stateless, so type identity is enough.
        """
        if not isinstance(other, ComponentMassesToChirpMassAndSymmetricMassRatio):
            return False
        return self.domain == other.domain


class DeltaToSymmetricMassRatio(Transform):
    r"""Transforms delta to symmetric mass ratio.

    .. math::
        \eta = f(\delta) = \frac{1-\delta^2}{4}

    .. math::
        \delta = f^{-1}(\eta) = \sqrt{1-4\eta}
    """

    domain = constraints.unit_interval
    r""":math:`\mathcal{D}(f) = [0, 1]`"""
    codomain = constraints.interval(0.0, 0.25)
    r""":math:`\mathcal{C}(f) = [0, 0.25]`"""

    def __call__(self, x):
        r"""Map a fractional mass difference to a symmetric mass ratio.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        delta_sq = jnp.square(x)
        return jnp.multiply(jnp.subtract(1.0, delta_sq), 0.25)

    def _inverse(self, y):
        r"""Map a symmetric mass ratio back to a fractional mass difference.

        Parameters
        ----------
        y : Array
            Values in the codomain, stacked along the last axis.

        Returns
        -------
        Array
            The preimage of ``y``, stacked along the last axis.
        """
        eta4 = jnp.multiply(4, y)
        return jnp.sqrt(jnp.subtract(1, eta4))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(\delta) - \ln(2)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        return jnp.subtract(jnp.log(x), jnp.log(2.0))

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`DeltaToSymmetricMassRatio`. The transform is
            stateless, so type identity is enough.
        """
        return isinstance(other, type(self))


class ComponentMassesToChirpMassAndDelta(Transform):
    r"""Transforms component masses to chirp mass and delta.

    .. math::
        f: (m_1, m_2) \to (M_c, \delta)
    """

    domain = positive_decreasing_vector
    r""":math:`\mathcal{D}(f)=\{(m_1,m_2)\in\mathbb{R}^2_+\mid m_1\geq m_2>0\}`"""

    codomain = constraints.independent(
        constraints.interval(
            jnp.zeros(2), jnp.array([jnp.finfo(jnp.result_type(float)).max, 1.0])
        ),
        1,
    )
    r""":math:`\mathcal{C}(f) = \mathbb{R}^2_+\times[0, 1]`"""

    def __call__(self, x):
        r"""Map the component masses :math:`(m_1, m_2)` to the chirp mass and fractional
        mass difference :math:`(M_c, \delta)`.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        m1 = x[..., 0]
        m2 = x[..., 1]
        Mc = chirp_mass(m1=m1, m2=m2)
        delta = delta_m(m1=m1, m2=m2)
        return jnp.stack((Mc, delta), axis=-1)

    def _inverse(self, y):
        r"""Map the chirp mass and fractional mass difference :math:`(M_c, \delta)` back
        to the component masses :math:`(m_1, m_2)`.

        Parameters
        ----------
        y : Array
            Values in the codomain, stacked along the last axis.

        Returns
        -------
        Array
            The preimage of ``y``, stacked along the last axis.
        """
        Mc, delta = jnp.unstack(y, axis=-1)
        eta = delta_m_to_symmetric_mass_ratio(delta_m=delta)
        m1, m2 = Mc_eta_to_m1_m2(Mc=Mc, eta=eta)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(2M_c) - 2\ln(m_1+m_2)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        m1, m2 = jnp.unstack(x, axis=-1)
        M = total_mass(m1=m1, m2=m2)
        log_Mc = jnp.log(y[..., 0])
        log_detJ = jnp.log(2.0)
        log_detJ = jnp.add(log_detJ, log_Mc)
        log_detJ = jnp.add(log_detJ, jnp.multiply(-2.0, jnp.log(M)))
        return log_detJ

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`ComponentMassesToChirpMassAndDelta`. The transform is
            stateless, so type identity is enough.
        """
        return isinstance(other, type(self))


class SourceMassAndRedshiftToDetectedMassAndRedshift(Transform):
    r"""Transforms source mass and redshift to detected mass and redshift.

    .. math::
        f: (m_{\text{source}}, z) \to (m_{\text{detected}}, z)
    """

    domain = constraints.independent(constraints.positive, 1)
    r""":math:`\mathcal{D}(f) = \mathbb{R}^2_+`"""
    codomain = constraints.independent(constraints.positive, 1)
    r""":math:`\mathcal{C}(f) = \mathbb{R}^2_+`"""

    def __call__(self, x):
        r"""Map the source-frame mass and redshift :math:`(m_{\text{source}}, z)` to the
        detector-frame mass and redshift :math:`(m_{\text{source}}(1+z), z)`.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        m_source, z = jnp.unstack(x, axis=-1)
        m_detected = m_source * (1 + z)
        return jnp.stack((m_detected, z), axis=-1)

    def _inverse(self, y):
        r"""Map the detector-frame mass and redshift :math:`(m_{\text{detected}}, z)`
        back to the source-frame mass and redshift :math:`(m_{\text{detected}}/(1+z),
        z)`.

        Parameters
        ----------
        y : Array
            Values in the codomain, stacked along the last axis.

        Returns
        -------
        Array
            The preimage of ``y``, stacked along the last axis.
        """
        m_detected, z = jnp.unstack(y, axis=-1)
        m_source = m_detected / (1 + z)
        return jnp.stack((m_source, z), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(1+z)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        z = x[..., 1]
        return jnp.log1p(z)

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`SourceMassAndRedshiftToDetectedMassAndRedshift`. The transform is
            stateless, so type identity is enough.
        """
        return isinstance(other, type(self))


class ComponentMassesAndRedshiftToDetectedMassAndRedshift(Transform):
    r"""Transforms component masses and redshift to detected masses and redshift.

    .. math::
        f: (m_1, m_2, z) \to (m_{1, \text{detected}}, m_{2, \text{detected}}, z)
    """

    domain = constraints.independent(constraints.positive, 1)
    r""":math:`\mathcal{D}(f) = \mathbb{R}^3_+`"""
    codomain = constraints.independent(constraints.positive, 1)
    r""":math:`\mathcal{C}(f) = \mathbb{R}^3_+`"""

    def __call__(self, x):
        """Map the source-frame component masses and redshift :math:`(m_1, m_2, z)` to
        the detector-frame component masses and redshift :math:`(m_1(1+z), m_2(1+z),
        z)`.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        m1_source, m2_source, z = jnp.unstack(x, axis=-1)
        m1_detected = m1_source * (1 + z)
        m2_detected = m2_source * (1 + z)
        return jnp.stack((m1_detected, m2_detected, z), axis=-1)

    def _inverse(self, y):
        """Map the detector-frame component masses and redshift :math:`(m_1, m_2, z)`
        back to the source-frame component masses and redshift :math:`(m_1/(1+z),
        m_2/(1+z), z)`.

        Parameters
        ----------
        y : Array
            Values in the codomain, stacked along the last axis.

        Returns
        -------
        Array
            The preimage of ``y``, stacked along the last axis.
        """
        m1_detected, m2_detected, z = jnp.unstack(y, axis=-1)
        m1_source = m1_detected / (1 + z)
        m2_source = m2_detected / (1 + z)
        return jnp.stack((m1_source, m2_source, z), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = 2\ln(1+z)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        z = x[..., 2]
        return 2 * jnp.log1p(z)

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`. The transform is
            stateless, so type identity is enough.
        """
        return isinstance(other, type(self))


class ComponentMassesToPrimaryMassAndMassRatio(Transform):
    r"""Transforms component masses and redshift to primary mass and mass ratio.

    .. math::
        f: (m_1, m_2) \to (m_1, q)

    .. math::
        f^{-1}: (m_1, q) \to (m_1, m_2)
    """

    domain = positive_decreasing_vector
    r""":math:`\mathcal{D}(f)=\{(m_1,m_2)\in\mathbb{R}^2_+\mid m_1\geq m_2>0\}`"""

    codomain = constraints.independent(
        constraints.open_interval(
            jnp.zeros(2), jnp.array([jnp.finfo(jnp.result_type(float)).max, 1.0])
        ),
        1,
    )
    r""":math:`\mathcal{C}(f) = \mathbb{R}^2_+\times(0, 1)`"""

    def __call__(self, x):
        """Map the component masses :math:`(m_1, m_2)` to the primary mass and mass
        ratio :math:`(m_1, m_2/m_1)`.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        m1, m2 = jnp.unstack(x, axis=-1)
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((m1, q), axis=-1)

    def _inverse(self, y):
        """Map the primary mass and mass ratio :math:`(m_1, q)` back to the component
        masses :math:`(m_1, m_1 q)`.

        Parameters
        ----------
        y : Array
            Values in the codomain, stacked along the last axis.

        Returns
        -------
        Array
            The preimage of ``y``, stacked along the last axis.
        """
        m1, q = jnp.unstack(y, axis=-1)
        m2 = m1_q_to_m2(m1=m1, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = -\ln(|m_1|)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        m1 = x[..., 0]
        return -jnp.log(jnp.abs(m1))

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`ComponentMassesToPrimaryMassAndMassRatio`. The transform is
            stateless, so type identity is enough.
        """
        return isinstance(other, type(self))


class ComponentMassesToMassRatioAndSecondaryMass(Transform):
    r"""Transforms component masses and redshift to mass ratio and secondary mass.

    .. math::
        f: (m_1, m_2) \to (q, m_2)

    .. math::
        f^{-1}: (q, m_2) \to (m_1, m_2)
    """

    domain = positive_decreasing_vector
    r""":math:`\mathcal{D}(f)=\{(m_1,m_2)\in\mathbb{R}^2_+\mid m_1\geq m_2>0\}`"""

    codomain = constraints.independent(
        constraints.interval(
            jnp.zeros(2), jnp.array([1.0, jnp.finfo(jnp.result_type(float)).max])
        ),
        1,
    )
    r""":math:`\mathcal{C}(f) = [0, 1]\times\mathbb{R}_+`"""

    def __call__(self, x):
        """Map the component masses :math:`(m_1, m_2)` to the mass ratio and secondary
        mass :math:`(m_2/m_1, m_2)`.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        m1, m2 = jnp.unstack(x, axis=-1)
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((q, m2), axis=-1)

    def _inverse(self, y):
        """Map the mass ratio and secondary mass :math:`(q, m_2)` back to the component
        masses :math:`(m_2/q, m_2)`.

        Parameters
        ----------
        y : Array
            Values in the codomain, stacked along the last axis.

        Returns
        -------
        Array
            The preimage of ``y``, stacked along the last axis.
        """
        q, m2 = jnp.unstack(y, axis=-1)
        m1 = m2_q_to_m1(m2=m2, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(q) - \ln(m_1)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        m1 = x[..., 0]
        q = y[..., 0]
        return jnp.log(q) - jnp.log(m1)

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`ComponentMassesToMassRatioAndSecondaryMass`. The transform is
            stateless, so type identity is enough.
        """
        return isinstance(other, type(self))


class ComponentMassesToTotalMassAndMassRatio(Transform):
    r"""Transforms component masses to total mass and mass ratio.

    .. math::
        f: (m_1, m_2) \to (M, q)
    """

    domain = positive_decreasing_vector
    codomain = constraints.independent(
        constraints.interval(
            jnp.zeros(2), jnp.array([jnp.finfo(jnp.result_type(float)).max, 1.0])
        ),
        1,
    )

    def __call__(self, x):
        """Map the component masses :math:`(m_1, m_2)` to the total mass and mass ratio
        :math:`(m_1 + m_2, m_2/m_1)`.

        Parameters
        ----------
        x : Array
            Values in the domain, stacked along the last axis.

        Returns
        -------
        Array
            The transformed values, stacked along the last axis.
        """
        m1, m2 = jnp.unstack(x, axis=-1)
        M = total_mass(m1=m1, m2=m2)
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((M, q), axis=-1)

    def _inverse(self, y):
        """Map the total mass and mass ratio back to the component masses.

        Parameters
        ----------
        y : Array
            Values :math:`(M, q)` stacked along the last axis.

        Returns
        -------
        Array
            The component masses :math:`(M/(1+q), qM/(1+q))`. The division is guarded
            against :math:`q = -1`, which is outside the codomain but can be reached by an
            unconstrained sampler; infinity is returned there rather than a NaN gradient.
        """
        M, q = jnp.unstack(y, axis=-1)
        safe_q = jnp.where(q == -1.0, 1.0, q)
        m1 = jnp.where(q == -1.0, jnp.inf, M / (1 + safe_q))
        m2 = m1_q_to_m2(m1=m1, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(1+q) - \ln(m_1)

        Parameters
        ----------
        x : Array
            Values in the domain.
        y : Array
            The corresponding values in the codomain.
        intermediates : Any, optional
            Accepted for API compatibility and ignored. Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        m1 = x[..., 0]
        q = y[..., 1]
        return jnp.log(1 + q) - jnp.log(m1)

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        The transform is stateless, so it has no
        children and no metadata.

        Returns
        -------
        tuple
            Two empty containers.
        """
        return (), ((), dict())

    def eq(self, other, static: bool = False):
        """Compare two transforms.

        Parameters
        ----------
        other : object
            The object to compare against.
        static : bool, optional
            Accepted for API compatibility with
            :meth:`numpyro.distributions.transforms.Transform.eq` and ignored.
            Defaults to :data:`False`.

        Returns
        -------
        bool
            :data:`True` if ``other`` is also a :class:`ComponentMassesToTotalMassAndMassRatio`. The transform is
            stateless, so type identity is enough.
        """
        return isinstance(other, type(self))


class RedshiftToLuminosityDistance(Transform):
    r"""Transforms redshift to luminosity distance.

    .. math::
        f: z \to D_L

    .. math::
        f^{-1}: D_L \to z

    .. math::
        \ln\left(|\mathrm{det}(J_f)|\right) = \ln\left(D_c + \frac{c(1+z)}{H_0 E(z)}\right)

    Both directions go through the cosmology's precomputed grid, so the transform is
    differentiable but only as accurate as that grid.

    Parameters
    ----------
    cosmology : Cosmology
        The cosmology supplying the distance/redshift relation.
    """

    domain = constraints.positive
    r""":math:`\mathcal{D}(f) = \mathbb{R}_+`"""
    codomain = constraints.positive
    r""":math:`\mathcal{C}(f) = \mathbb{R}_+`"""

    def __init__(self, cosmology: Cosmology) -> None:
        self.cosmology = cosmology

    def __call__(self, x):
        """Map a redshift to a luminosity distance.

        Parameters
        ----------
        x : Array
            Redshift.

        Returns
        -------
        Array
            Luminosity distance, in Mpc.
        """
        dL = self.cosmology.z_to_DL(x)
        return dL

    def _inverse(self, y):
        """Map a luminosity distance back to a redshift.

        Parameters
        ----------
        y : Array
            Luminosity distance, in Mpc.

        Returns
        -------
        Array
            Redshift.
        """
        z = self.cosmology.DL_to_z(y)
        return z

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""Log absolute determinant of the Jacobian of the forward map.

        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) =
            \ln\left(D_c(z) + (1+z)\frac{\mathrm{d}D_c}{\mathrm{d}z}\right)

        Parameters
        ----------
        x : Array
            Redshift.
        y : Array
            The corresponding luminosity distance. Unused; kept for API compatibility.
        intermediates : Any, optional
            Accepted for API compatibility and ignored.
            Defaults to :data:`None`.

        Returns
        -------
        Array
            The log absolute determinant of the Jacobian.
        """
        Dc = self.cosmology.z_to_Dc(x)
        dDcdz = self.cosmology.dDcdz(x)
        return jnp.log(Dc + (1.0 + x) * dDcdz)

    def tree_flatten(self):
        """Flatten the transform into a JAX pytree.

        Returns
        -------
        tuple
            The cosmology as the sole child, with no static metadata.
        """
        return (self.cosmology,), (("cosmology",), {})


@biject_to.register(type(positive_decreasing_vector))
@biject_to.register(type(decreasing_vector))
@biject_to.register(type(strictly_decreasing_vector))
def _transform_to_positive_ordered_vector(constraint):
    """Bijection from the unconstrained space to a positive, decreasing vector.

    Registered with :func:`~numpyro.distributions.transforms.biject_to` for
    :data:`~gwkokab.models.constraints.positive_decreasing_vector`,
    :data:`~gwkokab.models.constraints.decreasing_vector` and
    :data:`~gwkokab.models.constraints.strictly_decreasing_vector`.

    The chain takes an absolute value to reach the positive orthant, sorts into
    increasing order, and inverts elementwise, which reverses the order.

    Parameters
    ----------
    constraint : Constraint
        The constraint being bijected to. Unused; the transform is parameter-free.

    Returns
    -------
    ComposeTransform
        The bijection onto the constrained space.
    """
    return ComposeTransform([AbsTransform(), OrderedTransform(), PowerTransform(-1.0)])


@biject_to.register(type(positive_increasing_vector))
@biject_to.register(type(increasing_vector))
@biject_to.register(type(strictly_increasing_vector))
def _transform_to_positive_ordered_vector(constraint):
    """Bijection from the unconstrained space to a positive, increasing vector.

    Registered with :func:`~numpyro.distributions.transforms.biject_to` for
    :data:`~gwkokab.models.constraints.positive_increasing_vector`,
    :data:`~gwkokab.models.constraints.increasing_vector` and
    :data:`~gwkokab.models.constraints.strictly_increasing_vector`.

    The chain takes an absolute value to reach the positive orthant and then sorts into
    increasing order.

    Parameters
    ----------
    constraint : Constraint
        The constraint being bijected to. Unused; the transform is parameter-free.

    Returns
    -------
    ComposeTransform
        The bijection onto the constrained space.
    """
    return ComposeTransform([AbsTransform(), OrderedTransform()])


@biject_to.register(mass_sandwich)
def _transform_to_mass_sandwich(constraint):
    r"""Bijection from the unconstrained space onto a mass sandwich.

    Registered with :func:`~numpyro.distributions.transforms.biject_to` for
    :data:`~gwkokab.models.constraints.mass_sandwich`. The chain orders the pair
    decreasingly, squashes it into :math:`(0, 1)` with a sigmoid, and rescales onto
    :math:`[m_\min, m_\max]`, giving :math:`m_\min \leq m_2 \leq m_1 \leq m_\max`.

    Parameters
    ----------
    constraint : _MassSandwichConstraint
        The constraint being bijected to; supplies the mass bounds.

    Returns
    -------
    ComposeTransform
        The bijection onto the constrained space.
    """
    return ComposeTransform([
        AbsTransform(),
        OrderedTransform(),
        PowerTransform(-1.0),
        SigmoidTransform(),
        AffineTransform(loc=constraint.mmin, scale=constraint.mmax - constraint.mmin),
    ])


@biject_to.register(mass_ratio_mass_sandwich)
def _transform_to_mass_sandwich(constraint):
    r"""Bijection from the unconstrained space onto a mass-ratio mass sandwich.

    Registered with :func:`~numpyro.distributions.transforms.biject_to` for
    :data:`~gwkokab.models.constraints.mass_ratio_mass_sandwich`. As for
    :data:`~gwkokab.models.constraints.mass_sandwich`, except that the affine step
    rescales only the first coordinate onto :math:`[m_\min, m_\max]` and leaves the
    second in :math:`[0, 1]`, where the mass ratio lives.

    Parameters
    ----------
    constraint : _MassRatioMassSandwichConstraint
        The constraint being bijected to; supplies the mass bounds.

    Returns
    -------
    ComposeTransform
        The bijection onto the constrained space.
    """
    return ComposeTransform([
        AbsTransform(),
        OrderedTransform(),
        PowerTransform(-1.0),
        SigmoidTransform(),
        AffineTransform(
            loc=jnp.array([constraint.mmin, 0.0]),
            scale=jnp.array([constraint.mmax - constraint.mmin, 1.0]),
        ),
    ])
