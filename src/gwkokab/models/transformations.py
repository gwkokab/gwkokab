# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


#
"""Provides implementation of various transformations using
:class:`~numpyro.distributions.transforms.Transform`.
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
        self.domain = all_constraint(
            [t.domain for t in self.transforms], self.event_slices
        )
        self.codomain = all_constraint(
            [t.codomain for t in self.transforms], self.event_slices
        )

    def __call__(self, x: Array) -> Array:
        y_slices = []
        for transform, event_slice in zip(self.transforms, self.event_slices):
            if isinstance(event_slice, int):
                x_slice = jax.lax.dynamic_index_in_dim(
                    x, event_slice, axis=-1, keepdims=False
                )
            else:
                x_slice = jax.lax.dynamic_slice_in_dim(
                    x,
                    event_slice[0],
                    event_slice[1] - event_slice[0],
                    axis=-1,
                )
            y_slice = transform(x_slice)
            y_slices.append(y_slice)
        return jnp.column_stack(y_slices)

    def _inverse(self, y: Array) -> Array:
        x_slices = []
        for transform, event_slice in zip(self.transforms, self.event_slices):
            if isinstance(event_slice, int):
                y_slice = jax.lax.dynamic_index_in_dim(
                    y, event_slice, axis=-1, keepdims=False
                )
            else:
                y_slice = jax.lax.dynamic_slice_in_dim(
                    y,
                    event_slice[0],
                    event_slice[1] - event_slice[0],
                    axis=-1,
                )
            x_slice = transform.inv(y_slice)
            x_slices.append(x_slice)
        return jnp.column_stack(x_slices)

    def log_abs_det_jacobian(self, x: Array, y: Array, intermediates=None):
        log_detJ = 0.0
        for transform, event_slice in zip(self.transforms, self.event_slices):
            if isinstance(event_slice, int):
                x_slice = jax.lax.dynamic_index_in_dim(
                    x, event_slice, axis=-1, keepdims=False
                )
                y_slice = jax.lax.dynamic_index_in_dim(
                    y, event_slice, axis=-1, keepdims=False
                )
            else:
                start_idx, end_idx = event_slice
                x_slice = jax.lax.dynamic_slice_in_dim(
                    x,
                    start_idx,
                    end_idx - start_idx,
                    axis=-1,
                )
                y_slice = jax.lax.dynamic_slice_in_dim(
                    y,
                    start_idx,
                    end_idx - start_idx,
                    axis=-1,
                )
            log_detJ_slice = transform.log_abs_det_jacobian(
                x_slice, y_slice, intermediates
            )
            log_detJ += log_detJ_slice
        return log_detJ

    def tree_flatten(self):
        return (self.transforms,), (
            ("transforms",),
            {"event_slices": self.event_slices},
        )

    def __eq__(self, value):
        if not isinstance(value, BlockTransform):
            return False
        return self.event_slices == value.event_slices and all(
            t1 == t2 for t1, t2 in zip(self.transforms, value.transforms)
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
        m1, q = jnp.unstack(x, axis=-1)
        m2 = jnp.multiply(m1, q)
        m1m2 = jnp.stack((m1, m2), axis=-1)
        return m1m2

    def _inverse(self, y: Array):
        m1, m2 = jnp.unstack(y, axis=-1)
        q = mass_ratio(m2=m2, m1=m1)
        m1q = jnp.stack((m1, q), axis=-1)
        return m1q

    def log_abs_det_jacobian(self, x: Array, y: Array, intermediates=None):
        r"""
        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(|m_1|)
        """
        m1 = x[..., 0]
        return jnp.log(jnp.abs(m1))

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
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
        m1, m2 = jnp.unstack(x, axis=-1)
        Mc = chirp_mass(m1=m1, m2=m2)
        eta = symmetric_mass_ratio(m1=m1, m2=m2)
        return jnp.stack((Mc, eta), axis=-1)

    def _inverse(self, y):
        Mc, eta = jnp.unstack(y, axis=-1)
        m1, m2 = Mc_eta_to_m1_m2(Mc=Mc, eta=eta)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""
        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right)=\frac{6}{5}\ln(\eta)+\frac{1}{2}\ln(1-4\eta)-\ln(M_c)
        """
        Mc, eta = jnp.unstack(y, axis=-1)
        log_detJ = 1.2 * jnp.log(eta) + 0.5 * jnp.log1p(-4.0 * eta) - jnp.log(Mc)
        return log_detJ

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
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
        delta_sq = jnp.square(x)
        return jnp.multiply(jnp.subtract(1.0, delta_sq), 0.25)

    def _inverse(self, y):
        eta4 = jnp.multiply(4, y)
        return jnp.sqrt(jnp.subtract(1, eta4))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""
        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(\delta) - \ln(2)
        """
        return jnp.subtract(jnp.log(x), jnp.log(2.0))

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
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
        m1 = x[..., 0]
        m2 = x[..., 1]
        Mc = chirp_mass(m1=m1, m2=m2)
        delta = delta_m(m1=m1, m2=m2)
        return jnp.stack((Mc, delta), axis=-1)

    def _inverse(self, y):
        Mc, delta = jnp.unstack(y, axis=-1)
        eta = delta_m_to_symmetric_mass_ratio(delta_m=delta)
        m1, m2 = Mc_eta_to_m1_m2(Mc=Mc, eta=eta)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""
        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(2M_c) - 2\ln(m_1+m_2)
        """
        m1, m2 = jnp.unstack(x, axis=-1)
        M = total_mass(m1=m1, m2=m2)
        log_Mc = jnp.log(y[..., 0])
        log_detJ = jnp.log(2.0)
        log_detJ = jnp.add(log_detJ, log_Mc)
        log_detJ = jnp.add(log_detJ, jnp.multiply(-2.0, jnp.log(M)))
        return log_detJ

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
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
        m_source, z = jnp.unstack(x, axis=-1)
        m_detected = m_source * (1 + z)
        return jnp.stack((m_detected, z), axis=-1)

    def _inverse(self, y):
        m_detected, z = jnp.unstack(y, axis=-1)
        m_source = m_detected / (1 + z)
        return jnp.stack((m_source, z), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""
        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(1+z)
        """
        z = x[..., 1]
        return jnp.log1p(z)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
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
        m1_source, m2_source, z = jnp.unstack(x, axis=-1)
        m1_detected = m1_source * (1 + z)
        m2_detected = m2_source * (1 + z)
        return jnp.stack((m1_detected, m2_detected, z), axis=-1)

    def _inverse(self, y):
        m1_detected, m2_detected, z = jnp.unstack(y, axis=-1)
        m1_source = m1_detected / (1 + z)
        m2_source = m2_detected / (1 + z)
        return jnp.stack((m1_source, m2_source, z), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""
        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = 2\ln(1+z)
        """
        z = x[..., 2]
        return 2 * jnp.log1p(z)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
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
        m1, m2 = jnp.unstack(x, axis=-1)
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((m1, q), axis=-1)

    def _inverse(self, y):
        m1, q = jnp.unstack(y, axis=-1)
        m2 = m1_q_to_m2(m1=m1, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        m1 = x[..., 0]
        return -jnp.log(jnp.abs(m1))

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
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
        m1, m2 = jnp.unstack(x, axis=-1)
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((q, m2), axis=-1)

    def _inverse(self, y):
        q, m2 = jnp.unstack(y, axis=-1)
        m1 = m2_q_to_m1(m2=m2, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        r"""
        .. math::
            \ln\left(|\mathrm{det}(J_f)|\right) = \ln(q) - \ln(m_1)
        """
        m1 = x[..., 0]
        q = y[..., 0]
        return jnp.log(q) - jnp.log(m1)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
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
        m1, m2 = jnp.unstack(x, axis=-1)
        M = total_mass(m1=m1, m2=m2)
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((M, q), axis=-1)

    def _inverse(self, y):
        M, q = jnp.unstack(y, axis=-1)
        safe_q = jnp.where(q == -1.0, 1.0, q)
        m1 = jnp.where(q == -1.0, jnp.inf, M / (1 + safe_q))
        m2 = m1_q_to_m2(m1=m1, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        m1 = x[..., 0]
        q = y[..., 1]
        return jnp.log(1 + q) - jnp.log(m1)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, type(self))


@biject_to.register(type(positive_decreasing_vector))
@biject_to.register(type(decreasing_vector))
@biject_to.register(type(strictly_decreasing_vector))
def _transform_to_positive_ordered_vector(constraint):
    """Ensure things to be positive and in decreasing order."""
    return ComposeTransform([AbsTransform(), OrderedTransform(), PowerTransform(-1.0)])


@biject_to.register(type(positive_increasing_vector))
@biject_to.register(type(increasing_vector))
@biject_to.register(type(strictly_increasing_vector))
def _transform_to_positive_ordered_vector(constraint):
    """Ensure things to be positive and in decreasing order."""
    return ComposeTransform([AbsTransform(), OrderedTransform()])


@biject_to.register(mass_sandwich)
def _transform_to_mass_sandwich(constraint):
    return ComposeTransform(
        [
            AbsTransform(),
            OrderedTransform(),
            PowerTransform(-1.0),
            SigmoidTransform(),
            AffineTransform(
                loc=constraint.mmin, scale=constraint.mmax - constraint.mmin
            ),
        ]
    )


@biject_to.register(mass_ratio_mass_sandwich)
def _transform_to_mass_sandwich(constraint):
    return ComposeTransform(
        [
            AbsTransform(),
            OrderedTransform(),
            PowerTransform(-1.0),
            SigmoidTransform(),
            AffineTransform(
                loc=jnp.array([constraint.mmin, 0.0]),
                scale=jnp.array([constraint.mmax - constraint.mmin, 1.0]),
            ),
        ]
    )
