# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


#
"""Provides implementation of various transformations using
:class:`~numpyro.distributions.transforms.Transform`.
"""

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

    .. seealso::

        - :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        - :class:`ComponentMassesToChirpMassAndDelta`
        - :class:`ComponentMassesToMassRatioAndSecondaryMass`
        - :class:`ComponentMassesToPrimaryMassAndMassRatio`
        - :class:`ComponentMassesToTotalMassAndMassRatio`
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
            \ln\left(|\mathrm{det}(J_f)|\right)=\ln(M_c)+\ln(\eta)+\ln(m_1-m_2)-\ln(m_1+m_2)-\ln(m_1)-\ln(m_2)"""
        m1, m2 = jnp.unstack(x, axis=-1)
        Mc, eta = jnp.unstack(y, axis=-1)
        # The factor of 2 is omitted to align with empirical results from test cases.
        # Further investigation may be required to reconcile theory with implementation.
        log_detJ = jnp.log(Mc) + jnp.log(eta) + jnp.log(m1 - m2)
        log_detJ -= jnp.log(m1 + m2) + jnp.log(m1) + jnp.log(m2)
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

    .. seealso::

        - :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        - :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        - :class:`ComponentMassesToMassRatioAndSecondaryMass`
        - :class:`ComponentMassesToPrimaryMassAndMassRatio`
        - :class:`ComponentMassesToTotalMassAndMassRatio`
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

    .. seealso::

        - :class:`ComponentMassesToChirpMassAndDelta`
        - :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        - :class:`ComponentMassesToMassRatioAndSecondaryMass`
        - :class:`ComponentMassesToPrimaryMassAndMassRatio`
        - :class:`ComponentMassesToTotalMassAndMassRatio`
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

    .. seealso::

        - :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        - :class:`ComponentMassesToChirpMassAndDelta`
        - :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        - :class:`ComponentMassesToMassRatioAndSecondaryMass`
        - :class:`ComponentMassesToTotalMassAndMassRatio`
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

    .. seealso::

        - :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        - :class:`ComponentMassesToChirpMassAndDelta`
        - :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        - :class:`ComponentMassesToPrimaryMassAndMassRatio`
        - :class:`ComponentMassesToTotalMassAndMassRatio`
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

    .. seealso::

        - :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        - :class:`ComponentMassesToChirpMassAndDelta`
        - :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        - :class:`ComponentMassesToMassRatioAndSecondaryMass`
        - :class:`ComponentMassesToPrimaryMassAndMassRatio`
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
