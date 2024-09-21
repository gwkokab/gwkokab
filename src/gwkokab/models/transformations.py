# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import constraints
from numpyro.distributions.transforms import (
    AbsTransform,
    biject_to,
    ComposeTransform,
    OrderedTransform,
    PowerTransform,
    ReshapeTransform,
    Transform,
)

from ..utils.transformations import (
    chirp_mass,
    delta_m,
    m1_m2_to_Mc_eta,
    m1_q_to_m2,
    m2_q_to_m1,
    M_q_to_m1_m2,
    mass_ratio,
    Mc_delta_to_m1_m2,
    Mc_eta_to_m1_m2,
    symmetric_mass_ratio_to_delta_m,
    total_mass,
)
from .constraints import (
    decreasing_vector,
    increasing_vector,
    positive_decreasing_vector,
    positive_increasing_vector,
    strictly_decreasing_vector,
    strictly_increasing_vector,
    unique_intervals,
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
        \mathrm{det}(J_f) = m_1
    """

    domain = unique_intervals((0.0, 0.0), (jnp.inf, 1.0))
    codomain = positive_decreasing_vector

    def __call__(self, x: Array):
        m1 = x[..., 0]
        q = x[..., 1]
        m2 = jnp.multiply(m1, q)
        m1m2 = jnp.stack((m1, m2), axis=-1)
        return m1m2

    def _inverse(self, y: Array):
        m1 = y[..., 0]
        m2 = y[..., 1]
        q = mass_ratio(m2=m2, m1=m1)
        m1q = jnp.stack((m1, q), axis=-1)
        return m1q

    def log_abs_det_jacobian(self, x: Array, y: Array, intermediates=None):
        abs_m1 = jnp.abs(x[..., 0])
        return jnp.log(abs_m1)

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

    .. math::
        \mathrm{det}(J_f)=\frac{\delta_m\mathcal{M}}{m_1+m_2}

    .. seealso::

        :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        :class:`ComponentMassesToChirpMassAndDelta`
        :class:`ComponentMassesToMassRatioAndSecondaryMass`
        :class:`ComponentMassesToPrimaryMassAndMassRatio`
        :class:`ComponentMassesToTotalMassAndMassRatio`
    """

    domain = positive_decreasing_vector
    codomain = unique_intervals((0.0, 0.0), (jnp.inf, 0.25))

    def __call__(self, x):
        m1 = x[..., 0]
        m2 = x[..., 1]
        Mc, eta = m1_m2_to_Mc_eta(m1=m1, m2=m2)
        return jnp.stack((Mc, eta), axis=-1)

    def _inverse(self, y):
        Mc = y[..., 0]
        eta = y[..., 1]
        m1, m2 = Mc_eta_to_m1_m2(Mc=Mc, eta=eta)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        m1 = x[..., 0]
        m2 = x[..., 1]
        log_Mc = jnp.log(y[..., 0])
        delta_m = symmetric_mass_ratio_to_delta_m(eta=y[..., 1])
        log_delta_m = jnp.log(delta_m)
        log_M = jnp.log(total_mass(m1=m1, m2=m2))

        log_detJ = jnp.add(log_delta_m, log_Mc)
        log_detJ = jnp.subtract(log_detJ, log_M)
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

    .. math::
        \mathrm{det}(J_f) = -\frac{\delta}{2}
    """

    domain = constraints.interval(0.0, 1.0)
    codomain = constraints.interval(0.0, 0.25)

    def __call__(self, x):
        delta_sq = jnp.square(x)
        return jnp.multiply(jnp.subtract(1.0, delta_sq), 0.25)

    def _inverse(self, y):
        eta4 = jnp.multiply(4, y)
        return jnp.sqrt(jnp.subtract(1, eta4))

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.subtract(jnp.log(x), jnp.log(2.0))

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, type(self))


class ComponentMassesToChirpMassAndDelta(Transform):
    r"""Transforms component masses to chirp mass and delta.

    .. math::
        f: (m_1, m_2) \to (M_c, \delta)

    .. math::
        M_c = \frac{(m_1m_2)^{3/5}}{(m_1+m_2)^{1/5}}

    .. math::
        \delta = \frac{m_1-m_2}{m1+m_2}

    .. math::
        \mathrm{det}(J_f) = -\frac{2M_c}{(m_1+m_2)^2}

    .. seealso::

        :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        :class:`ComponentMassesToMassRatioAndSecondaryMass`
        :class:`ComponentMassesToPrimaryMassAndMassRatio`
        :class:`ComponentMassesToTotalMassAndMassRatio`
    """

    domain = positive_decreasing_vector
    codomain = unique_intervals((0.0, 0.0), (jnp.inf, 1.0))

    def __call__(self, x):
        m1 = x[..., 0]
        m2 = x[..., 1]
        Mc = chirp_mass(m1=m1, m2=m2)
        delta = delta_m(m1=m1, m2=m2)
        return jnp.stack((Mc, delta), axis=-1)

    def _inverse(self, y):
        Mc = y[..., 0]
        delta = y[..., 1]
        m1, m2 = Mc_delta_to_m1_m2(Mc=Mc, delta=delta)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        m1 = x[..., 0]
        m2 = x[..., 1]
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
    r"""Transforms source mass and redshift to detected mass and redshift."""

    domain = unique_intervals((0.0, 0.0), (jnp.inf, jnp.inf))
    codomain = unique_intervals((0.0, 0.0), (jnp.inf, jnp.inf))

    def __call__(self, x):
        m_source = x[..., 0]
        z = x[..., 1]
        m_detected = m_source * (1 + z)
        return jnp.stack((m_detected, z), axis=-1)

    def _inverse(self, y):
        m_detected = y[..., 0]
        z = y[..., 1]
        m_source = m_detected / (1 + z)
        return jnp.stack((m_source, z), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        z = x[..., 1]
        return jnp.log(1 + z)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, type(self))


class ComponentMassesAndRedshiftToDetectedMassAndRedshift(Transform):
    r"""Transforms component masses and redshift to detected masses and redshift.

    .. seealso::

        :class:`ComponentMassesToChirpMassAndDelta`
        :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        :class:`ComponentMassesToMassRatioAndSecondaryMass`
        :class:`ComponentMassesToPrimaryMassAndMassRatio`
        :class:`ComponentMassesToTotalMassAndMassRatio`
    """

    domain = unique_intervals((0.0, 0.0, 0.0), (jnp.inf, jnp.inf, jnp.inf))
    codomain = unique_intervals((0.0, 0.0, 0.0), (jnp.inf, jnp.inf, jnp.inf))

    def __call__(self, x):
        m1_source = x[..., 0]
        m2_source = x[..., 1]
        z = x[..., 2]
        m1_detected = m1_source * (1 + z)
        m2_detected = m2_source * (1 + z)
        return jnp.stack((m1_detected, m2_detected, z), axis=-1)

    def _inverse(self, y):
        m1_detected = y[..., 0]
        m2_detected = y[..., 1]
        z = y[..., 2]
        m1_source = m1_detected / (1 + z)
        m2_source = m2_detected / (1 + z)
        return jnp.stack((m1_source, m2_source, z), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        z = x[..., 1]
        return 2 * jnp.log(1 + z)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, type(self))


class ComponentMassesToPrimaryMassAndMassRatio(Transform):
    r"""Transforms component masses and redshift to primary mass and mass ratio.

    .. seealso::

        :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        :class:`ComponentMassesToChirpMassAndDelta`
        :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        :class:`ComponentMassesToMassRatioAndSecondaryMass`
        :class:`ComponentMassesToTotalMassAndMassRatio`
    """

    domain = positive_decreasing_vector
    codomain = unique_intervals((0.0, 0.0), (jnp.inf, 1.0))

    def __call__(self, x):
        m1 = x[..., 0]
        m2 = x[..., 1]
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((m1, q), axis=-1)

    def _inverse(self, y):
        m1 = y[..., 0]
        q = y[..., 1]
        m2 = m1_q_to_m2(m1=m1, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return -jnp.log(x[..., 0])

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, type(self))


class ComponentMassesToMassRatioAndSecondaryMass(Transform):
    r"""Transforms component masses and redshift to mass ratio and secondary mass.

    .. seealso::

        :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        :class:`ComponentMassesToChirpMassAndDelta`
        :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        :class:`ComponentMassesToPrimaryMassAndMassRatio`
        :class:`ComponentMassesToTotalMassAndMassRatio`
    """

    domain = positive_decreasing_vector
    codomain = unique_intervals((0.0, 0.0), (1.0, jnp.inf))

    def __call__(self, x):
        m1 = x[..., 0]
        m2 = x[..., 1]
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((q, m2), axis=-1)

    def _inverse(self, y):
        q = y[..., 0]
        m2 = y[..., 1]
        m1 = m2_q_to_m1(m2=m2, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        m1 = x[..., 0]
        q = y[..., 0]
        return jnp.log(q) - jnp.log(m1)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, type(self))


class ComponentMassesToTotalMassAndMassRatio(Transform):
    r"""Transforms component masses to total mass and mass ratio.

    .. seealso::

        :class:`ComponentMassesAndRedshiftToDetectedMassAndRedshift`
        :class:`ComponentMassesToChirpMassAndDelta`
        :class:`ComponentMassesToChirpMassAndSymmetricMassRatio`
        :class:`ComponentMassesToMassRatioAndSecondaryMass`
        :class:`ComponentMassesToPrimaryMassAndMassRatio`
    """

    domain = positive_decreasing_vector
    codomain = unique_intervals((0.0, 0.0), (jnp.inf, 1.0))

    def __call__(self, x):
        m1 = x[..., 0]
        m2 = x[..., 1]
        M = total_mass(m1=m1, m2=m2)
        q = mass_ratio(m1=m1, m2=m2)
        return jnp.stack((M, q), axis=-1)

    def _inverse(self, y):
        M = y[..., 0]
        q = y[..., 1]
        m1, m2 = M_q_to_m1_m2(M=M, q=q)
        return jnp.stack((m1, m2), axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        m2 = x[..., 1]
        q = y[..., 1]
        return jnp.log(m2) + jnp.log(q) + jnp.log(1 + q)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, type(self))


@biject_to.register(unique_intervals)
def _transform_to_unique_intervals(constraint):
    return ReshapeTransform(
        forward_shape=constraint.lower_bounds.shape,
        inverse_shape=constraint.lower_bounds.shape,
    )


@biject_to.register(type(positive_decreasing_vector))
@biject_to.register(type(decreasing_vector))
@biject_to.register(type(strictly_decreasing_vector))
def _transform_to_positive_ordered_vector(constraint):
    """I want things to be positive and in decreasing order."""
    return ComposeTransform(
        [
            AbsTransform(),
            OrderedTransform(),
            PowerTransform(-1.0),
        ]
    )


@biject_to.register(type(positive_increasing_vector))
@biject_to.register(type(increasing_vector))
@biject_to.register(type(strictly_increasing_vector))
def _transform_to_positive_ordered_vector(constraint):
    """I want things to be positive and in decreasing order."""
    return ComposeTransform(
        [
            AbsTransform(),
            OrderedTransform(),
        ]
    )
