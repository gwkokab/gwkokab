#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

from typing_extensions import Tuple

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions.constraints import Constraint
from numpyro.distributions.transforms import (
    biject_to,
    ComposeTransform,
    ExpTransform,
    OrderedTransform,
    Transform,
)

from .constraints import mass_ratio_mass_sandwich, mass_sandwich


class PrimaryMassMassRatioToComponentMassesTransform(Transform):
    r"""Transforms a primary mass and mass ratio to component masses.

    .. math::
        f: (m_1, q)\to (m_1, m_1q)

    .. math::
        det(J) = m_1
    """

    def __init__(self, mmin: Array, mmax: Array) -> None:
        self.mmin = mmin
        self.mmax = mmax

    @property
    def domain(self) -> Constraint:
        return mass_ratio_mass_sandwich(mmin=self.mmin, mmax=self.mmax)

    @property
    def codomain(self) -> Constraint:
        return mass_sandwich(mmin=self.mmin, mmax=self.mmax)

    def __call__(self, x: Array):
        m1 = x[..., 0]
        q = x[..., 1]
        m2 = jnp.multiply(m1, q)
        m1m2 = jnp.stack((m1, m2), axis=-1)
        return m1m2

    def _inverse(self, y: Array):
        m1 = y[..., 0]
        m2 = y[..., 1]
        q = jnp.divide(m2, m1)
        return jnp.stack((m1, q), axis=-1)

    def log_abs_det_jacobian(self, x: Array, y: Array, intermediates=None):
        # log(|det(J)|) = log(m1)
        return jnp.log(x[..., 0])

    def forward_shape(self, shape) -> Tuple[int, ...]:
        return shape

    def inverse_shape(self, shape) -> Tuple[int, ...]:
        return shape

    def tree_flatten(self):
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())

    def __eq__(self, other):
        if not isinstance(other, PrimaryMassMassRatioToComponentMassesTransform):
            return False
        return self.domain == other.domain


@biject_to.register(mass_ratio_mass_sandwich)
def _mass_ratio_mass_sandwich_bijector(constraint):
    return ComposeTransform(
        [
            PrimaryMassMassRatioToComponentMassesTransform(
                mmin=constraint.mmin, mmax=constraint.mmax
            ),
            OrderedTransform(),
            ExpTransform(),
        ]
    )
