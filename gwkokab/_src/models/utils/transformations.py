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
from numpyro.distributions.transforms import Transform

from .constraints import _MassRationMassSandwichConstraint, mass_sandwich


class PrimaryMassMassRatioToComponentMassTransform(Transform):
    r"""Transforms a primary mass and mass ratio to component masses.

    .. math::
        f: (m_1, q)\to (m_1, m_1q)

    .. math::
        det(J) = m_1
    """

    def __init__(self, domain: Constraint) -> None:
        assert isinstance(
            domain, _MassRationMassSandwichConstraint
        ), "Domain must be a mass ratio sandwich constraint"
        self.domain = domain

    @property
    def codomain(self) -> Constraint:
        mmin = self.domain.mmin
        mmax = self.domain.mmax
        return mass_sandwich(mmin=mmin, mmax=mmax)

    def __call__(self, x: Array):
        m1 = x[..., 0]
        q = x[..., 1]
        m1m2 = jnp.column_stack((m1, jnp.multiply(m1, q)))
        return m1m2

    def _inverse(self, y: Array):
        m1 = y[..., 0]
        m2 = y[..., 1]
        q = jnp.divide(m2, m1)
        return jnp.column_stack((m1, q))

    def log_abs_det_jacobian(self, x: Array, y: Array, intermediates=None):
        # log(|det(J)|) = log(m1)
        return jnp.log(x[..., 0])

    def forward_shape(self, shape) -> Tuple[int, ...]:
        return shape

    def inverse_shape(self, shape) -> Tuple[int, ...]:
        return shape

    def tree_flatten(self):
        return (self.domain,), (("domain",), dict())

    def __eq__(self, other):
        if not isinstance(other, PrimaryMassMassRatioToComponentMassTransform):
            return False
        return self.domain == other.domain

        return self.domain == other.domain
