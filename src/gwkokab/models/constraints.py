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


from jax import numpy as jnp
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
]


class _MassSandwichConstraint(Constraint):
    r"""Constrain mass values to lie within a sandwiched interval.

    .. math::
        m_{\text{min}} \leq m_2 \leq m_1 \leq m_{\text{max}}
    """

    event_dim = 1

    def __init__(self, mmin: float, mmax: float):
        """
        :param mmin: Minimum mass.
        :param mmax: Maximum mass.
        """
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, x):
        m1 = x[..., 0]
        m2 = x[..., 1]
        mask = 0.0 < self.mmin
        mask &= self.mmin <= m2
        mask &= m2 <= m1
        mask &= m1 <= self.mmax
        return mask

    def tree_flatten(self):
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MassSandwichConstraint):
            return False
        return jnp.array_equal(self.mmin, other.mmin) & jnp.array_equal(
            self.mmax, other.mmax
        )


class _MassRatioMassSandwichConstraint(Constraint):
    r"""Constrain primary mass to lie within a sandwiched interval
    and the mass ratio to lie within a given interval. This is a
    transformed version of the :class:`_MassSandwichConstraint`.
    
    .. math::
        \begin{align*}
            m_{\text{min}}             & \leq m_1 \leq m_{\max} \\
            \frac{m_{\text{min}}}{m_1} & \leq q   \leq 1
        \end{align*}
    """

    event_dim = 1

    def __init__(self, mmin: float, mmax: float):
        r"""
        :param mmin: Minimum mass.
        :param mmax: Maximum mass.
        """
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, x):
        m1 = x[..., 0]
        m2 = x[..., 1] * m1
        mask = 0.0 < self.mmin
        mask &= self.mmin <= m2
        mask &= m2 <= m1
        mask &= m1 <= self.mmax
        return mask

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

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _IncreasingVector)


class _DecreasingVector(_SingletonConstraint):
    r"""Constrain values to be decreasing, i.e. :math:`\forall i<j, x_i \geq x_j`."""

    event_dim = 1

    def __call__(self, x):
        return jnp.all(x[..., 1:] <= x[..., :-1], axis=-1)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _DecreasingVector)


class _StrictlyIncreasingVector(_SingletonConstraint):
    r"""Constrain values to be strictly increasing, i.e. :math:`\forall i<j, x_i < x_j`."""

    event_dim = 1

    def __call__(self, x):
        return jnp.all(x[..., 1:] > x[..., :-1], axis=-1)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _StrictlyIncreasingVector)


class _StrictlyDecreasingVector(_SingletonConstraint):
    r"""Constrain values to be strictly decreasing, i.e. :math:`\forall i<j,x_i > x_j`."""

    event_dim = 1

    def __call__(self, x):
        return jnp.all(x[..., 1:] < x[..., :-1], axis=-1)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _StrictlyDecreasingVector)


class _PositiveIncreasingVector(_SingletonConstraint):
    r"""Constrain values to be positive and increasing, i.e. :math:`\forall i<j, x_i \leq x_j`."""

    event_dim = 1

    def __call__(self, x):
        return increasing_vector.check(x) & independent(positive, 1).check(x)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _PositiveIncreasingVector)


class _PositiveDecreasingVector(_SingletonConstraint):
    r"""Constrain values to be positive and decreasing, i.e. :math:`\forall i<j, x_i \geq x_j`."""

    event_dim = 1

    def __call__(self, x):
        return decreasing_vector.check(x) & independent(positive, 1).check(x)

    def tree_flatten(self):
        return (), ((), dict())

    def __eq__(self, other):
        return isinstance(other, _PositiveDecreasingVector)


mass_sandwich = _MassSandwichConstraint
mass_ratio_mass_sandwich = _MassRatioMassSandwichConstraint
increasing_vector = _IncreasingVector()
decreasing_vector = _DecreasingVector()
strictly_increasing_vector = _StrictlyIncreasingVector()
strictly_decreasing_vector = _StrictlyDecreasingVector()
positive_increasing_vector = _PositiveIncreasingVector()
positive_decreasing_vector = _PositiveDecreasingVector()
