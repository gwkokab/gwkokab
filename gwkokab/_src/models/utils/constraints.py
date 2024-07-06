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


from jax import numpy as jnp
from numpyro.distributions import constraints


__all__ = [
    "chirp_mass_symmetric_mass_ratio_sandwich",
    "greater_than_equal_to",
    "less_than_equal_to",
    "mass_ratio_mass_sandwich",
    "mass_sandwich",
]


class _GreaterThanEqualTo(constraints.Constraint):
    r"""Constrain values to be greater than or equal to a given value."""

    def __init__(self, lower_bound: float):
        r"""
        :param lower_bound: The lower bound.
        """
        self.lower_bound = lower_bound

    def __call__(self, x):
        return x >= self.lower_bound

    def tree_flatten(self):
        return (self.lower_bound,), (("lower_bound",), dict())


class _LessThanEqualTo(constraints.Constraint):
    r"""Constrain values to be less than or equal to a given value."""

    def __init__(self, upper_bound: float):
        """
        :param upper_bound: The upper bound.
        """
        self.upper_bound = upper_bound

    def __call__(self, x):
        return x <= self.upper_bound

    def tree_flatten(self):
        return (self.upper_bound,), (("upper_bound",), dict())


class _MassSandwichConstraint(constraints.Constraint):
    r"""Constrain mass values to lie within a sandwiched interval.

    $$m_{\text{min}} \leq m_2 \leq m_1 \leq m_{\text{max}}$$
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
        mask = jnp.less_equal(self.mmin, m2)
        mask = jnp.logical_and(mask, jnp.less_equal(m2, m1))
        mask = jnp.logical_and(mask, jnp.less_equal(m1, self.mmax))
        return mask

    def tree_flatten(self):
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())


class _MassRationMassSandwichConstraint(constraints.Constraint):
    r"""Constrain primary mass to lie within a sandwiched interval
    and the mass ratio to lie within a given interval. This is a
    transformed version of the :class:`_MassSandwichConstraint`.
    
    $$
        \begin{align*}
            m_{\text{min}}             & \leq m_1 \leq m_{\max} \\
            \frac{m_{\text{min}}}{m_1} & \leq q   \leq 1
        \end{align*}
    $$
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
        m2 = jnp.multiply(x[..., 1], m1)
        mask = jnp.less_equal(self.mmin, m2)
        mask = jnp.logical_and(mask, jnp.less_equal(m2, m1))
        mask = jnp.logical_and(mask, jnp.less_equal(m1, self.mmax))
        return mask

    def tree_flatten(self):
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())


class _ChirpMassSymmetricMassRatioConstraint(constraints.Constraint):
    r"""Constraint for chirp mass and symmetric mass ratio.
    
    .. math ::
        \begin{align*}
            0 < M_c \\
            0 < \eta \leq \frac{1}{4}
        \end{align*}
    """

    event_dim = 1

    def __call__(self, x):
        Mc = x[..., 0]
        eta = x[..., 1]
        mask = jnp.greater(Mc, 0.0)
        mask = jnp.logical_and(mask, jnp.greater(eta, 0.0))
        mask = jnp.logical_and(mask, jnp.less_equal(eta, 0.25))
        return mask

    def tree_flatten(self):
        return (), ((), dict())


chirp_mass_symmetric_mass_ratio_sandwich = _ChirpMassSymmetricMassRatioConstraint()
greater_than_equal_to = _GreaterThanEqualTo
less_than_equal_to = _LessThanEqualTo
mass_sandwich = _MassSandwichConstraint
mass_ratio_mass_sandwich = _MassRationMassSandwichConstraint
