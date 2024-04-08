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


from numpyro.distributions import constraints


class _MassSandwichConstraint(constraints.Constraint):
    r"""Constrain mass values to lie within a sandwiched interval.

    .. math:: m_{\text{min}} \leq m_2 \leq m_1 \leq m_{\text{max}}
    """

    def __init__(self, mmin, mmax):
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, x):
        return (self.mmin <= x[..., 1]) & (x[..., 1] <= x[..., 0]) & (x[..., 0] <= self.mmax)

    def tree_flatten(self):
        return (self.mmin, self.mmax), (("mmin", "mmax"), dict())


mass_sandwich = _MassSandwichConstraint
