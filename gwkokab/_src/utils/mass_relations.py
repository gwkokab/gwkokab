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

from jax import jit, lax

from ..typing import Numeric


@jit
def chirp_mass(m1: Numeric, m2: Numeric) -> Numeric:
    r"""
    .. math::
        M_c = \frac{(m_1 m_2)^{\frac{3}{5}}}{(m_1 + m_2)^{\frac{1}{5}}}

    :param m1: mass 1
    :param m2: mass 2
    :return:
    """
    return lax.mul(lax.pow(lax.mul(m1, m2), 0.6), lax.pow(lax.add(m1, m2), -0.2))


@jit
def symmetric_mass_ratio(m1: Numeric, m2: Numeric) -> Numeric:
    r"""
    .. math::
        \eta = \frac{m_1 m_2}{(m_1 + m_2)^2}

    :param m1: mass 1
    :param m2: mass 2
    :return:
    """
    return lax.mul(lax.mul(m1, m2), lax.pow(lax.add(m1, m2), -2))


@jit
def reduced_mass(m1: Numeric, m2: Numeric) -> Numeric:
    """
    .. math::
        M_r = \frac{m_1 m_2}{m_1 + m_2}

    :param m1: mass 1
    :param m2: mass 2
    :return:
    """
    return lax.div(lax.mul(m1, m2), lax.add(m1, m2))


@jit
def mass_ratio(m1: Numeric, m2: Numeric) -> Numeric:
    r"""
    .. math::
        q=\frac{m_1}{m_2}

    :param m1: mass 1
    :param m2: mass 2
    :return:
    """
    return lax.div(m1, m2)
