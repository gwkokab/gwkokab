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

from functools import partial

from jax import jit, numpy as jnp
from jaxtyping import Array, Real


@partial(jit, inline=True)
def smoothing_kernel(
    mass: Array | Real, mass_min: Array | Real, delta: Array | Real
) -> Array | Real:
    r"""See equation B4 in `Population Properties of Compact Objects from the
    Second LIGO-Virgo Gravitational-Wave Transient Catalog 
    <https://arxiv.org/abs/2010.14533>`_.
    
    .. math::
        S(m\mid m_{\min}, \delta) = \begin{cases}
            0 & \text{if } m < m_{\min}, \\
            \left[\displaystyle 1 + \exp\left(\frac{\delta}{m}
            +\frac{\delta}{m-\delta}\right)\right]^{-1}
            & \text{if } m_{\min} \leq m < m_{\min} + \delta, \\
            1 & \text{if } m \geq m_{\min} + \delta
        \end{cases}

    :param mass: mass of the primary black hole
    :param mass_min: minimum mass of the primary black hole
    :param delta: small mass difference
    :return: smoothing kernel value
    """
    conditions = [
        mass < mass_min,
        (mass_min <= mass) & (mass < mass_min + delta),
    ]
    choices = [
        jnp.zeros_like(mass),
        jnp.reciprocal(
            1
            + jnp.exp((delta / (mass - mass_min)) + (delta / (mass - mass_min - delta)))
        ),
    ]
    return jnp.select(conditions, choices, default=jnp.ones_like(mass))
