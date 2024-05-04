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

import jax
from jax import numpy as jnp

from ...typing import Numeric


@jax.jit
def smoothing_kernel(mass: Numeric, mass_min: Numeric, delta: Numeric) -> Numeric:
    r"""See equation B4 in [Population Properties of Compact Objects from the Second
    LIGO-Virgo Gravitational-Wave Transient Catalog](https://arxiv.org/abs/2010.14533).
    
    $$
        S(m\mid m_{\min}, \delta) = \begin{cases}
            0 & \text{if } m < m_{\min}, \\
            \left[\displaystyle 1 + \exp\left(\frac{\delta}{m}
            +\frac{\delta}{m-\delta}\right)\right]^{-1}
            & \text{if } m_{\min} \leq m < m_{\min} + \delta, \\
            1 & \text{if } m \geq m_{\min} + \delta
        \end{cases}
    $$

    :param mass: mass of the primary black hole
    :param mass_min: minimum mass of the primary black hole
    :param delta: small mass difference
    :return: smoothing kernel value
    """

    @jax.jit
    def sub_kernel(mass_: Numeric, delta_m: Numeric) -> Numeric:
        r"""See equation B5 in `Population Properties of Compact Objects from the Second
        LIGO-Virgo Gravitational-Wave Transient Catalog <https://arxiv.org/abs/2010.14533>__`.

        :param mass_: mass of the primary black hole
        :param delta_m: small mass difference
        :return: smoothing function
        """
        return jnp.exp((delta_m / mass_) + (delta_m / (mass_ - delta_m)))

    conditions = [
        mass < mass_min,
        (mass_min <= mass) & (mass < mass_min + delta),
        mass >= mass_min + delta,
    ]
    choices = [
        jnp.zeros_like(mass),
        jnp.reciprocal(1 + sub_kernel(mass - mass_min, delta)),
        jnp.ones_like(mass),
    ]
    return jnp.select(conditions, choices)
