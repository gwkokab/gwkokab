#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

from jax import numpy as jnp
from jax.random import normal, uniform

from ..typing import Numeric
from ..utils.misc import chirp_mass, get_key, symmetric_mass_ratio
from .abstractmodel import AbstractModel


class AbstractMassModel(AbstractModel):
    def add_error(self, x: Numeric, scale: float, size: int) -> Numeric:
        """
        Adds error to the masses of the binaries according to the section 3 of the following paper.
        https://doi.org/10.1093/mnras/stw2883

        Converts the masses(m1,m2) to chirp mass and adds error to it. Then converts back to masses.
        """
        m1 = x[0]
        m2 = x[1]

        r0 = normal(key=get_key(), shape=(), dtype=x.dtype)
        r0p = normal(key=get_key(), shape=(), dtype=x.dtype)
        r = normal(key=get_key(), shape=(size,), dtype=x.dtype)
        rp = normal(key=get_key(), shape=(size,), dtype=x.dtype)

        rho = 8.0 * jnp.power(
            uniform(
                key=get_key(),
                shape=(size,),
                dtype=x.dtype,
                minval=0.0,
                maxval=1.0,
            ),
            -1.0 / 3.0,
        )

        Mc_true = chirp_mass(m1, m2)
        eta_true = symmetric_mass_ratio(m1, m2)

        alpha = jnp.zeros_like(r)
        alpha = jnp.where(eta_true >= 0.1, 0.01, alpha)
        alpha = jnp.where((0.1 > eta_true) & (eta_true >= 0.05), 0.03, alpha)
        alpha = jnp.where(0.05 > eta_true, 0.1, alpha)

        twelve_over_rho = 12.0 / rho

        Mc = Mc_true * (1.0 + alpha * twelve_over_rho * (r0 + r))
        eta = eta_true * (1.0 + 0.03 * twelve_over_rho * (r0p + rp))

        mask = 0.25 >= eta
        mask &= eta >= 0.01

        mtot = Mc * jnp.power(eta, -0.6)
        m1m2 = eta * jnp.power(mtot, 2)

        m2_final = 0.5 * (mtot - jnp.sqrt(mtot**2 - 4 * m1m2))
        m1_final = 0.5 * (mtot + jnp.sqrt(mtot**2 - 4 * m1m2))

        m1_final = jnp.where(mask, m1_final, jnp.nan)
        m2_final = jnp.where(mask, m2_final, jnp.nan)

        return jnp.column_stack([m1_final, m2_final])
