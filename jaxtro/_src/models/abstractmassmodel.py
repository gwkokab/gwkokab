#  Copyright 2023 The Jaxtro Authors
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
from jaxampler.rvs import Normal, Uniform
from jaxampler.typing import Numeric
from jaxtyping import Array
from RIFT import lalsimutils

from ..utils.misc import chirp_mass, symmetric_mass_ratio
from .abstractmodel import AbstractModel


class AbstractMassModel(AbstractModel):
    @staticmethod
    def add_error(x: Array, scale: float = 1, size: int = 5000) -> Numeric:
        """
        Adds error to the masses of the binaries according to the section 3 of the following paper.
        https://doi.org/10.1093/mnras/stw2883

        Converts the masses(m1,m2) to chirp mass and adds error to it. Then converts back to masses.
        """

        m2 = x[1]  # m2
        m1 = x[0]  # m1
        # Compute the M_chirp from component masses
        Mc_true = chirp_mass(m1, m2)
        # Compute symmetric mass ratio (Eta) from component masses
        eta_true = symmetric_mass_ratio(m1, m2)

        # Adding Errors in True M chirp and Eta according to paper
        U = Uniform(low=0, high=1)
        rho = 9 / jnp.power(U.rvs(shape=()), 1.0 / 3.0)  # SNR

        # TODO: Post Newtonian paramters?
        v_PN_param = (jnp.pi * Mc_true * 20.0 * lalsimutils.MsunInSec) ** (1.0 / 3.0)  # 'v' parameter
        v_PN_param_max = jnp.asarray(0.2)

        v_PN_param, v_PN_param_max = jnp.broadcast_arrays(v_PN_param, v_PN_param_max)
        v_PN_param = jnp.min(jnp.array([v_PN_param, v_PN_param_max]), axis=0)
        snr_fac = rho / 15.0

        # this ignores range due to redshift / distance, based on a low-order esti
        ln_mc_error_pseudo_fisher = 1.5 * 0.3 * (v_PN_param / v_PN_param_max) ** 7.0 / snr_fac

        snr_fac, ln_mc_error_pseudo_fisher = jnp.broadcast_arrays(snr_fac, ln_mc_error_pseudo_fisher)
        # Percentage error, we are adding 10%. Note already accounts for SNR effects
        alpha = jnp.min(jnp.array([0.07 / snr_fac, ln_mc_error_pseudo_fisher]))

        # Shifting the mean using standard normal distribution
        std_norm = Normal(loc=0, scale=1)
        ro = std_norm.rvs(shape=())
        rop = std_norm.rvs(shape=())

        r = std_norm.rvs(shape=(size,))
        rp = std_norm.rvs(shape=(size,))

        alpha = jnp.where(
            eta_true >= 0.1,
            0.01,
            alpha,
        )

        alpha = jnp.where(
            eta_true >= 0.05,
            0.03,
            alpha,
        )

        alpha = jnp.where(
            0.05 > eta_true,
            0.1,
            alpha,
        )

        # Defining the relation
        Mc = Mc_true * (1.0 + alpha * (12 / rho) * (ro + r))
        eta = eta_true * (1.0 + 0.03 * (12 / rho) * (rop + rp))

        # Compute component masses from Mc, eta. Returns m1 >= m2
        etaV = 1.0 - (4.0 * eta)
        if isinstance(eta, float):
            if etaV < 0:
                etaV_sqrt = 0
            else:
                etaV_sqrt = jnp.sqrt(etaV)
        else:
            indx_ok = etaV >= 0
            # etaV_sqrt = jnp.zeros(len(etaV), dtype=float)
            # etaV_sqrt[indx_ok] = jnp.sqrt(etaV[indx_ok])
            etaV_sqrt = jnp.where(indx_ok, jnp.sqrt(etaV), 0)

        m1_final = 0.5 * Mc * (eta**-0.6) * (1.0 + etaV_sqrt)
        m2_final = 0.5 * Mc * (eta**-0.6) * (1.0 - etaV_sqrt)
        m1_final: Array = jnp.where((0.25 >= eta) & (eta >= 0.01), m1_final, jnp.zeros_like(m1_final))
        m2_final: Array = jnp.where((0.25 >= eta) & (eta >= 0.01), m2_final, jnp.zeros_like(m2_final))

        m1m2 = jnp.column_stack([m1_final, m2_final])

        return m1m2
