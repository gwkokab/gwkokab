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

from .abstractmodel import AbstractModel


class AbstractMassModel(AbstractModel):
    @staticmethod
    def add_error(x: Array, scale: float = 0.2, size: int = 4000) -> Numeric:
        """Adds error to the masses of the binaries according to the section 3 of the following paper.
        https://doi.org/10.1093/mnras/stw2883

        Converts the masses(m1,m2) to chirp mass and adds error to it. Then converts back to masses.

        Parameters
        ----------
        x : ArrayLike
            True value of m_1
        y : ArrayLike
        scale : float
            True value of m_2
        size : int, optional
            number of points after adding error, by default 4000

        Returns
        -------
        Arrays
            array1: m_1 with error
            array2: m_2 with error
        """

        y = x[1]  # m2
        x = x[0]  # m1
        # Compute the M_chirp from component masses
        Mc_true = (x * y) ** (3 / 5) / (x + y) ** (1 / 5)
        # Compute symmetric mass ratio (Eta) from component masses
        eta_true = (x * y) / ((x + y) ** 2)
        # print("Mc_true: ", Mc_true, "eta_true: ", eta_true)

        # Adding Errors in True M chirp and Eta according to paper

        # row = 9 / jnp.power(uniform(), 1.0 / 3.0)  # SNR
        U = Uniform(low=0, high=1)
        row = 9 / jnp.power(U.rvs(shape=()), 1.0 / 3.0)  # SNR
        v_PN_param = (jnp.pi * Mc_true * 20.0 * lalsimutils.MsunInSec) ** (1.0 / 3.0)  # 'v' parameter
        v_PN_param_max = jnp.asarray(0.2)
        # print([v_PN_param, v_PN_param_max])
        v_PN_param, v_PN_param_max = jnp.broadcast_arrays(v_PN_param, v_PN_param_max)
        v_PN_param = jnp.min(jnp.array([v_PN_param, v_PN_param_max]), axis=0)
        snr_fac = row / 15.0
        ln_mc_error_pseudo_fisher = (
            1.5 * 0.3 * (v_PN_param / v_PN_param_max) ** 7.0 / snr_fac
        )  # this ignores range due to redshift / distance, based on a low-order esti

        #   print("ln_mc_error_pseudo_fisher: ", ln_mc_error_pseudo_fisher)
        snr_fac, ln_mc_error_pseudo_fisher = jnp.broadcast_arrays(snr_fac, ln_mc_error_pseudo_fisher)
        alpha = jnp.min(
            jnp.array([0.07 / snr_fac, ln_mc_error_pseudo_fisher])
        )  # Percentage error, we are adding 10%.  Note already accounts for SNR effects
        # print("alpha: ", alpha)
        # Shifting the mean using standard normal distribution
        std_norm = Normal(loc=0, scale=1)
        ro = std_norm.rvs(shape=())
        rop = std_norm.rvs(shape=())

        # print("ro: ", ro, "rop: ", rop)

        # std_dev = 1
        # print("std_dev: ", std_dev)

        # Changing the width across the shifted mean
        # r = Normal(0, std_dev).rvs(shape=(size,))
        # rp = Normal(0, std_dev).rvs(shape=(size,))
        r = std_norm.rvs(shape=(size,))
        rp = std_norm.rvs(shape=(size,))

        # Defining the relation
        Mc = Mc_true * (1.0 + alpha * (ro + r))
        eta = eta_true * (1.0 + 0.03 * (8 / row) * (rop + rp))

        # print("Mc_array: ", Mc, "eta_array: ", eta)

        # Compute component masses from Mc, eta. Returns m1 >= m2
        # etaV = Array(1 - (4 * eta), dtype=float)
        etaV = 1 - (4 * eta)
        if isinstance(eta, float):
            if etaV < 0:
                etaV = 0
                etaV_sqrt = 0
            else:
                etaV_sqrt = jnp.sqrt(etaV)
        else:
            indx_ok = etaV >= 0
            # etaV_sqrt = jnp.zeros(len(etaV), dtype=float)
            # etaV_sqrt[indx_ok] = jnp.sqrt(etaV[indx_ok])
            etaV_sqrt = jnp.where(indx_ok, jnp.sqrt(etaV), 0)

        m1 = 0.5 * Mc * (eta**-0.6) * (1.0 + etaV_sqrt)
        m2 = 0.5 * Mc * (eta**-0.6) * (1.0 - etaV_sqrt)

        return jnp.column_stack([m1, m2])
