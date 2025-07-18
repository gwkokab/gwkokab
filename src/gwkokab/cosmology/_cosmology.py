# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) 2023 Farr Out Lab
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# adapted from code written by Reed Essick included in the gw-distributions package at:
# https://git.ligo.org/reed.essick/gw-distributions/-/blob/master/gwdistributions/utils/cosmology.py

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import quadax
from jaxtyping import ArrayLike

from gwkokab.constants import C, DEFAULT_DZ


class Cosmology(eqx.Module):
    """Cosmology class for flat ΛCDM universe in SI units (internally), with Gpc/Gpc³
    output utilities for population inference.
    """

    _c_over_Ho: ArrayLike = eqx.field(init=False)
    _Dc: ArrayLike = eqx.field(init=False)
    _Ho: ArrayLike = eqx.field(init=False)
    _OmegaKappa: ArrayLike = eqx.field(init=False)
    _OmegaLambda: ArrayLike = eqx.field(init=False)
    _OmegaMatter: ArrayLike = eqx.field(init=False)
    _OmegaRadiation: ArrayLike = eqx.field(init=False)
    _z: ArrayLike = eqx.field(init=False)
    Vc: ArrayLike = eqx.field(init=False)

    def __init__(
        self,
        Ho: ArrayLike,
        omega_matter: ArrayLike,
        omega_radiation: ArrayLike,
        omega_lambda: ArrayLike,
        max_z: float = 2.3,
        dz: float = DEFAULT_DZ,
    ):
        self._Ho = Ho  # SI units: s^-1
        self._c_over_Ho = C / self._Ho  # (km/s) / (km/s/Mpc)
        self._OmegaMatter = omega_matter
        self._OmegaRadiation = omega_radiation
        self._OmegaLambda = omega_lambda
        self._OmegaKappa = 1.0 - (
            self._OmegaMatter + self._OmegaRadiation + self._OmegaLambda
        )
        assert jnp.isclose(self._OmegaKappa, 0.0, atol=1e-10), (
            "Only flat cosmologies are supported (Ω_k ≈ 0)."
        )

        self._z = jnp.arange(0.0, max_z, dz)
        self._Dc = quadax.cumulative_trapezoid(
            self.dDcdz(self._z), x=self._z, initial=0.0, axis=0
        )
        self.Vc = quadax.cumulative_trapezoid(
            self.dVcdz(self._z, self._Dc), x=self._z, initial=0.0, axis=0
        )

    # --------- E(z), distance derivatives ---------
    def z_to_E(self, z: ArrayLike) -> ArrayLike:
        zp1 = 1.0 + z
        return jnp.sqrt(
            self._OmegaLambda
            + self._OmegaKappa * zp1**2
            + self._OmegaMatter * zp1**3
            + self._OmegaRadiation * zp1**4
        )

    def dDcdz(self, z: ArrayLike) -> ArrayLike:
        return self._c_over_Ho / self.z_to_E(z)

    def dVcdz(self, z: ArrayLike, Dc: Optional[ArrayLike] = None) -> ArrayLike:
        return jnp.exp(self.logdVcdz(z, Dc=Dc))

    def logdVcdz(self, z: ArrayLike, Dc: Optional[ArrayLike] = None) -> ArrayLike:
        if Dc is None:
            Dc = self.z_to_Dc(z)
        return jnp.log(4 * jnp.pi) + 2 * jnp.log(Dc) + jnp.log(self.dDcdz(z))

    # --------- Interpolators ---------
    def z_to_Dc(self, z):
        return jnp.interp(z, self._z, self._Dc)

    def z_to_DL(self, z):
        return jnp.interp(z, self._z, self.DL)

    def DL_to_z(self, DL):
        return jnp.interp(DL, self.DL, self._z)

    # --------- Core property ---------
    @property
    def DL(self):
        return self._Dc * (1.0 + self._z)
