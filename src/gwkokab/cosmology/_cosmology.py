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

import jax.numpy as jnp
from jax.lax import fori_loop

from ..constants import C_SI


DEFAULT_DZ = 1e-3  # should be good enough for most numeric integrations we want to do


class Cosmology(object):
    """A class for basic cosmology calculations using jax.

    See https://arxiv.org/pdf/astro-ph/9905116 for description of parameters and functions.

    .. note:: We work in SI units throughout, though distances are specified in Mpc.
    """

    def __init__(
        self, Ho, omega_matter, omega_radiation, omega_lambda, max_z=10.0, dz=DEFAULT_DZ
    ):
        self.Ho = Ho
        self.c_over_Ho = C_SI / self.Ho
        self.OmegaMatter = omega_matter
        self.OmegaRadiation = omega_radiation
        self.OmegaLambda = omega_lambda
        self.OmegaKappa = 1.0 - (
            self.OmegaMatter + self.OmegaRadiation + self.OmegaLambda
        )
        assert (
            self.OmegaKappa == 0
        ), "we only implement flat cosmologies! OmegaKappa must be 0"

        self.extend(max_z, dz=dz)

    @property
    def DL(self):
        return self.Dc * (1 + self.z)

    def update(self, i, x):
        z = x[0]
        dz = z[1] - z[0]
        Dc = x[1]
        Vc = x[2]

        dDcdz = self.dDcdz(z[i])
        dVcdz = self.dVcdz(z[i], Dc[i])
        new_dDcdz = self.dDcdz(z[i] + dz)
        Dc = Dc.at[i + 1].set(Dc[i] + 0.5 * (dDcdz + new_dDcdz) * dz)

        new_dVcdz = self.dVcdz(z[i] + dz, Dc[i + 1])
        Vc = Vc.at[i + 1].set(Vc[i] + 0.5 * (dVcdz + new_dVcdz) * dz)

        return jnp.array([z, Dc, Vc])

    def extend(self, max_z, dz=DEFAULT_DZ):
        """Integrate to solve for distance measures."""

        self.z = jnp.arange(0, max_z, dz)
        Dc = jnp.zeros_like(self.z)
        Vc = jnp.zeros_like(self.z)

        X = jnp.array([self.z, Dc, Vc])
        extended_X = fori_loop(0, self.z.shape[0] - 1, self.update, X)
        # extended_X = lax.scan(self.update, X, jnp.arange(0, self.z.shape[0] - 1))
        self.Dc = extended_X[1]
        self.Vc = extended_X[2]

    def z_to_E(self, z):
        """Returns E(z) = sqrt(OmegaLambda + OmegaKappa*(1+z)**2 +
        OmegaMatter*(1+z)**3 + OmegaRadiation*(1+z)**4)"""
        one_plus_z = 1.0 + z
        return (
            self.OmegaLambda
            + self.OmegaKappa * one_plus_z**2
            + self.OmegaMatter * one_plus_z**3
            + self.OmegaRadiation * one_plus_z**4
        ) ** 0.5

    def dDcdz(self, z):
        """Returns (c/Ho)/E(z)"""
        dDc = self.c_over_Ho / self.z_to_E(z)
        return dDc

    def dVcdz(self, z, Dc=None):
        """Return dVc/dz."""
        return jnp.exp(self.logdVcdz(z, Dc=Dc))

    def logdVcdz(self, z, Dc=None):
        """Return ln(dVc/dz), useful when constructing probability distributions
        without overflow errors."""
        if Dc is None:
            Dc = self.z_to_Dc(z)
        return jnp.log(4 * jnp.pi) + 2 * jnp.log(Dc) + jnp.log(self.dDcdz(z))

    def z_to_Dc(self, z):
        """Return Dc for each z specified."""
        return jnp.interp(z, self.z, self.Dc)

    def DL_to_z(self, DL):
        """Returns redshifts for each DL specified."""
        return jnp.interp(DL, self.DL, self.z)

    def z_to_DL(self, z):
        """Returns luminosity distance at the specified redshifts."""
        return jnp.interp(z, self.z, self.DL)
