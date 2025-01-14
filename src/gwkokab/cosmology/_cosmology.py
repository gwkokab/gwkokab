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
from jaxtyping import ArrayLike

from gwkokab.constants import C_SI


DEFAULT_DZ = 1e-3  # should be good enough for most numeric integrations we want to do


class Cosmology(object):
    """A class for basic cosmology calculations using jax.

    See https://arxiv.org/pdf/astro-ph/9905116 for description of parameters and functions.

    .. note::

        We work in SI units throughout, though distances are specified in Mpc.
    ```
    """

    def __init__(
        self,
        Ho: ArrayLike,
        omega_matter: ArrayLike,
        omega_radiation: ArrayLike,
        omega_lambda: ArrayLike,
        max_z: float = 10.0,
        dz: float = DEFAULT_DZ,
    ):
        r"""Initialize the cosmology object.

        .. math::
            \Omega_{\Lambda}+\Omega_{\kappa}+\Omega_{m}+\Omega_{r}=1

        Parameters
        ----------
        Ho : ArrayLike
            Hubble constant
        omega_matter : ArrayLike
            Matter density :math:`\Omega_m`
        omega_radiation : ArrayLike
            Radiation density :math:`\Omega_r`
        omega_lambda : ArrayLike
            Dark energy density :math:`\Omega_{\Lambda}`
        max_z : float, optional
            Maximum redshift to integrate to, by default 10.0
        dz : float, optional
            Step size for integration, by default 1e-3
        """
        self._Ho = Ho
        self._c_over_Ho = C_SI / self._Ho
        self._OmegaMatter = omega_matter
        self._OmegaRadiation = omega_radiation
        self._OmegaLambda = omega_lambda
        self._OmegaKappa = 1.0 - (
            self._OmegaMatter + self._OmegaRadiation + self._OmegaLambda
        )
        assert self._OmegaKappa == 0, (
            "we only implement flat cosmologies! OmegaKappa must be 0"
        )

        self.extend(max_z, dz=dz)

    @property
    def DL(self):
        return self._Dc * (1.0 + self._z)

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
        """Integrate to solve for distance measures.

        Parameters
        ----------
        max_z : Array
            Maximum redshift to integrate to.
        dz : int, optional
            Step size for integration. Default is 1e-3.
        """
        self._z = jnp.arange(0, max_z, dz)
        Dc = jnp.zeros_like(self._z)
        Vc = jnp.zeros_like(self._z)

        X = jnp.array([self._z, Dc, Vc])
        extended_X = fori_loop(0, self._z.shape[0] - 1, self.update, X)
        # extended_X = lax.scan(self.update, X, jnp.arange(0, self.z.shape[0] - 1))
        self._Dc = extended_X[1]
        self.Vc = extended_X[2]

    def z_to_E(self, z: ArrayLike) -> ArrayLike:
        r"""Dimensionless Hubble parameter.

        .. math::
            \mathrm{E}(z)=\frac{\mathrm{H}(z)}{H_o}=\sqrt{\Omega_{\Lambda}+\Omega_{\kappa}(1+z)^2+\Omega_{m}(1+z)^3+\Omega_{r}(1+z)^4}

        Parameters
        ----------
        z : ArrayLike
            Redshift

        Returns
        -------
        ArrayLike
            Dimensionless Hubble parameter
        """
        zp1 = 1.0 + z
        return (
            self._OmegaLambda
            + self._OmegaKappa * zp1**2
            + self._OmegaMatter * zp1**3
            + self._OmegaRadiation * zp1**4
        ) ** 0.5

    def dDcdz(self, z: ArrayLike) -> ArrayLike:
        r"""

        .. math::
            \frac{\mathrm{d}D_c}{\mathrm{d}z}=\frac{c}{H_o\mathrm{E}(z)}


        Parameters
        ----------
        z : ArrayLike
            Redshifts

        Returns
        -------
        ArrayLike
        """
        dDc = self._c_over_Ho / self.z_to_E(z)
        return dDc

    def dVcdz(self, z: ArrayLike, Dc=None) -> ArrayLike:
        r"""Calculate the comoving volume element per unit redshift.

        .. math::
            \frac{dV_c}{dz}=4\pi D_c^2\frac{dD_c}{dz}

        Parameters
        ----------
        z : ArrayLike
            Redshifts
        Dc : _type_, optional
            Distance to the source, by default None

        Returns
        -------
        ArrayLike
            Comoving volume element per unit redshift
        """
        return jnp.exp(self.logdVcdz(z, Dc=Dc))

    def logdVcdz(self, z: ArrayLike, Dc: ArrayLike = None) -> ArrayLike:
        r"""Return :math:`\ln(\frac{dV_c}{dz})`. Useful when constructing probability
        distributions without overflow errors.

        Parameters
        ----------
        z : ArrayLike
            Redshifts
        Dc : ArrayLike, optional
            Distance to the source, by default None

        Returns
        -------
        ArrayLike
            :math:`\ln(\frac{dV_c}{dz})`
        """
        if Dc is None:
            Dc = self.z_to_Dc(z)
        return jnp.log(4 * jnp.pi) + 2 * jnp.log(Dc) + jnp.log(self.dDcdz(z))

    def z_to_Dc(self, z):
        """Return Dc for each z specified."""
        return jnp.interp(z, self._z, self._Dc)

    def DL_to_z(self, DL):
        """Returns redshifts for each DL specified."""
        return jnp.interp(DL, self.DL, self._z)

    def z_to_DL(self, z):
        """Returns luminosity distance at the specified redshifts."""
        return jnp.interp(z, self._z, self.DL)
