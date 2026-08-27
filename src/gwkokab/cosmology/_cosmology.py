# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Flat :math:`\Lambda\mathrm{CDM}` cosmology written as an :class:`equinox.Module`.

:class:`Cosmology` precomputes comoving distance and comoving volume on a redshift grid
at construction time, so the redshift/distance conversions the population models need
are plain :func:`jax.numpy.interp` lookups -- differentiable, ``jit``-able, and free of
Python control flow.
"""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import quadax
from jaxtyping import ArrayLike

from gwkokab.constants import SPEED_OF_LIGHT


# TODO: A better alternative for Cosmology class that can work with JAX
class Cosmology(eqx.Module):
    r"""Flat :math:`\Lambda\mathrm{CDM}` cosmology in Mpc-based units.

    The dimensionless expansion rate is

    .. math::
        E(z) = \sqrt{\Omega_\Lambda + \Omega_\kappa (1+z)^2
               + \Omega_m (1+z)^3 + \Omega_r (1+z)^4}

    with the curvature density fixed by closure,
    :math:`\Omega_\kappa = 1 - (\Omega_m + \Omega_r + \Omega_\Lambda)`. Comoving distance
    and comoving volume are integrated once on a uniform redshift grid at construction
    time and thereafter interpolated.

    Parameters
    ----------
    Ho : ArrayLike
        Hubble constant, in m/s/Mpc.
    omega_matter : ArrayLike
        Matter density parameter :math:`\Omega_m`.
    omega_radiation : ArrayLike
        Radiation density parameter :math:`\Omega_r`.
    omega_lambda : ArrayLike
        Dark energy density parameter :math:`\Omega_\Lambda`.
    max_z : float, optional
        Upper end of the precomputed redshift grid. Defaults to ``4.0``; conversions
        beyond it are clamped by :func:`jax.numpy.interp`.
    dz : float, optional
        Spacing of the precomputed redshift grid. Defaults to ``1e-3``.
    """

    Ho: ArrayLike
    OmegaKappa: ArrayLike
    """Curvature density parameter, derived from the other three."""

    OmegaLambda: ArrayLike
    OmegaMatter: ArrayLike
    OmegaRadiation: ArrayLike
    _z: ArrayLike
    _Dc: ArrayLike
    Vc: ArrayLike
    r"""Comoving volume on the redshift grid, in :math:`\mathrm{Gpc}^3`."""

    def __init__(
        self,
        Ho: ArrayLike,
        omega_matter: ArrayLike,
        omega_radiation: ArrayLike,
        omega_lambda: ArrayLike,
        max_z: float = 4.0,
        dz: float = 1e-3,
    ) -> None:
        self.Ho = Ho  # Hubble constant in m/s/Mpc
        self.OmegaMatter = omega_matter
        self.OmegaRadiation = omega_radiation
        self.OmegaLambda = omega_lambda
        self.OmegaKappa = 1.0 - (
            self.OmegaMatter + self.OmegaRadiation + self.OmegaLambda
        )

        self._z = jnp.arange(0.0, max_z + dz, dz)
        self._Dc = quadax.cumulative_trapezoid(
            self.dDcdz(self._z), x=self._z, initial=0.0, axis=0
        )
        self.Vc = quadax.cumulative_trapezoid(
            self.dVcdz(self._z, self._Dc), x=self._z, initial=0.0, axis=0
        )

    # --------- Core functions ---------
    def z_to_E(self, z: ArrayLike) -> ArrayLike:
        """Dimensionless expansion rate :math:`E(z) = H(z)/H_0`.

        Parameters
        ----------
        z : ArrayLike
            Redshift.

        Returns
        -------
        ArrayLike
            The expansion rate :math:`E(z)`.
        """
        zp1 = 1.0 + z
        return jnp.sqrt(
            self.OmegaLambda
            + self.OmegaKappa * zp1**2
            + self.OmegaMatter * zp1**3
            + self.OmegaRadiation * zp1**4
        )

    def dDcdz(self, z: ArrayLike) -> ArrayLike:
        r"""Derivative of comoving distance with respect to redshift.

        .. math::
            \frac{\mathrm{d}D_c}{\mathrm{d}z} = \frac{c}{H_0 E(z)}

        Parameters
        ----------
        z : ArrayLike
            Redshift.

        Returns
        -------
        ArrayLike
            :math:`\mathrm{d}D_c/\mathrm{d}z`, in Mpc.
        """
        return (SPEED_OF_LIGHT / self.Ho) / self.z_to_E(z)

    def dDLdz(self, z: ArrayLike) -> ArrayLike:
        r"""Derivative of luminosity distance with respect to redshift.

        .. math::
            \frac{\mathrm{d}D_L}{\mathrm{d}z} = (1+z)\frac{\mathrm{d}D_c}{\mathrm{d}z} + D_c(z)

        Parameters
        ----------
        z : ArrayLike
            Redshift.

        Returns
        -------
        ArrayLike
            :math:`\mathrm{d}D_L/\mathrm{d}z`, in Mpc.
        """
        return self.dDcdz(z) * (1.0 + z) + self.z_to_Dc(z)

    def logdVcdz(self, z: ArrayLike, Dc: Optional[ArrayLike] = None) -> ArrayLike:
        r"""Log differential comoving volume, integrated over the sky.

        .. math::
            \log\frac{\mathrm{d}V_c}{\mathrm{d}z} =
            \log 4\pi + 2\log D_c(z) + \log\frac{\mathrm{d}D_c}{\mathrm{d}z} - 9\log 10

        The final term converts from :math:`\mathrm{Mpc}^3` to :math:`\mathrm{Gpc}^3`.

        Parameters
        ----------
        z : ArrayLike
            Redshift.
        Dc : Optional[ArrayLike]
            Comoving distance at ``z``, in Mpc. Defaults to :data:`None`, in which case it
            is interpolated with :meth:`z_to_Dc`. Pass it explicitly to avoid a redundant
            lookup, or to break the circularity during ``__init__``.

        Returns
        -------
        ArrayLike
            :math:`\log(\mathrm{d}V_c/\mathrm{d}z)`, with the volume in
            :math:`\mathrm{Gpc}^3`.
        """
        if Dc is None:
            Dc = self.z_to_Dc(z)
        return (
            jnp.log(4.0 * jnp.pi)  # Conversion of steradians
            - 9.0 * jnp.log(10.0)  # Conversion from Mpc^3 to Gpc^3
            + 2 * jnp.log(Dc)
            + jnp.log(self.dDcdz(z))
        )

    def dVcdz(self, z: ArrayLike, Dc: Optional[ArrayLike] = None) -> ArrayLike:
        r"""Differential comoving volume, integrated over the sky.

        Parameters
        ----------
        z : ArrayLike
            Redshift.
        Dc : Optional[ArrayLike]
            Comoving distance at ``z``, in Mpc. Defaults to :data:`None`, in which case it
            is interpolated with :meth:`z_to_Dc`.

        Returns
        -------
        ArrayLike
            :math:`\mathrm{d}V_c/\mathrm{d}z`, in :math:`\mathrm{Gpc}^3`.
        """
        return jnp.exp(self.logdVcdz(z, Dc=Dc))

    # --------- Interpolators ---------
    def z_to_Dc(self, z: ArrayLike) -> ArrayLike:
        """Comoving distance at a given redshift.

        Interpolated on the precomputed grid, so it is JAX-safe and differentiable.

        Parameters
        ----------
        z : ArrayLike
            Redshift.

        Returns
        -------
        ArrayLike
            Comoving distance, in Mpc.
        """
        return jnp.interp(z, self._z, self._Dc)

    def z_to_DL(self, z: ArrayLike) -> ArrayLike:
        """Luminosity distance at a given redshift.

        Parameters
        ----------
        z : ArrayLike
            Redshift.

        Returns
        -------
        ArrayLike
            Luminosity distance, in Mpc.
        """
        return jnp.interp(z, self._z, self.DL)

    def DL_to_z(self, DL: ArrayLike) -> ArrayLike:
        """Redshift at a given luminosity distance.

        Inverted by interpolating the precomputed :math:`D_L(z)` grid, which is monotone,
        so the inversion is well defined up to the grid resolution.

        Parameters
        ----------
        DL : ArrayLike
            Luminosity distance, in Mpc.

        Returns
        -------
        ArrayLike
            Redshift.
        """
        return jnp.interp(DL, self.DL, self._z)

    # --------- Properties ---------
    @property
    def DL(self) -> ArrayLike:
        """Luminosity distance on the precomputed redshift grid, in Mpc.

        Returns
        -------
        ArrayLike
            :math:`D_L = (1 + z) D_c` evaluated on :attr:`z`.
        """
        return self._Dc * (1.0 + self._z)

    @property
    def z(self) -> ArrayLike:
        """The precomputed redshift grid.

        Returns
        -------
        ArrayLike
            Redshifts from ``0`` to ``max_z`` in steps of ``dz``.
        """
        return self._z

    @property
    def Dc(self) -> ArrayLike:
        """Comoving distance on the precomputed redshift grid, in Mpc.

        Returns
        -------
        ArrayLike
            :math:`D_c` evaluated on :attr:`z`.
        """
        return self._Dc
