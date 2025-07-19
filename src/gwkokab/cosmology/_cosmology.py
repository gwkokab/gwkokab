# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import quadax
from jaxtyping import ArrayLike

from gwkokab.constants import C, DEFAULT_DZ


class Cosmology(eqx.Module):
    """Flat ΛCDM cosmology with Mpc-based units for distances and comoving volumes."""

    _c_over_Ho: ArrayLike = eqx.field(init=False)
    _Ho: ArrayLike = eqx.field(init=False)
    _OmegaKappa: ArrayLike = eqx.field(init=False)
    _OmegaLambda: ArrayLike = eqx.field(init=False)
    _OmegaMatter: ArrayLike = eqx.field(init=False)
    _OmegaRadiation: ArrayLike = eqx.field(init=False)
    _z: ArrayLike = eqx.field(init=False)
    _Dc: ArrayLike = eqx.field(init=False)
    Vc: ArrayLike = eqx.field(init=False)

    def __init__(
        self,
        Ho: ArrayLike,
        omega_matter: ArrayLike,
        omega_radiation: ArrayLike,
        omega_lambda: ArrayLike,
        max_z: float = 4.0,
        dz: float = DEFAULT_DZ,
    ):
        self._Ho = Ho  # Hubble constant in m/s/Mpc
        self._c_over_Ho = C / Ho  # Units: Mpc= (m/s)/(m/s/Mpc)
        self._OmegaMatter = omega_matter
        self._OmegaRadiation = omega_radiation
        self._OmegaLambda = omega_lambda
        self._OmegaKappa = 1.0 - (
            self._OmegaMatter + self._OmegaRadiation + self._OmegaLambda
        )

        assert jnp.isclose(self._OmegaKappa, 0.0, atol=1e-10), (
            "Only flat cosmologies are supported (Ω_k ≈ 0)."
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
        zp1 = 1.0 + z
        return jnp.sqrt(
            self._OmegaLambda
            + self._OmegaKappa * zp1**2
            + self._OmegaMatter * zp1**3
            + self._OmegaRadiation * zp1**4
        )

    def dDcdz(self, z: ArrayLike) -> ArrayLike:
        return self._c_over_Ho / self.z_to_E(z)

    def logdVcdz(self, z: ArrayLike, Dc: Optional[ArrayLike] = None) -> ArrayLike:
        if Dc is None:
            Dc = self.z_to_Dc(z)
        return jnp.log(4 * jnp.pi) + 2 * jnp.log(Dc) + jnp.log(self.dDcdz(z))  # Mpc³

    def dVcdz(self, z: ArrayLike, Dc: Optional[ArrayLike] = None) -> ArrayLike:
        return jnp.exp(self.logdVcdz(z, Dc=Dc))

    # --------- Interpolators ---------
    def z_to_Dc(self, z: ArrayLike) -> ArrayLike:
        """Fast JAX-safe interpolation of comoving distance."""
        return jnp.interp(z, self._z, self._Dc)

    def z_to_DL(self, z: ArrayLike) -> ArrayLike:
        """Luminosity distance in Mpc."""
        return self.z_to_Dc(z) * (1.0 + z)

    def DL_to_z(self, DL: ArrayLike) -> ArrayLike:
        """Approximate inversion DL -> z using precomputed grid."""
        return jnp.interp(DL, self.DL, self._z)

    # --------- Properties ---------
    @property
    def DL(self) -> ArrayLike:
        return self._Dc * (1.0 + self._z)

    @property
    def z(self) -> ArrayLike:
        return self._z

    @property
    def Dc(self) -> ArrayLike:
        return self._Dc

    # --------- Utility validation ---------
    def assert_z_range_covers(self, z: ArrayLike, name="z"):
        """Raise if any redshift exceeds grid limit (run outside JAX tracing)."""
        max_z = float(jnp.max(z))
        if max_z > float(self._z[-1]):
            raise ValueError(
                f"{name} contains z={max_z:.3f} which exceeds cosmology grid limit z={float(self._z[-1]):.3f}. "
                f"Please re-instantiate with a higher max_z."
            )
