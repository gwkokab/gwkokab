# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import chex
import numpy as np
from absl.testing import parameterized
from astropy import cosmology as astro_cosmo
from jax import numpy as jnp

from gwkokab.cosmology import Cosmology, PLANCK_2015_Cosmology, PLANCK_2018_Cosmology


class TestCosmology(parameterized.TestCase):
    @chex.variants(  # pyright: ignore
        with_jit=True,  # test case failing
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters(
        ("PLANCK_2015_Cosmology", PLANCK_2015_Cosmology),
        ("PLANCK_2018_Cosmology", PLANCK_2018_Cosmology),
    )
    def test_z_to_z(self, cosmo: Cosmology):
        @self.variant
        def _z_to_z(z):
            DL = cosmo.z_to_DL(z)
            z_ = cosmo.DL_to_z(DL)
            return z_

        z = jnp.linspace(0, 4, 100)

        assert jnp.allclose(z, _z_to_z(z), atol=1e-3)

    @chex.variants(  # pyright: ignore
        with_jit=True,  # test case failing
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters(
        ("PLANCK_2015_Cosmology", PLANCK_2015_Cosmology),
        ("PLANCK_2018_Cosmology", PLANCK_2018_Cosmology),
    )
    def test_DL_to_DL(self, cosmo: Cosmology):
        @self.variant
        def _DL_to_DL(DL):
            z_ = cosmo.DL_to_z(DL)
            DL = cosmo.z_to_DL(z_)
            return DL

        DL = jnp.linspace(0, 4, 100)

        assert jnp.allclose(DL, _DL_to_DL(DL), atol=1e-3)


def test_luminosity_distance_with_astropy():
    planck15: astro_cosmo.LambdaCDM = astro_cosmo.Planck15
    planck18: astro_cosmo.LambdaCDM = astro_cosmo.Planck18

    z = np.linspace(0, 4, 100)

    np.testing.assert_allclose(
        planck15.luminosity_distance(z).value,
        PLANCK_2015_Cosmology.z_to_DL(z),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        planck18.luminosity_distance(z).value,
        PLANCK_2018_Cosmology.z_to_DL(z),
        rtol=1e-2,
    )


# def test_differential_comoving_volume_with_astropy():
#     planck15: astro_cosmo.FlatLambdaCDM = astro_cosmo.Planck15
#     planck18: astro_cosmo.FlatLambdaCDM = astro_cosmo.Planck18

#     z = np.linspace(0, 2, 100)

#     np.testing.assert_allclose(
#         4 * np.pi * planck15.differential_comoving_volume(z).value,
#         PLANCK_2015_Cosmology.dVcdz(z),
#         rtol=1e-3,
#     )
#     np.testing.assert_allclose(
#         4 * np.pi * planck18.differential_comoving_volume(z).value,
#         PLANCK_2018_Cosmology.dVcdz(z),
#         rtol=1e-3,
#     )
