# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import chex
from absl.testing import parameterized
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

        z = jnp.linspace(0, 10, 100)

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

        DL = jnp.linspace(0, 10, 100)

        assert jnp.allclose(DL, _DL_to_DL(DL), atol=1e-3)
