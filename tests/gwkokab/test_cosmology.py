# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import chex
import pytest
from absl.testing import parameterized
from jax import numpy as jnp

import gwkokab.cosmology as cosmo_mod
from gwkokab.cosmology import (
    Cosmology,
    PLANCK_2015_Cosmology,
    PLANCK_2018_Cosmology,
)


class TestCosmology(parameterized.TestCase):
    @chex.variants(  # pyright: ignore
        with_jit=True,  # test case failing
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters(
        ("PLANCK_2015_Cosmology", PLANCK_2015_Cosmology()),
        ("PLANCK_2018_Cosmology", PLANCK_2018_Cosmology()),
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
        ("PLANCK_2015_Cosmology", PLANCK_2015_Cosmology()),
        ("PLANCK_2018_Cosmology", PLANCK_2018_Cosmology()),
    )
    def test_DL_to_DL(self, cosmo: Cosmology):
        @self.variant
        def _DL_to_DL(DL):
            z_ = cosmo.DL_to_z(DL)
            DL = cosmo.z_to_DL(z_)
            return DL

        DL = jnp.linspace(0, 4, 100)

        assert jnp.allclose(DL, _DL_to_DL(DL), atol=1e-3)


class TestCosmologyFactories:
    """Tests for the individual factory functions."""

    def test_planck2015_values(self):
        res = cosmo_mod.PLANCK_2015_Cosmology()
        assert res.OmegaLambda == pytest.approx(0.6925)
        assert res.OmegaRadiation == 0.0

    def test_planck2018_values(self):
        res = cosmo_mod.PLANCK_2018_Cosmology()
        assert res.OmegaLambda == pytest.approx(0.69034)
        assert res.OmegaRadiation == 0.0


class TestDefaultCosmologyJIT(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_default_cosmology_under_jit(self):
        """Verify that default_cosmology works inside a JIT-compiled function."""

        @self.variant
        def compute_dist(z):
            # Calling the cached function inside JIT
            cosmo = cosmo_mod.default_cosmology()
            return cosmo.z_to_DL(z)

        z = jnp.array([0.1, 1.0, 2.0])
        # This will fail if JAX cannot handle the cached Equinox module
        # or if it tries to re-read the environment variable during transform
        result = compute_dist(z)

        assert result.shape == (3,)
        assert not jnp.any(jnp.isnan(result))


def test_registry_is_immutable():
    """Verify that the registry cannot be modified at runtime."""
    COSMOLOGY_REGISTRY = cosmo_mod._planck.COSMOLOGY_REGISTRY

    # Attempting to add a new key should raise a TypeError
    with pytest.raises(TypeError):
        COSMOLOGY_REGISTRY["NewCosmo"] = lambda: None

    # Attempting to delete a key should raise a TypeError
    with pytest.raises(TypeError):
        del COSMOLOGY_REGISTRY["Planck15"]
