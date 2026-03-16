# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
from unittest.mock import patch

import chex
import numpy as np
import pytest
from absl.testing import parameterized
from astropy import cosmology as astro_cosmo
from jax import numpy as jnp

import gwkokab.cosmology as cosmo_mod
from gwkokab.cosmology import (
    Cosmology,
    PLANCK_2015_Cosmology,
    PLANCK_2018_Cosmology,
)
from gwkokab.utils.exceptions import LoggedValueError


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


def test_luminosity_distance_with_astropy():
    planck15: astro_cosmo.LambdaCDM = astro_cosmo.Planck15
    planck18: astro_cosmo.LambdaCDM = astro_cosmo.Planck18

    z = np.linspace(0, 4, 100)

    np.testing.assert_allclose(
        planck15.luminosity_distance(z).value,
        PLANCK_2015_Cosmology().z_to_DL(z),
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        planck18.luminosity_distance(z).value,
        PLANCK_2018_Cosmology().z_to_DL(z),
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


class TestDefaultCosmology:
    @pytest.fixture(autouse=True)
    def clear_cosmo_cache(self):
        """Clears the LRU cache before and after every test."""
        cosmo_mod.default_cosmology.cache_clear()
        yield
        cosmo_mod.default_cosmology.cache_clear()

    def test_default_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            result = cosmo_mod.default_cosmology()
            assert float(result.Ho) == 67740.0

    def test_planck18_selection(self):
        with patch.dict(os.environ, {"GWKOKAB_DEFAULT_COSMOLOGY": "Planck18"}):
            result = cosmo_mod.default_cosmology()
            assert float(result.Ho) == 67660.0

    def test_invalid_cosmology_raises_error(self):
        with patch.dict(os.environ, {"GWKOKAB_DEFAULT_COSMOLOGY": "InvalidName"}):
            with pytest.raises(
                LoggedValueError, match="Invalid or unavailable cosmology"
            ):
                cosmo_mod.default_cosmology()

    def test_jit_consistency(self):
        """Ensures the cache returns the exact same object instance."""
        obj1 = cosmo_mod.default_cosmology()
        obj2 = cosmo_mod.default_cosmology()
        assert obj1 is obj2


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
    @pytest.fixture(autouse=True)
    def clear_cosmo_cache(self):
        """Ensures a clean state for every JIT test."""
        cosmo_mod.default_cosmology.cache_clear()
        yield
        cosmo_mod.default_cosmology.cache_clear()

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

    @chex.variants(with_jit=True, without_jit=True)
    def test_cosmology_as_jit_argument(self):
        """Verify that the object returned by default_cosmology is a valid JIT
        argument.
        """

        @self.variant
        def get_h0(c: Cosmology):
            return c.Ho

        cosmo = cosmo_mod.default_cosmology()
        h0 = get_h0(cosmo)

        assert jnp.isclose(h0, cosmo.Ho)

    def test_jit_recompile_on_env_change(self):
        """Ensures that if the environment changes and the cache is cleared, the JITed
        result reflects the new values.
        """
        import jax

        # We define the JIT function inside the test to avoid
        # persistent JAX cache interference between test runs.
        @jax.jit
        def get_h0_jit():
            return cosmo_mod.default_cosmology().Ho

        # 1. Set to Planck15
        with patch.dict(os.environ, {"GWKOKAB_DEFAULT_COSMOLOGY": "Planck15"}):
            cosmo_mod.default_cosmology.cache_clear()
            h0_15 = get_h0_jit()
            assert jnp.isclose(h0_15, 67740.0)

        # 2. Set to Planck18
        # IMPORTANT: We define a NEW jit function or use a unique key.
        # In real code, users usually pass 'cosmo' as an argument to JIT.
        with patch.dict(os.environ, {"GWKOKAB_DEFAULT_COSMOLOGY": "Planck18"}):
            cosmo_mod.default_cosmology.cache_clear()

            # We redefine the JIT wrapper here so JAX creates a new cache entry
            @jax.jit
            def get_h0_jit_new():
                return cosmo_mod.default_cosmology().Ho

            h0_18 = get_h0_jit_new()
            assert jnp.isclose(h0_18, 67660.0)


def test_registry_is_immutable():
    """Verify that the registry cannot be modified at runtime."""
    COSMOLOGY_REGISTRY = cosmo_mod._planck.COSMOLOGY_REGISTRY

    # Attempting to add a new key should raise a TypeError
    with pytest.raises(TypeError):
        COSMOLOGY_REGISTRY["NewCosmo"] = lambda: None

    # Attempting to delete a key should raise a TypeError
    with pytest.raises(TypeError):
        del COSMOLOGY_REGISTRY["Planck15"]
