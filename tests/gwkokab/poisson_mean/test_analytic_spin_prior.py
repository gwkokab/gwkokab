# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from scipy.integrate import trapezoid

from gwkokab.poisson_mean._analytic_spin_prior import (
    chi_effective_prior_from_isotropic_spins,
    Li2,
    prior_chieff_chip_isotropic,
)


PISQ_6 = np.pi**2 / 6.0


@pytest.mark.parametrize(
    "z, expected",
    [
        (0.0, 0.0),
        (1.0, PISQ_6),
        (-1.0, -(np.pi**2) / 12.0),
        (0.5, np.pi**2 / 12.0 - np.log(2.0) ** 2 / 2.0),
    ],
)
def test_dilogarithm_known_values(z, expected):
    np.testing.assert_allclose(Li2(z), [expected], rtol=1e-12, atol=1e-14)


def test_dilogarithm_reflection_formula():
    r""":math:`\mathrm{Li}_2(z) + \mathrm{Li}_2(1 - z) = \pi^2/6 - \ln z \ln(1 - z)`."""
    z = np.linspace(0.05, 0.95, 19)
    np.testing.assert_allclose(
        Li2(z) + Li2(1.0 - z),
        PISQ_6 - np.log(z) * np.log1p(-z),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# p(chi_eff | q)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.15, 0.4, 0.65, 0.9])
def test_chi_effective_prior_is_normalised(q):
    chi_eff = np.linspace(-1.0, 1.0, 4001)
    pdf = chi_effective_prior_from_isotropic_spins(chi_eff, q)
    np.testing.assert_allclose(trapezoid(pdf, chi_eff), 1.0, rtol=1e-4)


@pytest.mark.parametrize("q", [0.15, 0.4, 0.65, 0.9])
def test_chi_effective_prior_is_even_and_non_negative(q):
    chi_eff = np.linspace(0.0, 1.0, 501)
    pdf = chi_effective_prior_from_isotropic_spins(chi_eff, q)
    np.testing.assert_array_equal(
        pdf, chi_effective_prior_from_isotropic_spins(-chi_eff, q)
    )
    assert np.all(pdf >= 0.0)
    assert np.all(np.isfinite(pdf))


@pytest.mark.parametrize("q", [0.15, 0.4, 0.65, 0.9])
def test_chi_effective_prior_vanishes_outside_the_support(q):
    chi_eff = np.asarray([-4.0, -1.5, -1.0, 1.0, 1.5, 4.0])
    pdf = chi_effective_prior_from_isotropic_spins(chi_eff, q)
    np.testing.assert_array_equal(pdf, np.zeros_like(chi_eff))


@pytest.mark.parametrize("amax", [0.3, 0.7])
@pytest.mark.parametrize("q", [0.3, 0.8])
def test_chi_effective_prior_scales_with_amax(amax, q):
    r"""A maximum spin magnitude only rescales the density:
    :math:`p_{a}(\chi) = p_{1}(\chi / a) / a`.
    """
    chi_eff = np.linspace(-amax, amax, 1001)
    scaled = chi_effective_prior_from_isotropic_spins(chi_eff, q, amax=amax)
    reference = chi_effective_prior_from_isotropic_spins(chi_eff / amax, q) / amax

    np.testing.assert_allclose(scaled, reference, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(trapezoid(scaled, chi_eff), 1.0, rtol=1e-4)


def test_chi_effective_prior_squeezes_scalar_input():
    value = chi_effective_prior_from_isotropic_spins(0.1, 0.5)
    assert np.ndim(value) == 0
    assert value > 0.0


@pytest.mark.xfail(
    reason="at q = 1 the equal-mass branch evaluates 0 * log(inf) at chi_eff = 0",
    strict=False,
)
def test_chi_effective_prior_at_equal_masses_and_zero_spin():
    assert np.isfinite(chi_effective_prior_from_isotropic_spins(0.0, 1.0))


# ---------------------------------------------------------------------------
# p(chi_eff, chi_p | q)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.4, 0.8])
def test_joint_spin_prior_is_non_negative(q):
    chi_eff = np.linspace(-0.99, 0.99, 61)
    chi_p = np.linspace(1e-3, 0.99, 61)
    grid_eff, grid_p = np.meshgrid(chi_eff, chi_p, indexing="ij")
    with np.errstate(all="ignore"):
        pdf = prior_chieff_chip_isotropic(grid_eff, grid_p, q)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf >= 0.0)


@pytest.mark.parametrize("chi_p", [0.0, 1.0, 1.5])
def test_joint_spin_prior_vanishes_outside_the_chi_p_support(chi_p):
    chi_eff = np.linspace(-0.5, 0.5, 11)
    with np.errstate(all="ignore"):
        pdf = prior_chieff_chip_isotropic(chi_eff, np.full_like(chi_eff, chi_p), 0.6)
    np.testing.assert_array_equal(pdf, np.zeros_like(chi_eff))


@pytest.mark.parametrize("q", [0.4, 0.8])
def test_joint_spin_prior_is_normalised(q):
    chi_eff = np.linspace(-0.999, 0.999, 141)
    chi_p = np.linspace(1e-4, 0.999, 141)
    grid_eff, grid_p = np.meshgrid(chi_eff, chi_p, indexing="ij")
    with np.errstate(all="ignore"):
        pdf = prior_chieff_chip_isotropic(grid_eff, grid_p, q)
    integral = trapezoid(trapezoid(pdf, chi_p, axis=1), chi_eff)
    np.testing.assert_allclose(integral, 1.0, rtol=5e-3)


def test_joint_spin_prior_marginalises_to_the_chi_effective_prior():
    r""":math:`\int p(\chi_{eff}, \chi_p \mid q) \, d\chi_p = p(\chi_{eff} \mid q)`."""
    q = 0.6
    chi_eff = np.linspace(-0.95, 0.95, 31)
    chi_p = np.linspace(1e-4, 1.0, 800)
    grid_eff, grid_p = np.meshgrid(chi_eff, chi_p, indexing="ij")
    with np.errstate(all="ignore"):
        pdf = prior_chieff_chip_isotropic(grid_eff, grid_p, q)

    marginal = trapezoid(pdf, chi_p, axis=1)
    reference = chi_effective_prior_from_isotropic_spins(chi_eff, q)
    np.testing.assert_allclose(marginal, reference, atol=2e-3)
