#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pytest
from jax import numpy as jnp

from gwkokab.utils.transformations import (
    cart_to_polar,
    cart_to_spherical,
    chi_costilt_to_chiz,
    chirp_mass,
    delta_m,
    delta_m_to_symmetric_mass_ratio,
    m1_m2_chi1_chi2_costilt1_costilt2_to_chieff,
    m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus,
    m1_m2_chi1z_chi2z_to_chieff,
    m1_m2_chi1z_chi2z_to_chiminus,
    m1_m2_chieff_chiminus_to_chi1z_chi2z,
    m1_m2_ordering,
    m1_m2_to_Mc_eta,
    m1_q_to_m2,
    m1_times_m2,
    m2_q_to_m1,
    m_det_z_to_m_source,
    M_q_to_m1_m2,
    m_source_z_to_m_det,
    mass_ratio,
    Mc_delta_chieff_chiminus_to_chi1z_chi2z,
    Mc_delta_to_m1_m2,
    Mc_eta_to_m1_m2,
    polar_to_cart,
    reduced_mass,
    spherical_to_cart,
    symmetric_mass_ratio,
    symmetric_mass_ratio_to_delta_m,
    total_mass,
)


_primary_masses = np.unique(np.random.uniform(0.5, 200, 13))
_secondary_masses = np.array([np.random.uniform(0.5, m) for m in _primary_masses])


_m1m2 = [(m1, m2) for m1, m2 in zip(_primary_masses, _secondary_masses)]


_redshifts = np.random.uniform(0, 1, 3)

_rho = np.random.uniform(0, 10, 3)
_thetas = np.random.uniform(0, 2 * np.pi, 3)
_phis = np.random.uniform(0, np.pi, 3)

_x = np.random.uniform(-10, 10, 7)
_y = np.random.uniform(-10, 10, 7)
_z = np.random.uniform(-10, 10, 7)

_chi = np.random.uniform(-1, 1, 3)
_chi1 = np.random.uniform(-1, 1, 3)
_chi2 = np.random.uniform(-1, 1, 3)

_costilt = np.random.uniform(-1, 1, 3)
_costilt1 = np.random.uniform(-1, 1, 3)
_costilt2 = np.random.uniform(-1, 1, 3)


@pytest.mark.parametrize("m1, m2", _m1m2)
def test_different_mass_representations(m1, m2):
    m1m2 = m1 * m2
    assert jnp.allclose(m1m2, m1_times_m2(m1=m1, m2=m2))
    M = m1 + m2
    assert jnp.allclose(M, total_mass(m1=m1, m2=m2))
    q = m2 / m1
    assert jnp.allclose(q, mass_ratio(m1=m1, m2=m2))
    assert jnp.allclose(m1, m2_q_to_m1(m2=m2, q=q))
    assert jnp.allclose(m2, m1_q_to_m2(m1=m1, q=q))
    Mc = m1m2**0.6 * M**-0.2
    assert jnp.allclose(Mc, chirp_mass(m1=m1, m2=m2))
    eta = m1m2 * M**-2.0
    assert jnp.allclose(eta, symmetric_mass_ratio(m1=m1, m2=m2))
    Mr = m1m2 / M
    assert jnp.allclose(Mr, reduced_mass(m1=m1, m2=m2))
    delta = (m1 - m2) / M
    assert jnp.allclose(delta, delta_m(m1=m1, m2=m2))
    eta_ = (1 - jnp.square(delta)) / 4
    assert jnp.allclose(eta_, delta_m_to_symmetric_mass_ratio(delta_m=delta))
    delta_ = jnp.sqrt(1 - 4 * eta)
    assert jnp.allclose(delta_, symmetric_mass_ratio_to_delta_m(eta=eta))
    m1_, m2_ = M_q_to_m1_m2(M=M, q=q)
    assert jnp.allclose(m1, m1_)
    assert jnp.allclose(m2, m2_)
    Mc_, eta_ = m1_m2_to_Mc_eta(m1=m1, m2=m2)
    assert jnp.allclose(Mc, Mc_)
    assert jnp.allclose(eta, eta_)
    m1_, m2_ = Mc_eta_to_m1_m2(Mc=Mc, eta=eta)
    assert jnp.allclose(m1, m1_)
    assert jnp.allclose(m2, m2_)
    m1_, m2_ = Mc_delta_to_m1_m2(Mc=Mc, delta=delta)
    assert jnp.allclose(m1, m1_)
    assert jnp.allclose(m2, m2_)


def test_m1_m2_ordering():
    m1_, m2_ = m1_m2_ordering(m1=_primary_masses, m2=_secondary_masses)
    i_sorted = _primary_masses >= _secondary_masses
    m1_sorted = jnp.where(i_sorted, _primary_masses, _secondary_masses)
    m2_sorted = jnp.where(i_sorted, _secondary_masses, _primary_masses)
    assert jnp.allclose(m1_sorted, m1_)
    assert jnp.allclose(m2_sorted, m2_)


@pytest.mark.parametrize("m", _primary_masses)
@pytest.mark.parametrize("z", _redshifts)
def test_mass_and_reshift(m, z):
    m_source = m / (1.0 + z)
    assert jnp.allclose(m_source, m_det_z_to_m_source(m_det=m, z=z))
    m_det = m * (1.0 + z)
    assert jnp.allclose(m_det, m_source_z_to_m_det(m_source=m, z=z))


@pytest.mark.parametrize("chi", _chi)
@pytest.mark.parametrize("costilt", _costilt)
def test_chi_costilt_to_chiz(chi, costilt):
    chi_z = chi * costilt
    assert jnp.allclose(chi_z, chi_costilt_to_chiz(chi=chi, costilt=costilt))


@pytest.mark.parametrize("m1, m2", _m1m2)
@pytest.mark.parametrize("chi1", _chi1)
@pytest.mark.parametrize("chi2", _chi2)
def test_m1_m2_chi1z_chi2z(m1, m2, chi1, chi2):
    chieff_ = m1_m2_chi1z_chi2z_to_chieff(m1=m1, m2=m2, chi1z=chi1, chi2z=chi2)
    chieff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    assert jnp.allclose(chieff, chieff_)
    chiminus_ = m1_m2_chi1z_chi2z_to_chiminus(m1=m1, m2=m2, chi1z=chi1, chi2z=chi2)
    chiminus = (m1 * chi1 - m2 * chi2) / (m1 + m2)
    assert jnp.allclose(chiminus, chiminus_)


@pytest.mark.parametrize("m1, m2", _m1m2)
@pytest.mark.parametrize("chieff", _chi)
@pytest.mark.parametrize("chiminus", _chi)
def test_m1_m2_chieff_chiminus(m1, m2, chieff, chiminus):
    chi1z_, chi2z_ = m1_m2_chieff_chiminus_to_chi1z_chi2z(
        m1=m1, m2=m2, chieff=chieff, chiminus=chiminus
    )
    chi1z = (m1 + m2) * (chieff + chiminus) / (2 * m1)
    chi2z = (m1 + m2) * (chieff - chiminus) / (2 * m2)
    assert jnp.allclose(chi1z, chi1z_)
    assert jnp.allclose(chi2z, chi2z_)
    m1m2 = m1 * m2
    M = m1 + m2
    Mc = m1m2**0.6 * M**-0.2
    delta = (m1 - m2) / M
    chi1z_, chi2z_ = Mc_delta_chieff_chiminus_to_chi1z_chi2z(
        Mc=Mc, delta=delta, chieff=chieff, chiminus=chiminus
    )
    assert jnp.allclose(chi1z, chi1z_)
    assert jnp.allclose(chi2z, chi2z_)


@pytest.mark.parametrize("m1, m2", _m1m2)
@pytest.mark.parametrize("chi1", _chi1)
@pytest.mark.parametrize("chi2", _chi2)
@pytest.mark.parametrize("costilt1", _costilt1)
@pytest.mark.parametrize("costilt2", _costilt2)
def test_m1_m2_chi1_chi2_costilt1_costilt2(m1, m2, chi1, chi2, costilt1, costilt2):
    chieff_ = m1_m2_chi1_chi2_costilt1_costilt2_to_chieff(
        m1=m1, m2=m2, chi1=chi1, chi2=chi2, costilt1=costilt1, costilt2=costilt2
    )
    chieff = (m1 * chi1 * costilt1 + m2 * chi2 * costilt2) / (m1 + m2)
    assert jnp.allclose(chieff, chieff_)
    chiminus_ = m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus(
        m1=m1, m2=m2, chi1=chi1, chi2=chi2, costilt1=costilt1, costilt2=costilt2
    )
    chiminus = (m1 * chi1 * costilt1 - m2 * chi2 * costilt2) / (m1 + m2)
    assert jnp.allclose(chiminus, chiminus_)


@pytest.mark.parametrize("x", _x)
@pytest.mark.parametrize("y", _y)
def test_cart_to_polar(x, y):
    r, theta = cart_to_polar(x=x, y=y)
    r_ = jnp.sqrt(jnp.square(x) + jnp.square(y))
    theta_ = jnp.arctan2(y, x)
    assert jnp.allclose(r, r_)
    assert jnp.allclose(theta, theta_)


@pytest.mark.parametrize("r", _rho)
@pytest.mark.parametrize("theta", _thetas)
def test_polar_to_cart(r, theta):
    x, y = polar_to_cart(r=r, theta=theta)
    x_ = r * jnp.cos(theta)
    y_ = r * jnp.sin(theta)
    assert jnp.allclose(x, x_)
    assert jnp.allclose(y, y_)


@pytest.mark.parametrize("r", _rho)
@pytest.mark.parametrize("theta", _thetas)
@pytest.mark.parametrize("phi", _phis)
def test_spherical_to_cart(r, theta, phi):
    x, y, z = spherical_to_cart(r=r, theta=theta, phi=phi)
    x_ = r * jnp.sin(theta) * jnp.cos(phi)
    y_ = r * jnp.sin(theta) * jnp.sin(phi)
    z_ = r * jnp.cos(theta)
    assert jnp.allclose(x, x_)
    assert jnp.allclose(y, y_)
    assert jnp.allclose(z, z_)


@pytest.mark.parametrize("x", _x)
@pytest.mark.parametrize("y", _y)
@pytest.mark.parametrize("z", _z)
def test_cart_to_spherical(x, y, z):
    r, theta, phi = cart_to_spherical(x=x, y=y, z=z)
    r_ = jnp.sqrt(jnp.square(x) + jnp.square(y) + jnp.square(z))
    theta_ = jnp.arccos(z / r_)
    phi_ = jnp.arctan2(y, x)
    assert jnp.allclose(r, r_)
    assert jnp.allclose(theta, theta_)
    assert jnp.allclose(phi, phi_)
