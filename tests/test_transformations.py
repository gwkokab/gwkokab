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
    chirp_mass,
    delta_m,
    delta_m_to_symmetric_mass_ratio,
    m1_q_to_m2,
    m1_times_m2,
    m2_q_to_m1,
    m_det_z_to_m_source,
    m_source_z_to_m_det,
    mass_ratio,
    polar_to_cart,
    reduced_mass,
    spherical_to_cart,
    symmetric_mass_ratio,
    symmetric_mass_ratio_to_delta_m,
    total_mass,
)


_primary_masses = np.unique(np.random.uniform(0.5, 200, 7))
_secondary_masses = np.array([np.random.uniform(0.5, m) for m in _primary_masses])
_redshifts = np.random.uniform(0, 1, 5)
_rho = np.random.uniform(0, 10, 5)
_thetas = np.random.uniform(0, 2 * np.pi, 7)
_phis = np.random.uniform(0, np.pi, 7)
_x = np.random.uniform(-10, 10, 11)
_y = np.random.uniform(-10, 10, 11)
_z = np.random.uniform(-10, 10, 11)


@pytest.mark.parametrize("m1", _primary_masses)
@pytest.mark.parametrize("m2", _secondary_masses)
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
    delta_to_eta = (1 - delta**2) / 4
    assert jnp.allclose(delta_to_eta, delta_m_to_symmetric_mass_ratio(delta_m=delta))
    eta_to_delta = jnp.sqrt(1 - 4 * eta)
    assert jnp.allclose(eta_to_delta, symmetric_mass_ratio_to_delta_m(eta=eta))


@pytest.mark.parametrize("m", _primary_masses + _secondary_masses)
@pytest.mark.parametrize("z", _redshifts)
def test_mass_and_reshift(m, z):
    m_source = m / (1.0 + z)
    assert jnp.allclose(m_source, m_det_z_to_m_source(m_det=m, z=z))
    m_det = m * (1.0 + z)
    assert jnp.allclose(m_det, m_source_z_to_m_det(m_source=m, z=z))


@pytest.mark.parametrize("x", _x)
@pytest.mark.parametrize("y", _y)
def test_cart_to_polar(x, y):
    r, theta = cart_to_polar(x=x, y=y)
    r_ = jnp.sqrt(x**2 + y**2)
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
    r_ = jnp.sqrt(x**2 + y**2 + z**2)
    theta_ = jnp.arccos(z / r_)
    phi_ = jnp.arctan2(y, x)
    assert jnp.allclose(r, r_)
    assert jnp.allclose(theta, theta_)
    assert jnp.allclose(phi, phi_)
