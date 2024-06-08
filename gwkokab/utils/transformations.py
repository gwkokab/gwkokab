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

from __future__ import annotations

from numbers import Real

import jax
from jax import lax, numpy as jnp
from jaxtyping import Array


__all__ = [
    "cart_to_polar",
    "cart_to_spherical",
    "chi_cos_tilt_to_chiz",
    "chirp_mass",
    "delta_m",
    "delta_m_to_symmetric_mass_ratio",
    "m1_m2_chi1_chi2_costilt1_costilt2_to_chieff",
    "m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus",
    "m1_m2_chi1z_chi2z_to_chieff",
    "m1_m2_chi1z_chi2z_to_chiminus",
    "m1_m2_chieff_chiminus_to_chi1z_chi2z",
    "m1_m2_ordering",
    "m1_m2_to_Mc_eta",
    "m1_q_to_m2",
    "m1_times_m2",
    "m2_q_to_m1",
    "m_det_z_to_m_source",
    "M_q_to_m1_m2",
    "m_source_z_to_m_det",
    "mass_ratio",
    "Mc_delta_chieff_chiminus_to_chi1z_chi2z",
    "MC_delta_to_m1_m2",
    "Mc_eta_to_m1_m2",
    "polar_to_cart",
    "reduced_mass",
    "spherical_to_cart",
    "symmetric_mass_ratio",
    "symmetric_mass_ratio_to_delta_m",
    "total_mass",
]


def _m1_times_m2(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    return lax.mul(m1, m2)


def _total_mass(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    return lax.add(m1, m2)


def _mass_ratio(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    return lax.div(m2, m1)


def _chirp_mass(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    M = _total_mass(m1, m2)
    m1m2 = _m1_times_m2(m1, m2)
    return lax.mul(lax.pow(m1m2, 0.6), lax.pow(M, -0.2))


def _symmetric_mass_ratio(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    M = _total_mass(m1, m2)
    m1m2 = _m1_times_m2(m1, m2)
    return lax.mul(m1m2, lax.pow(M, -2))


def _reduced_mass(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    M = _total_mass(m1, m2)
    m1m2 = _m1_times_m2(m1, m2)
    return lax.div(m1m2, M)


def _delta_m(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    diff = lax.sub(m1, m2)
    M = _total_mass(m1, m2)
    return lax.div(diff, M)


def _delta_m_to_symmetric_mass_ratio(*, delta_m: Array | Real) -> Array | Real:
    delta_m_sq = lax.square(delta_m)  # delta_m^2
    eta = lax.mul(0.25, lax.sub(1, delta_m_sq))  # (1 - delta_m^2) / 4
    return eta


def _symmetric_mass_ratio_to_delta_m(*, eta: Array | Real) -> Array | Real:
    eta_4 = lax.mul(eta, 4)  #  eta*4
    delta_m = lax.sqrt(lax.sub(1, eta_4))  # sqrt(1 - 4 * eta)
    return delta_m


def _m1_m2_ordering(*, m1: Array | Real, m2: Array | Real) -> tuple[Array, Array]:
    i_sorted = m1 >= m2
    m1_sorted = jnp.where(i_sorted, m1, m2)
    m2_sorted = jnp.where(i_sorted, m2, m1)
    return m1_sorted, m2_sorted


def _m_det_z_to_m_source(*, m_det: Array | Real, z: Array | Real) -> Array | Real:
    return lax.div(
        m_det,
        lax.add(1.0, z),  # 1 + z
    )


def _m_source_z_to_m_det(*, m_source: Array | Real, z: Array | Real) -> Array | Real:
    return lax.mul(
        m_source,
        lax.add(1.0, z),  # 1 + z
    )


def _m1_q_to_m2(*, m1: Array | Real, q: Array | Real) -> Array | Real:
    return lax.mul(m1, q)  # m2 = m1 * q


def _m2_q_to_m1(*, m2: Array | Real, q: Array | Real) -> Array | Real:
    return lax.div(m2, q)  # m1 = m2 / q


def _M_q_to_m1_m2(*, M: Array | Real, q: Array | Real) -> tuple[Array, Array]:
    m1 = lax.div(
        M,
        lax.add(1, q),  # 1 + q
    )
    m2 = lax.mul(m1, q)  # m2 = m1 * q
    return m1, m2


def _chi_cos_tilt_to_chiz(*, chi: Array | Real, cos_tilt: Array | Real) -> Array | Real:
    return lax.mul(chi, cos_tilt)


def _m1_m2_chi1z_chi2z_to_chiminus(
    *, m1: Array | Real, m2: Array | Real, chi1z: Array | Real, chi2z: Array | Real
) -> Array | Real:
    m1_chi1z = lax.mul(m1, chi1z)
    m2_chi2z = lax.mul(m2, chi2z)
    M = _total_mass(m1, m2)
    diff = lax.sub(m1_chi1z, m2_chi2z)
    return lax.div(diff, M)


def _m1_m2_chi1z_chi2z_to_chieff(
    *, m1: Array | Real, m2: Array | Real, chi1z: Array | Real, chi2z: Array | Real
) -> Array | Real:
    m1_chi1z = lax.mul(m1, chi1z)
    m2_chi2z = lax.mul(m2, chi2z)
    M = _total_mass(m1, m2)
    m_dot_chi = lax.add(m1_chi1z, m2_chi2z)
    return lax.div(m_dot_chi, M)


def _m1_m2_chi1_chi2_costilt1_costilt2_to_chieff(
    *,
    m1: Array | Real,
    m2: Array | Real,
    chi1: Array | Real,
    chi2: Array | Real,
    costilt1: Array | Real,
    costilt2: Array | Real,
) -> Array | Real:
    chi1z = _chi_cos_tilt_to_chiz(chi=chi1, cos_tilt=costilt1)
    chi2z = _chi_cos_tilt_to_chiz(chi=chi2, cos_tilt=costilt2)
    return _m1_m2_chi1z_chi2z_to_chieff(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


def _m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus(
    *,
    m1: Array | Real,
    m2: Array | Real,
    chi1: Array | Real,
    chi2: Array | Real,
    costilt1: Array | Real,
    costilt2: Array | Real,
) -> Array | Real:
    chi1z = _chi_cos_tilt_to_chiz(chi=chi1, cos_tilt=costilt1)
    chi2z = _chi_cos_tilt_to_chiz(chi=chi2, cos_tilt=costilt2)
    return _m1_m2_chi1z_chi2z_to_chiminus(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


def _m1_m2_chieff_chiminus_to_chi1z_chi2z(
    *, m1: Array | Real, m2: Array | Real, chieff: Array | Real, chiminus: Array | Real
) -> tuple[Array, Array]:
    half_M = lax.mul(0.5, _total_mass(m1, m2))  # M/2
    chi1z = lax.div(lax.mul(half_M, lax.add(chieff, chiminus)), m1)  # chi1z = M/2 * (chieff + chiminus) / m1
    chi2z = lax.div(lax.mul(half_M, lax.sub(chieff, chiminus)), m2)  # chi2z = M/2 * (chieff - chiminus) / m2
    return chi1z, chi2z


def _Mc_delta_chieff_chiminus_to_chi1z_chi2z(
    *, Mc: Array | Real, delta: Array | Real, chieff: Array | Real, chiminus: Array | Real
) -> tuple[Array, Array]:
    m1, m2 = _M_q_to_m1_m2(Mc, delta)
    return _m1_m2_chieff_chiminus_to_chi1z_chi2z(m1=m1, m2=m2, chieff=chieff, chiminus=chiminus)


def _Mc_eta_to_m1_m2(*, Mc: Array | Real, eta: Array | Real) -> tuple[Array, Array]:
    delta_sq = lax.sub(1, lax.mul(4, eta))  # 1 - 4 * eta
    delta_sq = lax.max(delta_sq, 0)  # to avoid negative values
    delta = lax.sqrt(delta_sq)  # sqrt(1 - 4 * eta)
    half_Mc = lax.mul(0.5, Mc)  # Mc/2
    eta_pow_point_six = lax.pow(eta, -0.6)  # eta^-0.6
    m1 = lax.mul(half_Mc, eta_pow_point_six)  # Mc/2 * eta^-0.6
    m2 = lax.mul(m1, lax.sub(1, delta))  # m2 = Mc/2 * eta^-0.6 * (1 - delta)
    m1 = lax.mul(m1, lax.add(1, delta))  # m1 = Mc/2 * eta^-0.6 * (1 + delta)
    return m1, m2


def _m1_m2_to_Mc_eta(*, m1: Array | Real, m2: Array | Real) -> tuple[Array, Array]:
    M = _total_mass(m1, m2)
    eta = m1 * m2 * lax.reciprocal(M * M)  # eta = m1 * m2 / M^2
    Mc = lax.mul(lax.pow(eta, 0.6), M)  # Mc = M * eta^0.6
    return Mc, eta


def _MC_delta_to_m1_m2(*, Mc: Array | Real, delta: Array | Real) -> tuple[Array, Array]:
    eta = _symmetric_mass_ratio_to_delta_m(0.25 * delta)  # eta = sqrt(1 - 4 * delta) / 4
    return _Mc_eta_to_m1_m2(Mc=Mc, eta=eta)


def _polar_to_cart(*, r: Array | Real, theta: Array | Real) -> tuple[Array, Array]:
    x = r * lax.cos(theta)
    y = r * lax.sin(theta)
    return x, y


def _cart_to_polar(*, x: Array | Real, y: Array | Real) -> tuple[Array, Array]:
    r = lax.sqrt(lax.square(x) + lax.square(y))
    theta = jnp.arctan2(y, x)
    return r, theta


def _spherical_to_cart(*, r: Array | Real, theta: Array | Real, phi: Array | Real) -> tuple[Array, Array, Array]:
    x = r * lax.sin(theta) * lax.cos(phi)
    y = r * lax.sin(theta) * lax.sin(phi)
    z = r * lax.cos(theta)
    return x, y, z


def _cart_to_spherical(*, x: Array | Real, y: Array | Real, z: Array | Real) -> tuple[Array, Array, Array]:
    r = lax.sqrt(lax.square(x) + lax.square(y) + lax.square(z))
    theta = jnp.arccos(lax.div(z, r))
    phi = jnp.arctan2(y, x)
    return r, theta, phi


cart_to_polar = jax.jit(_cart_to_polar, inline=True)
cart_to_spherical = jax.jit(_cart_to_spherical, inline=True)
chi_cos_tilt_to_chiz = jax.jit(_chi_cos_tilt_to_chiz, inline=True)
chirp_mass = jax.jit(_chirp_mass, inline=True)
delta_m = jax.jit(_delta_m, inline=True)
delta_m_to_symmetric_mass_ratio = jax.jit(_delta_m_to_symmetric_mass_ratio, inline=True)
m1_m2_chi1_chi2_costilt1_costilt2_to_chieff = jax.jit(_m1_m2_chi1_chi2_costilt1_costilt2_to_chieff, inline=True)
m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus = jax.jit(_m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus, inline=True)
m1_m2_chi1z_chi2z_to_chieff = jax.jit(_m1_m2_chi1z_chi2z_to_chieff, inline=True)
m1_m2_chi1z_chi2z_to_chiminus = jax.jit(_m1_m2_chi1z_chi2z_to_chiminus, inline=True)
m1_m2_chieff_chiminus_to_chi1z_chi2z = jax.jit(_m1_m2_chieff_chiminus_to_chi1z_chi2z, inline=True)
m1_m2_ordering = jax.jit(_m1_m2_ordering, inline=True)
m1_m2_to_Mc_eta = jax.jit(_m1_m2_to_Mc_eta, inline=True)
m1_q_to_m2 = jax.jit(_m1_q_to_m2, inline=True)
m1_times_m2 = jax.jit(_m1_times_m2, inline=True)
m2_q_to_m1 = jax.jit(_m2_q_to_m1, inline=True)
m_det_z_to_m_source = jax.jit(_m_det_z_to_m_source, inline=True)
M_q_to_m1_m2 = jax.jit(_M_q_to_m1_m2, inline=True)
m_source_z_to_m_det = jax.jit(_m_source_z_to_m_det, inline=True)
mass_ratio = jax.jit(_mass_ratio, inline=True)
Mc_delta_chieff_chiminus_to_chi1z_chi2z = jax.jit(_Mc_delta_chieff_chiminus_to_chi1z_chi2z, inline=True)
MC_delta_to_m1_m2 = jax.jit(_MC_delta_to_m1_m2, inline=True)
Mc_eta_to_m1_m2 = jax.jit(_Mc_eta_to_m1_m2, inline=True)
polar_to_cart = jax.jit(_polar_to_cart, inline=True)
reduced_mass = jax.jit(_reduced_mass, inline=True)
spherical_to_cart = jax.jit(_spherical_to_cart, inline=True)
symmetric_mass_ratio = jax.jit(_symmetric_mass_ratio, inline=True)
symmetric_mass_ratio_to_delta_m = jax.jit(_symmetric_mass_ratio_to_delta_m, inline=True)
total_mass = jax.jit(_total_mass, inline=True)
