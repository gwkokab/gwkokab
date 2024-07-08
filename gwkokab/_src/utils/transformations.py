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

from jax import numpy as jnp
from jaxtyping import Array, Real


__all__ = [
    "cart_to_polar",
    "cart_to_spherical",
    "chi_costilt_to_chiz",
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
    "Mc_delta_to_m1_m2",
    "Mc_eta_to_m1_m2",
    "polar_to_cart",
    "reduced_mass",
    "spherical_to_cart",
    "symmetric_mass_ratio",
    "symmetric_mass_ratio_to_delta_m",
    "total_mass",
]


def m1_times_m2(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    r"""
    .. math::
        f(m_1, m_2) = m_1m_2
    """
    return jnp.multiply(m1, m2)


def total_mass(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    r"""
    .. math::
        f(m_1, m_2) = m_1 + m_2
    """
    return jnp.add(m1, m2)


def mass_ratio(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    r"""
    .. math::
        q(m_1, m_2) = \frac{m_2}{m_1}
    """
    return jnp.divide(m2, m1)


def chirp_mass(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    r"""
    .. math::
        M_c(m_1, m_2) = \frac{(m_1m_2)^{3/5}}{(m_1 + m_2)^{1/5}}
    """
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return jnp.multiply(jnp.power(m1m2, 0.6), jnp.power(M, -0.2))


def symmetric_mass_ratio(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    r"""
    .. math::
        \eta(m_1, m_2) = \frac{m_1m_2}{(m_1 + m_2)^2}
    """
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return jnp.multiply(m1m2, jnp.power(M, -2.0))


def reduced_mass(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    r"""
    .. math::
        M_r(m_1, m_2) = \frac{m_1m_2}{(m_1 + m_2)}
    """
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return jnp.divide(m1m2, M)


def delta_m(*, m1: Array | Real, m2: Array | Real) -> Array | Real:
    r"""
    .. math::
        \delta_m(m_1, m_2) = \frac{m_1 - m_2}{m_1 + m_2}
    """
    diff = jnp.subtract(m1, m2)
    M = total_mass(m1=m1, m2=m2)
    return jnp.divide(diff, M)


def delta_m_to_symmetric_mass_ratio(*, delta_m: Array | Real) -> Array | Real:
    r"""
    .. math::
        \eta(\delta_m) = \frac{1 - \delta_m^2}{4}
    """
    delta_m_sq = jnp.square(delta_m)  # delta_m^2
    eta = jnp.multiply(0.25, jnp.subtract(1, delta_m_sq))  # (1 - delta_m^2) / 4
    return eta


def symmetric_mass_ratio_to_delta_m(*, eta: Array | Real) -> Array | Real:
    r"""
    .. math::
        \delta_m(\eta) = \sqrt{1 - 4\eta}
    """
    eta_4 = jnp.multiply(eta, 4)  #  eta*4
    delta_m = jnp.sqrt(jnp.subtract(1, eta_4))  # sqrt(1 - 4 * eta)
    return delta_m


def m1_m2_ordering(*, m1: Array | Real, m2: Array | Real) -> tuple[Array, Array]:
    r"""
    .. math::
        f(m_1, m_2) = \begin{cases}
            (m_1, m_2) & \text{if } m_1 \geq m_2 \\
            (m_2, m_1) & \text{otherwise}
        \end{cases}
    """
    i_sorted = m1 >= m2
    m1_sorted = jnp.where(i_sorted, m1, m2)
    m2_sorted = jnp.where(i_sorted, m2, m1)
    return m1_sorted, m2_sorted


def m_det_z_to_m_source(*, m_det: Array | Real, z: Array | Real) -> Array | Real:
    r"""
    .. math::
        m_{\text{source}}(m_{\text{det}}, z) = \frac{m_{\text{det}}}{1 + z}
    """
    return jnp.divide(
        m_det,
        jnp.add(1.0, z),  # 1 + z
    )


def m_source_z_to_m_det(*, m_source: Array | Real, z: Array | Real) -> Array | Real:
    r"""
    .. math::
        m_{\text{det}}(m_{\text{source}}, z) = m_{\text{source}}(1 + z)
    """
    return jnp.multiply(
        m_source,
        jnp.add(1.0, z),  # 1 + z
    )


def m1_q_to_m2(*, m1: Array | Real, q: Array | Real) -> Array | Real:
    r"""
    .. math::
        m_2(m_1, q) = m_1q
    """
    return jnp.multiply(m1, q)  # m2 = m1 * q


def m2_q_to_m1(*, m2: Array | Real, q: Array | Real) -> Array | Real:
    r"""
    .. math::
        m_1(m_2, q) = \frac{m_2}{q}
    """
    return jnp.divide(m2, q)  # m1 = m2 / q


def M_q_to_m1_m2(*, M: Array | Real, q: Array | Real) -> tuple[Array, Array]:
    m1 = jnp.divide(M, jnp.add(1, q))  # M/(1 + q)
    m2 = m1_q_to_m2(m1=m1, q=q)
    return m1, m2


def chi_costilt_to_chiz(*, chi: Array | Real, costilt: Array | Real) -> Array | Real:
    return jnp.multiply(chi, costilt)


def m1_m2_chi1z_chi2z_to_chiminus(
    *,
    m1: Array | Real,
    m2: Array | Real,
    chi1z: Array | Real,
    chi2z: Array | Real,
) -> Array | Real:
    m1_chi1z = jnp.multiply(m1, chi1z)
    m2_chi2z = jnp.multiply(m2, chi2z)
    M = total_mass(m1=m1, m2=m2)
    diff = jnp.subtract(m1_chi1z, m2_chi2z)
    return jnp.divide(diff, M)


def m1_m2_chi1z_chi2z_to_chieff(
    *,
    m1: Array | Real,
    m2: Array | Real,
    chi1z: Array | Real,
    chi2z: Array | Real,
) -> Array | Real:
    m1_chi1z = jnp.multiply(m1, chi1z)
    m2_chi2z = jnp.multiply(m2, chi2z)
    M = total_mass(m1=m1, m2=m2)
    m_dot_chi = jnp.add(m1_chi1z, m2_chi2z)
    return jnp.divide(m_dot_chi, M)


def m1_m2_chi1_chi2_costilt1_costilt2_to_chieff(
    *,
    m1: Array | Real,
    m2: Array | Real,
    chi1: Array | Real,
    chi2: Array | Real,
    costilt1: Array | Real,
    costilt2: Array | Real,
) -> Array | Real:
    chi1z = chi_costilt_to_chiz(chi=chi1, costilt=costilt1)
    chi2z = chi_costilt_to_chiz(chi=chi2, costilt=costilt2)
    return m1_m2_chi1z_chi2z_to_chieff(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


def m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus(
    *,
    m1: Array | Real,
    m2: Array | Real,
    chi1: Array | Real,
    chi2: Array | Real,
    costilt1: Array | Real,
    costilt2: Array | Real,
) -> Array | Real:
    chi1z = chi_costilt_to_chiz(chi=chi1, costilt=costilt1)
    chi2z = chi_costilt_to_chiz(chi=chi2, costilt=costilt2)
    return m1_m2_chi1z_chi2z_to_chiminus(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


def m1_m2_chieff_chiminus_to_chi1z_chi2z(
    *,
    m1: Array | Real,
    m2: Array | Real,
    chieff: Array | Real,
    chiminus: Array | Real,
) -> tuple[Array, Array]:
    half_M = jnp.multiply(0.5, total_mass(m1=m1, m2=m2))  # M/2
    chi1z = jnp.divide(
        jnp.multiply(half_M, jnp.add(chieff, chiminus)), m1
    )  # chi1z = M/2 * (chieff + chiminus) / m1
    chi2z = jnp.divide(
        jnp.multiply(half_M, jnp.subtract(chieff, chiminus)), m2
    )  # chi2z = M/2 * (chieff - chiminus) / m2
    return chi1z, chi2z


def Mc_delta_chieff_chiminus_to_chi1z_chi2z(
    *,
    Mc: Array | Real,
    delta: Array | Real,
    chieff: Array | Real,
    chiminus: Array | Real,
) -> tuple[Array, Array]:
    m1, m2 = Mc_delta_to_m1_m2(Mc=Mc, delta=delta)
    return m1_m2_chieff_chiminus_to_chi1z_chi2z(
        m1=m1, m2=m2, chieff=chieff, chiminus=chiminus
    )


def Mc_eta_to_m1_m2(*, Mc: Array | Real, eta: Array | Real) -> tuple[Array, Array]:
    delta_sq = jnp.subtract(1, jnp.multiply(4.0, eta))  # 1 - 4 * eta
    delta_sq = jnp.maximum(delta_sq, 0.0)  # to avoid negative values
    delta = jnp.sqrt(delta_sq)  # sqrt(1 - 4 * eta)
    half_Mc = jnp.multiply(0.5, Mc)  # Mc/2
    eta_pow_neg_point_six = jnp.power(eta, -0.6)  # eta^-0.6
    half_Mc_times_eta_pow_neg_point_six = jnp.multiply(
        half_Mc, eta_pow_neg_point_six
    )  # Mc/2 * eta^-0.6
    m2 = jnp.multiply(
        half_Mc_times_eta_pow_neg_point_six, jnp.subtract(1.0, delta)
    )  # m2 = Mc/2 * eta^-0.6 * (1 - delta)
    m1 = jnp.multiply(
        half_Mc_times_eta_pow_neg_point_six, jnp.add(1.0, delta)
    )  # m1 = Mc/2 * eta^-0.6 * (1 + delta)
    return m1, m2


def m1_m2_to_Mc_eta(*, m1: Array | Real, m2: Array | Real) -> tuple[Array, Array]:
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    eta = jnp.multiply(m1m2, jnp.reciprocal(M * M))  # eta = m1 * m2 / M^2
    Mc = jnp.multiply(jnp.power(eta, 0.6), M)  # Mc = M * eta^0.6
    return Mc, eta


def Mc_delta_to_m1_m2(*, Mc: Array | Real, delta: Array | Real) -> tuple[Array, Array]:
    eta = delta_m_to_symmetric_mass_ratio(delta_m=delta)
    return Mc_eta_to_m1_m2(Mc=Mc, eta=eta)


def polar_to_cart(*, r: Array | Real, theta: Array | Real) -> tuple[Array, Array]:
    x = jnp.multiply(r, jnp.cos(theta))
    y = jnp.multiply(r, jnp.sin(theta))
    return x, y


def cart_to_polar(*, x: Array | Real, y: Array | Real) -> tuple[Array, Array]:
    r = jnp.sqrt(jnp.add(jnp.square(x), jnp.square(y)))
    theta = jnp.arctan2(y, x)
    return r, theta


def spherical_to_cart(
    *, r: Array | Real, theta: Array | Real, phi: Array | Real
) -> tuple[Array, Array, Array]:
    x = jnp.multiply(
        jnp.multiply(r, jnp.sin(theta)), jnp.cos(phi)
    )  # x = r * sin(theta) * cos(phi)
    y = jnp.multiply(
        jnp.multiply(r, jnp.sin(theta)), jnp.sin(phi)
    )  # y = r * sin(theta) * sin(phi)
    z = jnp.multiply(r, jnp.cos(theta))  # z = r * cos(theta)
    return x, y, z


def cart_to_spherical(
    *, x: Array | Real, y: Array | Real, z: Array | Real
) -> tuple[Array, Array, Array]:
    r = jnp.sqrt(
        jnp.add(jnp.add(jnp.square(x), jnp.square(y)), jnp.square(z))
    )  # r = sqrt(x^2 + y^2 + z^2)
    theta = jnp.arccos(jnp.divide(z, r))  # theta = arccos(z / r)
    phi = jnp.arctan2(y, x)  # phi = arctan(y / x)
    return r, theta, phi
