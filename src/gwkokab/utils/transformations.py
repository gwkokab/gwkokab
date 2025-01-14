# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial
from typing import Optional

import jax
from jax import numpy as jnp
from jaxtyping import ArrayLike


def m1_times_m2(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_1m_2(m_1, m_2) = m_1 m_2
    """
    return m1 * m2


def total_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        M(m_1, m_2) = m_1 + m_2
    """
    return m1 + m2


@partial(jax.jit, inline=True)
def _mass_ratio_from_m1_m2(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    safe_m1 = jnp.where(m1 == 0.0, 1.0, m1)
    return jnp.where(m1 == 0.0, jnp.inf, m2 / safe_m1)


@partial(jax.jit, inline=True)
def _mass_ratio_from_eta(eta: ArrayLike) -> ArrayLike:
    """
    implementation reference: https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.conversion.symmetric_mass_ratio_to_mass_ratio.html
    """
    temp = 1.0 / (2.0 * eta) - 1.0
    return temp - jnp.sqrt(jnp.square(temp) - 1.0)


def mass_ratio(
    *,
    m1: Optional[ArrayLike] = None,
    m2: Optional[ArrayLike] = None,
    eta: Optional[ArrayLike] = None,
) -> ArrayLike:
    r"""
    .. math::
        q(m_1, m_2) = \frac{m_2}{m_1}

    .. math::
        q(\eta) = \frac{1}{2\eta} - 1 - \sqrt{\left(\frac{1}{2\eta} - 1\right)^2 - 1}
    """
    if m1 is not None and m2 is not None and eta is not None:
        raise ValueError("Only one of (m1, m2) or eta should be provided.")
    if eta is not None:
        return _mass_ratio_from_eta(eta=eta)
    return _mass_ratio_from_m1_m2(m1=m1, m2=m2)


@partial(jax.jit, inline=True)
def _chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return jnp.power(m1m2, 0.6) * jnp.power(M, -0.2)


def chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        M_c(m_1, m_2) = \frac{(m_1m_2)^{3/5}}{(m_1 + m_2)^{1/5}}
    """
    return _chirp_mass(m1=m1, m2=m2)


@partial(jax.jit, inline=True)
def _log_chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    M = total_mass(m1=m1, m2=m2)
    return 0.6 * (jnp.log(m1) + jnp.log(m2)) - 0.2 * jnp.log(M)


def log_chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \log(M_c(m_1, m_2)) = 3/5\times (\log(m_1) + \log(m_2)) - \log(m_1 + m_2)/5
    """
    return _log_chirp_mass(m1=m1, m2=m2)


@partial(jax.jit, inline=True)
def _symmetric_mass_ratio_from_m1_m2(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return m1m2 * jnp.power(M, -2.0)


@partial(jax.jit, inline=True)
def _symmetric_mass_ratio_from_q(q: ArrayLike) -> ArrayLike:
    safe_q = jnp.where(q == -1.0, 1.0, q)
    return jnp.where(q == -1.0, jnp.inf, safe_q / jnp.square(1.0 + safe_q))


def symmetric_mass_ratio(
    *, m1: Optional[ArrayLike], m2: Optional[ArrayLike], q: Optional[ArrayLike] = None
) -> ArrayLike:
    r"""
    .. math::
        \eta(m_1, m_2) = \frac{m_1m_2}{(m_1 + m_2)^2}

    .. math::
        \eta(q) = \frac{q}{(1 + q)^2}
    """
    if m1 is not None and m2 is not None and q is not None:
        raise ValueError("Only one of (m1, m2) or q should be provided.")
    if q is not None:
        return _symmetric_mass_ratio_from_q(q=q)
    return _symmetric_mass_ratio_from_m1_m2(m1=m1, m2=m2)


@partial(jax.jit, inline=True)
def _reduced_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return jnp.where(M == 0.0, jnp.inf, m1m2 / M)


def reduced_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        M_r(m_1, m_2) = \frac{m_1m_2}{m_1 + m_2}
    """
    return _reduced_mass(m1=m1, m2=m2)


@partial(jax.jit, inline=True)
def _delta_m(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    diff = m1 - m2
    M = total_mass(m1=m1, m2=m2)
    return jnp.where(M == 0.0, jnp.inf, diff / M)


def delta_m(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \delta_m(m_1, m_2) = \frac{m_1 - m_2}{m_1 + m_2}
    """
    return _delta_m(m1=m1, m2=m2)


@partial(jax.jit, inline=True)
def _delta_m_to_symmetric_mass_ratio(delta_m: ArrayLike) -> ArrayLike:
    δ_m_sq = jnp.square(delta_m)  # delta_m^2
    η = 0.25 * (1.0 - δ_m_sq)  # (1 - delta_m^2) / 4
    return η


def delta_m_to_symmetric_mass_ratio(delta_m: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \eta(\delta_m) = \frac{1 - \delta_m^2}{4}
    """
    return _delta_m_to_symmetric_mass_ratio(delta_m=delta_m)


@partial(jax.jit, inline=True)
def _symmetric_mass_ratio_to_delta_m(eta: ArrayLike) -> ArrayLike:
    η_4 = jnp.multiply(eta, 4)  #  eta*4
    δ_m = jnp.sqrt(jnp.subtract(1.0, η_4))  # sqrt(1 - 4 * eta)
    return δ_m


def symmetric_mass_ratio_to_delta_m(eta: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \delta_m(\eta) = \sqrt{1 - 4\eta}
    """
    return _symmetric_mass_ratio_to_delta_m(eta=eta)


@partial(jax.jit, inline=True)
def _m_det_z_to_m_source(m_det: ArrayLike, z: ArrayLike) -> ArrayLike:
    safe_one_more_z = jnp.where(z == -1.0, 1.0, jnp.add(1.0, z))
    m_source = jnp.where(z == -1.0, jnp.inf, m_det / safe_one_more_z)
    return m_source


def m_det_z_to_m_source(m_det: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_{\text{source}}(m_{\text{det}}, z) = \frac{m_{\text{det}}}{1 + z}
    """
    return _m_det_z_to_m_source(m_det=m_det, z=z)


@partial(jax.jit, inline=True)
def _m_source_z_to_m_det(m_source: ArrayLike, z: ArrayLike) -> ArrayLike:
    return m_source * (1.0 + z)


def m_source_z_to_m_det(m_source: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_{\text{det}}(m_{\text{source}}, z) = m_{\text{source}}(1 + z)
    """
    return _m_source_z_to_m_det(m_source=m_source, z=z)


def m1_q_to_m2(m1: ArrayLike, q: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_2(m_1, q) = m_1q
    """
    return m1 * q


@partial(jax.jit, inline=True)
def _m2_q_to_m1(m2: ArrayLike, q: ArrayLike) -> ArrayLike:
    safe_q = jnp.where(q == 0.0, 1.0, q)
    m1 = jnp.where(q == 0.0, jnp.inf, m2 / safe_q)
    return m1


def m2_q_to_m1(m2: ArrayLike, q: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_1(m_2, q) = \frac{m_2}{q}
    """
    return _m2_q_to_m1(m2=m2, q=q)


def chi_costilt_to_chiz(chi: ArrayLike, costilt: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \chi_z(\chi, \cos(\theta)) = \chi \cos(\theta)
    """
    return chi * costilt


@partial(jax.jit, inline=True)
def _m1_m2_chi1z_chi2z_to_chiminus(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    m1_χ1z = m1 * chi1z
    m2_χ2z = m2 * chi2z
    M = total_mass(m1=m1, m2=m2)
    diff = m1_χ1z - m2_χ2z
    return jnp.where(M == 0.0, jnp.inf, diff / M)


def m1_m2_chi1z_chi2z_to_chiminus(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    r"""
    .. math::
        \chi_{\text{minus}}(m_1, m_2, \chi_{1z}, \chi_{2z}) = \frac{m_1\chi_{1z} - m_2\chi_{2z}}{m_1 + m_2}
    """
    return _m1_m2_chi1z_chi2z_to_chiminus(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


@partial(jax.jit, inline=True)
def _chi_eff_by_m1_m2(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    m1_χ1z = m1 * chi1z
    m2_χ2z = m2 * chi2z
    M = total_mass(m1=m1, m2=m2)
    m_dot_χ = m1_χ1z + m2_χ2z
    return jnp.where(M == 0, jnp.inf, m_dot_χ / M)


@partial(jax.jit, inline=True)
def _chi_eff_by_q(q: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike) -> ArrayLike:
    safe_q = jnp.where(q == -1.0, 1.0, q)
    safe_χ_eff = jnp.where(q == -1.0, 1.0, (chi1z + safe_q * chi2z) / (1 + safe_q))
    return safe_χ_eff


def chi_eff(
    *,
    chi1z: ArrayLike,
    chi2z: ArrayLike,
    m1: Optional[ArrayLike] = None,
    m2: Optional[ArrayLike] = None,
    q: Optional[ArrayLike] = None,
) -> ArrayLike:
    r"""
    .. math::
        \chi_{\text{eff}}(m_1, m_2, \chi_{1z}, \chi_{2z}) = \frac{m_1\chi_{1z} + m_2\chi_{2z}}{m_1 + m_2}

    .. math::
        \chi_{\text{eff}}(q, \chi_{1z}, \chi_{2z}) = \frac{\chi_{1z} + q\chi_{2z}}{1 + q}
    """
    if m1 is not None and m2 is not None and q is not None:
        raise ValueError("Only one of (m1, m2) or q should be provided.")
    if q is not None:
        return _chi_eff_by_q(q=q, chi1z=chi1z, chi2z=chi2z)
    return _chi_eff_by_m1_m2(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


@partial(jax.jit, inline=True)
def _chi_a(chi1z: ArrayLike, chi2z: ArrayLike) -> ArrayLike:
    return (chi1z + chi2z) * 0.5


def chi_a(chi1z: ArrayLike, chi2z: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \chi_a(\chi_{1z}, \chi_{2z}) = \frac{\chi_{1z} + \chi_{2z}}{2}
    """
    return _chi_a(chi1z=chi1z, chi2z=chi2z)


def m1_m2_chi1_chi2_costilt1_costilt2_to_chieff(
    *,
    m1: ArrayLike,
    m2: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    costilt1: ArrayLike,
    costilt2: ArrayLike,
) -> ArrayLike:
    r"""
    .. math::
        \chi_{\text{eff}}(m_1, m_2, \chi_1, \chi_2, \cos(\theta_1), \cos(\theta_2)) =
        \frac{m_1\chi_1\cos(\theta_1) + m_2\chi_2\cos(\theta_2)}{m_1 + m_2}
    """
    χ1z = chi_costilt_to_chiz(chi=chi1, costilt=costilt1)
    χ2z = chi_costilt_to_chiz(chi=chi2, costilt=costilt2)
    return chi_eff(m1=m1, m2=m2, chi1z=χ1z, chi2z=χ2z)


def m1_m2_chi1_chi2_costilt1_costilt2_to_chiminus(
    *,
    m1: ArrayLike,
    m2: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    costilt1: ArrayLike,
    costilt2: ArrayLike,
) -> ArrayLike:
    r"""
    .. math::
        \chi_{\text{minus}}(m_1, m_2, \chi_1, \chi_2, \cos(\theta_1), \cos(\theta_2)) =
        \frac{m_1\chi_1\cos(\theta_1) - m_2\chi_2\cos(\theta_2)}{m_1 + m_2}
    """
    χ1z = chi_costilt_to_chiz(chi=chi1, costilt=costilt1)
    χ2z = chi_costilt_to_chiz(chi=chi2, costilt=costilt2)
    return m1_m2_chi1z_chi2z_to_chiminus(m1=m1, m2=m2, chi1z=χ1z, chi2z=χ2z)


def m1_m2_chieff_chiminus_to_chi1z_chi2z(
    m1: ArrayLike, m2: ArrayLike, chieff: ArrayLike, chiminus: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    r"""
    .. math::
        \begin{align*}
        \chi_{1z}(m_1, m_2, \chi_{\text{eff}}, \chi_{\text{minus}}) &=
        \frac{m_1+m_2}{2m_1} \left( \chi_{\text{eff}} + \chi_{\text{minus}} \right)\\
        \chi_{2z}(m_1, m_2, \chi_{\text{eff}}, \chi_{\text{minus}}) &=
        \frac{m_1+m_2}{2m_2} \left( \chi_{\text{eff}} - \chi_{\text{minus}} \right)
        \end{align*}
    """
    half_M = jnp.multiply(0.5, total_mass(m1=m1, m2=m2))  # M/2
    χ1z = jnp.divide(
        jnp.multiply(half_M, jnp.add(chieff, chiminus)), m1
    )  # chi1z = M/2 * (chieff + chiminus) / m1
    χ2z = jnp.divide(
        jnp.multiply(half_M, jnp.subtract(chieff, chiminus)), m2
    )  # chi2z = M/2 * (chieff - chiminus) / m2
    return χ1z, χ2z


def Mc_eta_to_m1_m2(Mc: ArrayLike, eta: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    r"""
    .. math::
        \begin{align*}
            m_1(M_c, \eta) &= \frac{M_c}{2} \eta^{-0.6} (1 + \sqrt{1 - 4\eta}) \\
            m_2(M_c, \eta) &= \frac{M_c}{2} \eta^{-0.6} (1 - \sqrt{1 - 4\eta})
        \end{align*}
    """
    δ_sq = jnp.subtract(1, jnp.multiply(4.0, eta))  # 1 - 4 * eta
    δ_sq = jnp.maximum(δ_sq, jnp.zeros_like(δ_sq))  # to avoid negative values
    δ = jnp.sqrt(δ_sq)  # sqrt(1 - 4 * eta)
    half_Mc = jnp.multiply(0.5, Mc)  # Mc/2
    η_pow_neg_point_six = jnp.power(eta, -0.6)  # eta^-0.6
    half_Mc_times_η_pow_neg_point_six = jnp.multiply(
        half_Mc, η_pow_neg_point_six
    )  # Mc/2 * eta^-0.6
    m2 = jnp.multiply(
        half_Mc_times_η_pow_neg_point_six, jnp.subtract(1.0, δ)
    )  # m2 = Mc/2 * eta^-0.6 * (1 - delta)
    m1 = jnp.multiply(
        half_Mc_times_η_pow_neg_point_six, jnp.add(1.0, δ)
    )  # m1 = Mc/2 * eta^-0.6 * (1 + delta)
    return m1, m2


def polar_to_cart(r: ArrayLike, theta: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    r"""
    .. math::
        \begin{align*}
            x(r, \theta) &= r \cos(\theta) \\
            y(r, \theta) &= r \sin(\theta)
        \end{align*}
    """
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return x, y


def cart_to_polar(x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    r"""
    .. math::
        \begin{align*}
            r(x, y) &= \sqrt{x^2 + y^2} \\
            \theta(x, y) &= \arctan(y/x)
        \end{align*}
    """
    r = jnp.sqrt(x * x + y * y)
    θ = jnp.arctan2(y, x)
    return r, θ


def spherical_to_cart(
    r: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""
    .. math::
        \begin{align*}
            x(r, \theta, \phi) &= r \sin(\theta) \cos(\phi) \\
            y(r, \theta, \phi) &= r \sin(\theta) \sin(\phi) \\
            z(r, \theta, \phi) &= r \cos(\theta)
        \end{align*}
    """
    x = r * jnp.sin(theta) * jnp.cos(phi)  # x = r * sin(theta) * cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)  # y = r * sin(theta) * sin(phi)
    z = r * jnp.cos(theta)  # z = r * cos(theta)
    return x, y, z


def cart_to_spherical(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""
    .. math::
        \begin{align*}
            \rho(x, y, z) &= \sqrt{x^2 + y^2 + z^2} \\
            \theta(x, y, z) &= \arccos\left(\frac{z}{\rho}\right) \\
            \phi(x, y, z) &= \arctan\left(\frac{y}{x}\right)
        \end{align*}
    """
    ρ = jnp.sqrt(x * x + y * y + z * z)
    safe_ρ = jnp.where(ρ == 0.0, 1.0, ρ)
    θ = jnp.arccos(jnp.where(ρ == 0.0, jnp.inf, z / safe_ρ))
    φ = jnp.arctan2(y, x)
    return ρ, θ, φ
