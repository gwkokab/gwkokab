# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


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


def _mass_ratio(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    safe_m1 = jnp.where(m1 == 0.0, 1.0, m1)
    return jnp.where(m1 == 0.0, jnp.inf, m2 / safe_m1)


def mass_ratio(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        q(m_1, m_2) = \frac{m_2}{m_1}
    """
    return jax.jit(_mass_ratio, inline=True)(m1=m1, m2=m2)


def _chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return jnp.power(m1m2, 0.6) * jnp.power(M, -0.2)


def chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        M_c(m_1, m_2) = \frac{(m_1m_2)^{3/5}}{(m_1 + m_2)^{1/5}}
    """
    return jax.jit(_chirp_mass, inline=True)(m1=m1, m2=m2)


def _log_chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    M = total_mass(m1=m1, m2=m2)
    return 0.6 * (jnp.log(m1) + jnp.log(m2)) - 0.2 * jnp.log(M)


def log_chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \log(M_c(m_1, m_2)) = 3/5\times (\log(m_1) + \log(m_2)) - \log(m_1 + m_2)/5
    """
    return jax.jit(_log_chirp_mass, inline=True)(m1=m1, m2=m2)


def _symmetric_mass_ratio(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return m1m2 * jnp.power(M, -2.0)


def symmetric_mass_ratio(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \eta(m_1, m_2) = \frac{m_1m_2}{(m_1 + m_2)^2}
    """
    return jax.jit(_symmetric_mass_ratio, inline=True)(m1=m1, m2=m2)


def _reduced_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    M = total_mass(m1=m1, m2=m2)
    m1m2 = m1_times_m2(m1=m1, m2=m2)
    return jnp.where(M == 0.0, jnp.inf, m1m2 / M)


def reduced_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        M_r(m_1, m_2) = \frac{m_1m_2}{m_1 + m_2}
    """
    return jax.jit(_reduced_mass, inline=True)(m1=m1, m2=m2)


def _delta_m(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    diff = m1 - m2
    M = total_mass(m1=m1, m2=m2)
    return jnp.where(M == 0.0, jnp.inf, diff / M)


def delta_m(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \delta_m(m_1, m_2) = \frac{m_1 - m_2}{m_1 + m_2}
    """
    return jax.jit(_delta_m, inline=True)(m1=m1, m2=m2)


def _delta_m_to_symmetric_mass_ratio(delta_m: ArrayLike) -> ArrayLike:
    delta_m_sq = jnp.square(delta_m)  # delta_m^2
    eta = 0.25 * (1 - delta_m_sq)  # (1 - delta_m^2) / 4
    return eta


def delta_m_to_symmetric_mass_ratio(delta_m: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \eta(\delta_m) = \frac{1 - \delta_m^2}{4}
    """
    return jax.jit(_delta_m_to_symmetric_mass_ratio, inline=True)(delta_m=delta_m)


def _symmetric_mass_ratio_to_delta_m(eta: ArrayLike) -> ArrayLike:
    eta_4 = jnp.multiply(eta, 4)  #  eta*4
    delta_m = jnp.sqrt(jnp.subtract(1, eta_4))  # sqrt(1 - 4 * eta)
    return delta_m


def symmetric_mass_ratio_to_delta_m(eta: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \delta_m(\eta) = \sqrt{1 - 4\eta}
    """
    return jax.jit(_symmetric_mass_ratio_to_delta_m, inline=True)(eta=eta)


def _m_det_z_to_m_source(m_det: ArrayLike, z: ArrayLike) -> ArrayLike:
    safe_one_more_z = jnp.where(z == -1.0, 1.0, jnp.add(1.0, z))
    m_source = jnp.where(z == -1.0, jnp.inf, m_det / safe_one_more_z)
    return m_source


def m_det_z_to_m_source(m_det: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_{\text{source}}(m_{\text{det}}, z) = \frac{m_{\text{det}}}{1 + z}
    """
    return jax.jit(_m_det_z_to_m_source, inline=True)(m_det=m_det, z=z)


def _m_source_z_to_m_det(m_source: ArrayLike, z: ArrayLike) -> ArrayLike:
    return m_source * (1.0 + z)


def m_source_z_to_m_det(m_source: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_{\text{det}}(m_{\text{source}}, z) = m_{\text{source}}(1 + z)
    """
    return jax.jit(_m_source_z_to_m_det, inline=True)(m_source=m_source, z=z)


def m1_q_to_m2(m1: ArrayLike, q: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_2(m_1, q) = m_1q
    """
    return m1 * q


def _m2_q_to_m1(m2: ArrayLike, q: ArrayLike) -> ArrayLike:
    safe_q = jnp.where(q == 0.0, 1.0, q)
    m1 = jnp.where(q == 0.0, jnp.inf, m2 / safe_q)
    return m1


def m2_q_to_m1(m2: ArrayLike, q: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        m_1(m_2, q) = \frac{m_2}{q}
    """
    return jax.jit(_m2_q_to_m1, inline=True)(m2=m2, q=q)


def chi_costilt_to_chiz(chi: ArrayLike, costilt: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \chi_z(\chi, \cos(\theta)) = \chi \cos(\theta)
    """
    return chi * costilt


def _m1_m2_chi1z_chi2z_to_chiminus(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    m1_chi1z = m1 * chi1z
    m2_chi2z = m2 * chi2z
    M = total_mass(m1=m1, m2=m2)
    diff = m1_chi1z - m2_chi2z
    return jnp.where(M == 0, jnp.inf, diff / M)


def m1_m2_chi1z_chi2z_to_chiminus(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    r"""
    .. math::
        \chi_{\text{minus}}(m_1, m_2, \chi_{1z}, \chi_{2z}) = \frac{m_1\chi_{1z} - m_2\chi_{2z}}{m_1 + m_2}
    """
    return jax.jit(_m1_m2_chi1z_chi2z_to_chiminus, inline=True)(
        m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z
    )


def _chieff(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    m1_chi1z = m1 * chi1z
    m2_chi2z = m2 * chi2z
    M = total_mass(m1=m1, m2=m2)
    m_dot_chi = m1_chi1z + m2_chi2z
    return jnp.where(M == 0, jnp.inf, m_dot_chi / M)


def chieff(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    r"""
    .. math::
        \chi_{\text{eff}}(m_1, m_2, \chi_{1z}, \chi_{2z}) = \frac{m_1\chi_{1z} + m_2\chi_{2z}}{m_1 + m_2}
    """
    return jax.jit(_chieff, inline=True)(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


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
    chi1z = chi_costilt_to_chiz(chi=chi1, costilt=costilt1)
    chi2z = chi_costilt_to_chiz(chi=chi2, costilt=costilt2)
    return chieff(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


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
    chi1z = chi_costilt_to_chiz(chi=chi1, costilt=costilt1)
    chi2z = chi_costilt_to_chiz(chi=chi2, costilt=costilt2)
    return m1_m2_chi1z_chi2z_to_chiminus(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


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
    chi1z = jnp.divide(
        jnp.multiply(half_M, jnp.add(chieff, chiminus)), m1
    )  # chi1z = M/2 * (chieff + chiminus) / m1
    chi2z = jnp.divide(
        jnp.multiply(half_M, jnp.subtract(chieff, chiminus)), m2
    )  # chi2z = M/2 * (chieff - chiminus) / m2
    return chi1z, chi2z


def Mc_eta_to_m1_m2(Mc: ArrayLike, eta: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    r"""
    .. math::
        \begin{align*}
            m_1(M_c, \eta) &= \frac{M_c}{2} \eta^{-0.6} (1 + \sqrt{1 - 4\eta}) \\
            m_2(M_c, \eta) &= \frac{M_c}{2} \eta^{-0.6} (1 - \sqrt{1 - 4\eta})
        \end{align*}
    """
    delta_sq = jnp.subtract(1, jnp.multiply(4.0, eta))  # 1 - 4 * eta
    delta_sq = jnp.maximum(
        delta_sq, jnp.zeros_like(delta_sq)
    )  # to avoid negative values
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


def _eta_from_q(q: ArrayLike) -> ArrayLike:
    safe_q_from_neg = jnp.where(q <= 0, 1, q)
    log_eta = jnp.where(
        q <= 0, -jnp.inf, jnp.log(safe_q_from_neg) - 2.0 * jnp.log1p(safe_q_from_neg)
    )
    return jnp.exp(log_eta)


def eta_from_q(q: ArrayLike) -> ArrayLike:
    r"""
    .. math::
        \eta(q) = \frac{q}{(1 + q)^2}
    """
    return jax.jit(_eta_from_q, inline=True)(q)


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
    theta = jnp.arctan2(y, x)
    return r, theta


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
            r(x, y, z) &= \sqrt{x^2 + y^2 + z^2} \\
            \theta(x, y, z) &= \arccos\left(\frac{z}{r}\right) \\
            \phi(x, y, z) &= \arctan\left(\frac{y}{x}\right)
        \end{align*}
    """
    r = jnp.sqrt(x * x + y * y + z * z)  # r = sqrt(x^2 + y^2 + z^2)
    safe_r = jnp.where(r == 0.0, 1.0, r)
    theta = jnp.arccos(
        jnp.where(r == 0.0, jnp.inf, z / safe_r)
    )  # theta = arccos(z / r)
    phi = jnp.arctan2(y, x)  # phi = arctan(y / x)
    return r, theta, phi
