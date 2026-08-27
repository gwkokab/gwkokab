# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure formulas relating gravitational-wave source parameters.

Every function here is a closed-form algebraic map between source parameters -- masses,
mass ratios, spin components, coordinate systems -- written in terms of :mod:`jax.numpy`
so it is differentiable and traceable. There is no state and no cosmology:
redshift/distance conversions live in :mod:`gwkokab.cosmology`.

These are the edges of the relation graph in :mod:`gwkokab.parameters`; the
:class:`~gwkokab.parameters.RelationMesh` composes them to derive whichever parameters
are reachable from those a dataset already carries, which is what backs the ``--derive-
parameters`` flag of the synthetic-data CLIs.
"""

from jax import numpy as jnp
from jaxtyping import ArrayLike


def m1_times_m2(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    """Product of the two component masses.

    .. math::
        m_1m_2(m_1, m_2) = m_1 m_2

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.

    Returns
    -------
    ArrayLike
        The product :math:`m_1 m_2`.
    """
    return m1 * m2


def total_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    """Total mass of the binary.

    .. math::
        M(m_1, m_2) = m_1 + m_2

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.

    Returns
    -------
    ArrayLike
        The total mass :math:`M`.
    """
    return m1 + m2


def mass_ratio(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""Mass ratio of the binary.

    .. math::
        q(m_1, m_2) = \frac{m_2}{m_1}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.

    Returns
    -------
    ArrayLike
        The mass ratio :math:`q \in (0, 1]` for :math:`m_1 \geq m_2`.
    """
    return m2 / m1


def chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""Chirp mass of the binary.

    .. math::
        M_c(m_1, m_2) = \frac{(m_1m_2)^{3/5}}{(m_1 + m_2)^{1/5}}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.

    Returns
    -------
    ArrayLike
        The chirp mass :math:`M_c`.
    """
    return jnp.power(m1 * m2, 0.6) / jnp.power(m1 + m2, 0.2)


def log_chirp_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""Natural logarithm of the chirp mass.

    .. math::
        \log(M_c(m_1, m_2)) = 3/5\times (\log(m_1) + \log(m_2)) - \log(m_1 + m_2)/5

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.

    Returns
    -------
    ArrayLike
        The log chirp mass :math:`\log M_c`.

    Notes
    -----
    Computed directly in log space, so it stays finite for chirp masses that would
    underflow when exponentiated.
    """
    return 0.6 * (jnp.log(m1) + jnp.log(m2)) - 0.2 * jnp.log(m1 + m2)


def symmetric_mass_ratio(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""Symmetric mass ratio of the binary.

    .. math::
        \eta(m_1, m_2) = \frac{m_1m_2}{(m_1 + m_2)^2}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.

    Returns
    -------
    ArrayLike
        The symmetric mass ratio :math:`\eta \in (0, 1/4]`.
    """
    return m1 * m2 * jnp.power(m1 + m2, -2.0)


def reduced_mass(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""Reduced mass of the binary.

    .. math::
        M_r(m_1, m_2) = \frac{m_1m_2}{m_1 + m_2}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.

    Returns
    -------
    ArrayLike
        The reduced mass :math:`M_r`.
    """
    return m1 * m2 / (m1 + m2)


def delta_m(m1: ArrayLike, m2: ArrayLike) -> ArrayLike:
    r"""Fractional mass difference of the binary.

    .. math::
        \delta_m(m_1, m_2) = \frac{m_1 - m_2}{m_1 + m_2}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.

    Returns
    -------
    ArrayLike
        The asymmetry :math:`\delta_m \in [0, 1)` for :math:`m_1 \geq m_2`.
    """
    return (m1 - m2) / (m1 + m2)


def delta_m_to_symmetric_mass_ratio(delta_m: ArrayLike) -> ArrayLike:
    r"""Convert a fractional mass difference to a symmetric mass ratio.

    .. math::
        \eta(\delta_m) = \frac{1 - \delta_m^2}{4}

    Parameters
    ----------
    delta_m : ArrayLike
        The fractional mass difference :math:`\delta_m`.

    Returns
    -------
    ArrayLike
        The symmetric mass ratio :math:`\eta`.
    """
    delta_m_sq = jnp.square(delta_m)  # delta_m^2
    eta = 0.25 * (1 - delta_m_sq)  # (1 - delta_m^2) / 4
    return eta


def symmetric_mass_ratio_to_delta_m(eta: ArrayLike) -> ArrayLike:
    r"""Convert a symmetric mass ratio to a fractional mass difference.

    .. math::
        \delta_m(\eta) = \sqrt{1 - 4\eta}

    Parameters
    ----------
    eta : ArrayLike
        The symmetric mass ratio :math:`\eta \in (0, 1/4]`.

    Returns
    -------
    ArrayLike
        The fractional mass difference :math:`\delta_m`.
    """
    eta_4 = jnp.multiply(eta, 4)  #  eta*4
    delta_m = jnp.sqrt(jnp.subtract(1, eta_4))  # sqrt(1 - 4 * eta)
    return delta_m


def m_det_z_to_m_source(m_det: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""Convert a detector-frame mass to the source frame.

    .. math::
        m_{\text{source}}(m_{\text{det}}, z) = \frac{m_{\text{det}}}{1 + z}

    Parameters
    ----------
    m_det : ArrayLike
        Detector-frame (redshifted) mass.
    z : ArrayLike
        Redshift of the source.

    Returns
    -------
    ArrayLike
        The source-frame mass.
    """
    return m_det / (1.0 + z)


def m_source_z_to_m_det(m_source: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""Convert a source-frame mass to the detector frame.

    .. math::
        m_{\text{det}}(m_{\text{source}}, z) = m_{\text{source}}(1 + z)

    Parameters
    ----------
    m_source : ArrayLike
        Source-frame mass.
    z : ArrayLike
        Redshift of the source.

    Returns
    -------
    ArrayLike
        The detector-frame (redshifted) mass.
    """
    return m_source * (1.0 + z)


def m1_q_to_m2(m1: ArrayLike, q: ArrayLike) -> ArrayLike:
    r"""Recover the secondary mass from the primary mass and mass ratio.

    .. math::
        m_2(m_1, q) = m_1q

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    q : ArrayLike
        Mass ratio :math:`q = m_2/m_1 \in (0, 1]`.

    Returns
    -------
    ArrayLike
        The secondary mass :math:`m_2`.
    """
    return m1 * q


def m2_q_to_m1(m2: ArrayLike, q: ArrayLike) -> ArrayLike:
    r"""Recover the primary mass from the secondary mass and mass ratio.

    .. math::
        m_1(m_2, q) = \frac{m_2}{q}

    Parameters
    ----------
    m2 : ArrayLike
        Secondary (lighter) component mass.
    q : ArrayLike
        Mass ratio :math:`q = m_2/m_1 \in (0, 1]`.

    Returns
    -------
    ArrayLike
        The primary mass :math:`m_1`.
    """
    return m2 / q


def chi_costilt_to_chiz(chi: ArrayLike, costilt: ArrayLike) -> ArrayLike:
    r"""Project a spin magnitude onto the orbital angular momentum axis.

    .. math::
        \chi_z(\chi, \cos(\theta)) = \chi \cos(\theta)

    Parameters
    ----------
    chi : ArrayLike
        Dimensionless spin magnitude :math:`\chi \in [0, 1]`.
    costilt : ArrayLike
        Cosine of the tilt angle between the spin and the orbital
        angular momentum, :math:`\cos\theta \in [-1, 1]`.

    Returns
    -------
    ArrayLike
        The aligned spin component :math:`\chi_z`.
    """
    return chi * costilt


def m1_m2_chi1z_chi2z_to_chiminus(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    r"""Mass-weighted antisymmetric aligned spin.

    .. math::
        \chi_{\text{minus}}(m_1, m_2, \chi_{1z}, \chi_{2z}) = \frac{m_1\chi_{1z} - m_2\chi_{2z}}{m_1 + m_2}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.
    chi1z : ArrayLike
        Aligned spin component of the primary.
    chi2z : ArrayLike
        Aligned spin component of the secondary.

    Returns
    -------
    ArrayLike
        The antisymmetric spin :math:`\chi_{\text{minus}}`.
    """
    m1_chi1z = m1 * chi1z
    m2_chi2z = m2 * chi2z
    M = m1 + m2
    diff = m1_chi1z - m2_chi2z
    return diff / M


def chieff(
    m1: ArrayLike, m2: ArrayLike, chi1z: ArrayLike, chi2z: ArrayLike
) -> ArrayLike:
    r"""Effective inspiral spin of the binary.

    .. math::
        \chi_{\text{eff}}(m_1, m_2, \chi_{1z}, \chi_{2z}) = \frac{m_1\chi_{1z} + m_2\chi_{2z}}{m_1 + m_2}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.
    chi1z : ArrayLike
        Aligned spin component of the primary.
    chi2z : ArrayLike
        Aligned spin component of the secondary.

    Returns
    -------
    ArrayLike
        The effective spin :math:`\chi_{\text{eff}} \in [-1, 1]`.
    """
    m1_chi1z = m1 * chi1z
    m2_chi2z = m2 * chi2z
    M = m1 + m2
    m_dot_chi = m1_chi1z + m2_chi2z
    return m_dot_chi / M


def m1_m2_chi1_chi2_costilt1_costilt2_to_chieff(
    *,
    m1: ArrayLike,
    m2: ArrayLike,
    chi1: ArrayLike,
    chi2: ArrayLike,
    costilt1: ArrayLike,
    costilt2: ArrayLike,
) -> ArrayLike:
    r"""Effective inspiral spin from spin magnitudes and tilt angles.

    .. math::
        \chi_{\text{eff}}(m_1, m_2, \chi_1, \chi_2, \cos(\theta_1), \cos(\theta_2)) =
        \frac{m_1\chi_1\cos(\theta_1) + m_2\chi_2\cos(\theta_2)}{m_1 + m_2}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.
    chi1 : ArrayLike
        Dimensionless spin magnitude of the primary.
    chi2 : ArrayLike
        Dimensionless spin magnitude of the secondary.
    costilt1 : ArrayLike
        Cosine of the primary tilt angle.
    costilt2 : ArrayLike
        Cosine of the secondary tilt angle.

    Returns
    -------
    ArrayLike
        The effective spin :math:`\chi_{\text{eff}} \in [-1, 1]`.
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
    r"""Antisymmetric aligned spin from spin magnitudes and tilt angles.

    .. math::
        \chi_{\text{minus}}(m_1, m_2, \chi_1, \chi_2, \cos(\theta_1), \cos(\theta_2)) =
        \frac{m_1\chi_1\cos(\theta_1) - m_2\chi_2\cos(\theta_2)}{m_1 + m_2}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.
    chi1 : ArrayLike
        Dimensionless spin magnitude of the primary.
    chi2 : ArrayLike
        Dimensionless spin magnitude of the secondary.
    costilt1 : ArrayLike
        Cosine of the primary tilt angle.
    costilt2 : ArrayLike
        Cosine of the secondary tilt angle.

    Returns
    -------
    ArrayLike
        The antisymmetric spin :math:`\chi_{\text{minus}}`.
    """
    chi1z = chi_costilt_to_chiz(chi=chi1, costilt=costilt1)
    chi2z = chi_costilt_to_chiz(chi=chi2, costilt=costilt2)
    return m1_m2_chi1z_chi2z_to_chiminus(m1=m1, m2=m2, chi1z=chi1z, chi2z=chi2z)


def m1_m2_chieff_chiminus_to_chi1z_chi2z(
    m1: ArrayLike, m2: ArrayLike, chieff: ArrayLike, chiminus: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    r"""Invert the effective/antisymmetric spin pair back to aligned spin components.

    .. math::
        \begin{align*}
        \chi_{1z}(m_1, m_2, \chi_{\text{eff}}, \chi_{\text{minus}}) &=
        \frac{m_1+m_2}{2m_1} \left( \chi_{\text{eff}} + \chi_{\text{minus}} \right)\\
        \chi_{2z}(m_1, m_2, \chi_{\text{eff}}, \chi_{\text{minus}}) &=
        \frac{m_1+m_2}{2m_2} \left( \chi_{\text{eff}} - \chi_{\text{minus}} \right)
        \end{align*}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    m2 : ArrayLike
        Secondary (lighter) component mass.
    chieff : ArrayLike
        Effective inspiral spin :math:`\chi_{\text{eff}}`.
    chiminus : ArrayLike
        Antisymmetric aligned spin :math:`\chi_{\text{minus}}`.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        The aligned spin components :math:`(\chi_{1z}, \chi_{2z})`.
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
    r"""Recover the component masses from chirp mass and symmetric mass ratio.

    .. math::
        \begin{align*}
            m_1(M_c, \eta) &= \frac{M_c}{2} \eta^{-0.6} (1 + \sqrt{1 - 4\eta}) \\
            m_2(M_c, \eta) &= \frac{M_c}{2} \eta^{-0.6} (1 - \sqrt{1 - 4\eta})
        \end{align*}

    Parameters
    ----------
    Mc : ArrayLike
        Chirp mass :math:`M_c`.
    eta : ArrayLike
        Symmetric mass ratio :math:`\eta \in (0, 1/4]`.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        The component masses :math:`(m_1, m_2)` with :math:`m_1 \geq m_2`.

    Notes
    -----
    The discriminant :math:`1 - 4\eta` is clipped at zero, so :math:`\eta` marginally
    above :math:`1/4` (from round-off) yields the equal-mass solution rather than NaN.
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


def eta_from_q(q: ArrayLike) -> ArrayLike:
    r"""Convert a mass ratio to a symmetric mass ratio.

    .. math::
        \eta(q) = \frac{q}{(1 + q)^2}

    Parameters
    ----------
    q : ArrayLike
        Mass ratio :math:`q = m_2/m_1 \in (0, 1]`.

    Returns
    -------
    ArrayLike
        The symmetric mass ratio :math:`\eta`.
    """
    return q / (1.0 + q) ** 2.0


def q_from_eta(eta: ArrayLike) -> ArrayLike:
    r"""

    .. math ::
        q(\eta) = \frac{1 - 2\eta - \sqrt{1 - 4\eta}}{2\eta}

    Parameters
    ----------
    eta : ArrayLike
        Symmetric mass ratio

    Returns
    -------
    ArrayLike
        Mass ratio
    """
    return (1.0 - 2.0 * eta - jnp.sqrt(1.0 - 4.0 * eta)) / (2.0 * eta)


def polar_to_cart(r: ArrayLike, theta: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    r"""Convert plane polar coordinates to Cartesian coordinates.

    .. math::
        \begin{align*}
            x(r, \theta) &= r \cos(\theta) \\
            y(r, \theta) &= r \sin(\theta)
        \end{align*}

    Parameters
    ----------
    r : ArrayLike
        Radius :math:`r`.
    theta : ArrayLike
        Polar angle :math:`\theta`, in radians.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        The Cartesian coordinates :math:`(x, y)`.
    """
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return x, y


def cart_to_polar(x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    r"""Convert Cartesian coordinates to plane polar coordinates.

    .. math::
        \begin{align*}
            r(x, y) &= \sqrt{x^2 + y^2} \\
            \theta(x, y) &= \arctan(y/x)
        \end{align*}

    Parameters
    ----------
    x : ArrayLike
        Cartesian :math:`x` coordinate.
    y : ArrayLike
        Cartesian :math:`y` coordinate.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        The polar coordinates :math:`(r, \theta)`, with :math:`\theta \in (-\pi, \pi]`
        from :func:`jax.numpy.arctan2`.
    """
    r = jnp.sqrt(x * x + y * y)
    theta = jnp.arctan2(y, x)
    return r, theta


def spherical_to_cart(
    r: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Convert spherical polar coordinates to Cartesian coordinates.

    .. math::
        \begin{align*}
            x(r, \theta, \phi) &= r \sin(\theta) \cos(\phi) \\
            y(r, \theta, \phi) &= r \sin(\theta) \sin(\phi) \\
            z(r, \theta, \phi) &= r \cos(\theta)
        \end{align*}

    Parameters
    ----------
    r : ArrayLike
        Radius :math:`r`.
    theta : ArrayLike
        Polar angle :math:`\theta` from the :math:`z` axis, in radians.
    phi : ArrayLike
        Azimuthal angle :math:`\phi`, in radians.

    Returns
    -------
    tuple[ArrayLike, ArrayLike, ArrayLike]
        The Cartesian coordinates :math:`(x, y, z)`.
    """
    x = r * jnp.sin(theta) * jnp.cos(phi)  # x = r * sin(theta) * cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)  # y = r * sin(theta) * sin(phi)
    z = r * jnp.cos(theta)  # z = r * cos(theta)
    return x, y, z


def cart_to_spherical(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""Convert Cartesian coordinates to spherical polar coordinates.

    .. math::
        \begin{align*}
            r(x, y, z) &= \sqrt{x^2 + y^2 + z^2} \\
            \theta(x, y, z) &= \arccos\left(\frac{z}{r}\right) \\
            \phi(x, y, z) &= \arctan\left(\frac{y}{x}\right)
        \end{align*}

    Parameters
    ----------
    x : ArrayLike
        Cartesian :math:`x` coordinate.
    y : ArrayLike
        Cartesian :math:`y` coordinate.
    z : ArrayLike
        Cartesian :math:`z` coordinate.

    Returns
    -------
    tuple[ArrayLike, ArrayLike, ArrayLike]
        The spherical coordinates :math:`(r, \theta, \phi)`.

    Notes
    -----
    At the origin :math:`\theta` is returned as NaN (``arccos`` of infinity) rather
    than raising, and the division is guarded so the gradient stays finite.
    """
    r = jnp.sqrt(x * x + y * y + z * z)  # r = sqrt(x^2 + y^2 + z^2)
    safe_r = jnp.where(r == 0.0, 1.0, r)
    theta = jnp.arccos(
        jnp.where(r == 0.0, jnp.inf, z / safe_r)
    )  # theta = arccos(z / r)
    phi = jnp.arctan2(y, x)  # phi = arctan(y / x)
    return r, theta, phi


def sin_tilt(costilt: ArrayLike) -> ArrayLike:
    r"""Sine of a tilt angle given its cosine.

    .. math::
        \sin(\theta) = \sqrt{1 - \cos^2(\theta)}

    Parameters
    ----------
    costilt : ArrayLike
        Cosine of the tilt angle, :math:`\cos\theta \in [-1, 1]`.

    Returns
    -------
    ArrayLike
        The non-negative sine :math:`\sin\theta`.
    """
    return jnp.sqrt(1 - jnp.square(costilt))


def chi_p_from_components(
    a_1: ArrayLike,
    cos_tilt_1: ArrayLike,
    a_2: ArrayLike,
    cos_tilt_2: ArrayLike,
    mass_ratio: ArrayLike,
) -> ArrayLike:
    r"""Effective precessing spin of the binary.

    .. math::
        \chi_p(a_1, \cos(\theta_1), a_2, \cos(\theta_2), q) =
        \max \left(
            a_1 \sin(\theta_1),
            \frac{3 + 4q}{4 + 3q} q a_2 \sin(\theta_2)
        \right)

    Parameters
    ----------
    a_1 : ArrayLike
        Dimensionless spin magnitude of the primary.
    cos_tilt_1 : ArrayLike
        Cosine of the primary tilt angle.
    a_2 : ArrayLike
        Dimensionless spin magnitude of the secondary.
    cos_tilt_2 : ArrayLike
        Cosine of the secondary tilt angle.
    mass_ratio : ArrayLike
        Mass ratio :math:`q = m_2/m_1 \in (0, 1]`.

    Returns
    -------
    ArrayLike
        The precessing spin :math:`\chi_p \in [0, 1]`.
    """
    sin_tilt_1 = sin_tilt(cos_tilt_1)
    sin_tilt_2 = sin_tilt(cos_tilt_2)
    return jnp.maximum(
        a_1 * sin_tilt_1,
        (3.0 + 4.0 * mass_ratio)
        / (4.0 + 3.0 * mass_ratio)
        * mass_ratio
        * a_2
        * sin_tilt_2,
    )


def spin_magnitude_from_components(
    chi_x: ArrayLike, chi_y: ArrayLike, chi_z: ArrayLike
) -> ArrayLike:
    r"""Spin magnitude from its Cartesian components.

    .. math::
        \chi(\chi_x, \chi_y, \chi_z) = \sqrt{\chi_x^2 + \chi_y^2 + \chi_z^2}

    Parameters
    ----------
    chi_x : ArrayLike
        Cartesian :math:`x` component of the spin.
    chi_y : ArrayLike
        Cartesian :math:`y` component of the spin.
    chi_z : ArrayLike
        Cartesian :math:`z` component of the spin, aligned with the orbital
        angular momentum.

    Returns
    -------
    ArrayLike
        The dimensionless spin magnitude :math:`\chi`.
    """
    return jnp.sqrt(jnp.square(chi_x) + jnp.square(chi_y) + jnp.square(chi_z))


def spin_costilt_from_components(
    chi_x: ArrayLike, chi_y: ArrayLike, chi_z: ArrayLike
) -> ArrayLike:
    r"""Cosine of the spin tilt angle from the Cartesian spin components.

    .. math::
        \cos(\theta)(\chi_x, \chi_y, \chi_z) = \frac{\chi_z}{\sqrt{\chi_x^2 + \chi_y^2 + \chi_z^2}}

    Parameters
    ----------
    chi_x : ArrayLike
        Cartesian :math:`x` component of the spin.
    chi_y : ArrayLike
        Cartesian :math:`y` component of the spin.
    chi_z : ArrayLike
        Cartesian :math:`z` component of the spin, aligned with the orbital
        angular momentum.

    Returns
    -------
    ArrayLike
        The cosine of the tilt angle, :math:`\cos\theta \in [-1, 1]`.

    Notes
    -----
    For a vanishing spin the tilt is undefined; infinity is returned instead of a
    division by zero, and the division is guarded so the gradient stays finite.
    """
    spin_magnitude = spin_magnitude_from_components(chi_x, chi_y, chi_z)
    safe_spin_magnitude = jnp.where(spin_magnitude == 0.0, 1.0, spin_magnitude)
    return jnp.where(spin_magnitude == 0.0, jnp.inf, chi_z / safe_spin_magnitude)


def chirp_mass_from_m1_q(m1: ArrayLike, q: ArrayLike) -> ArrayLike:
    r"""Chirp mass from the primary mass and mass ratio.

    .. math::
        M_c(m_1, q) = \frac{(m_1^3 q^3)^{1/5}}{(m_1 + m_1 q)^{1/5}} = m_1 \frac{q^{3/5}}{(1 + q)^{1/5}}

    Parameters
    ----------
    m1 : ArrayLike
        Primary (heavier) component mass.
    q : ArrayLike
        Mass ratio :math:`q = m_2/m_1 \in (0, 1]`.

    Returns
    -------
    ArrayLike
        The chirp mass :math:`M_c`.
    """
    return m1 * jnp.power(q, 0.6) / jnp.power(1 + q, 0.2)
