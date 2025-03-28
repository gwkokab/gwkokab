# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from jaxtyping import Array


def constraint(
    x: Array,
    has_spin: bool,
    has_tilt: bool,
    has_eccentricity: bool,
    has_redshift: bool,
    has_cos_inclination: bool,
    has_phi_12: bool,
    has_polarization_angle: bool,
    has_right_ascension: bool,
    has_sin_declination: bool,
) -> Array:
    """Applies physical constraints to the input array.

    Parameters
    ----------

    x : Array
        Input array where:

        - x[..., 0] is mass m1
        - x[..., 1] is mass m2
        - x[..., 2] is chi1 (if has_spin is True)
        - x[..., 3] is chi2 (if has_spin is True)
        - x[..., 4] is cos_tilt_1 (if has_tilt is True)
        - x[..., 5] is cos_tilt_2 (if has_tilt is True)
        - x[..., 6] is eccentricity (if has_eccentricity is True)
        - x[..., 7] is redshift (if has_redshift is True)
        - x[..., 8] is cos_inclination (if has_cos_inclination is True)
        - x[..., 9] is phi_12 (if has_phi_12 is True)
        - x[..., 10] is polarization_angle (if has_polarization_angle is True)
        - x[..., 11] is right_ascension (if has_right_ascension is True)
        - x[..., 12] is sin_declination (if has_sin_declination is True)

    has_spin : bool
        Whether to apply spin constraints.
    has_tilt : bool
        Whether to apply tilt constraints.
    has_eccentricity : bool
        Whether to apply eccentricity constraints.
    has_redshift : bool
        Whether to apply redshift constraints.
    has_cos_inclination : bool
        Whether to apply cos_inclination constraints.
    has_phi_12 : bool
        Whether to apply phi_12 constraints.
    has_polarization_angle : bool
        Whether to apply polarization_angle constraints.
    has_right_ascension : bool
        Whether to apply right_ascension constraints.
    has_sin_declination : bool
        Whether to apply sin_declination constraints.

    Return
    ------
    Array
        Boolean array indicating which samples satisfy the constraints.
    """
    m1 = x[..., 0]
    m2 = x[..., 1]

    mask = m1 > 0.0
    mask &= m2 > 0.0
    mask &= m1 >= m2

    i = 2

    if has_spin:
        chi1 = x[..., i]
        chi2 = x[..., i + 1]

        mask &= chi1 >= 0.0
        mask &= chi1 <= 1.0

        mask &= chi2 >= 0.0
        mask &= chi2 <= 1.0

        i += 2

    if has_tilt:
        cos_tilt_1 = x[..., i]
        cos_tilt_2 = x[..., i + 1]

        mask &= cos_tilt_1 >= -1.0
        mask &= cos_tilt_1 <= 1.0

        mask &= cos_tilt_2 >= -1.0
        mask &= cos_tilt_2 <= 1.0

        i += 2

    if has_eccentricity:
        ecc = x[..., i]

        mask &= ecc >= 0.0
        mask &= ecc <= 1.0

    if has_redshift:
        z = x[..., i]

        mask &= z >= 1e-3

        i += 1

    if has_cos_inclination:
        cos_inclination = x[..., i]

        mask &= cos_inclination >= -1.0
        mask &= cos_inclination <= 1.0

        i += 1

    if has_phi_12:
        phi_12 = x[..., i]

        mask &= phi_12 >= 0.0
        mask &= phi_12 <= 2.0 * np.pi

        i += 1

    if has_polarization_angle:
        polarization_angle = x[..., i]

        mask &= polarization_angle >= 0.0
        mask &= polarization_angle <= np.pi

        i += 1

    if has_right_ascension:
        right_ascension = x[..., i]

        mask &= right_ascension >= 0.0
        mask &= right_ascension <= 2.0 * np.pi

        i += 1

    if has_sin_declination:
        sin_declination = x[..., i]

        mask &= sin_declination >= -1.0
        mask &= sin_declination <= 1.0

        i += 1

    return mask
