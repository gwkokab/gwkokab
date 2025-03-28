# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from jaxtyping import Array


def constraint(
    x: Array,
    has_spin: bool,
    has_tilt: bool,
    has_eccentricity: bool,
    has_redshift: bool,
) -> Array:
    """Applies physical constraints to the input array.

    :param x: Input array where:

        - x[..., 0] is mass m1
        - x[..., 1] is mass m2
        - x[..., 2] is chi1 (if has_spin is True)
        - x[..., 3] is chi2 (if has_spin is True)
        - x[..., 4] is cos_tilt_1 (if has_tilt is True)
        - x[..., 5] is cos_tilt_2 (if has_tilt is True)
        - x[..., 6] is eccentricity (if has_eccentricity is True)

    :param has_spin: Whether to apply spin constraints.
    :param has_tilt: Whether to apply tilt constraints.
    :param has_eccentricity: Whether to apply eccentricity constraints.
    :return: Boolean array indicating which samples satisfy the constraints.
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

    return mask
