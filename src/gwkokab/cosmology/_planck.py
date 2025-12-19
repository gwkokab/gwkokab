# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from ._cosmology import Cosmology


def PLANCK_2015_Cosmology() -> Cosmology:
    """Cosmology : See Table4 in arXiv:1502.01589, OmegaMatter from astropy Planck 2015"""
    PLANCK_2015_Ho = 67.74 * 1e3
    PLANCK_2015_OmegaMatter = 0.3089
    PLANCK_2015_OmegaLambda = 1.0 - PLANCK_2015_OmegaMatter
    PLANCK_2015_OmegaRadiation = 0.0

    return Cosmology(
        PLANCK_2015_Ho,
        PLANCK_2015_OmegaMatter,
        PLANCK_2015_OmegaRadiation,
        PLANCK_2015_OmegaLambda,
    )


def PLANCK_2018_Cosmology() -> Cosmology:
    """Cosmology : See Table1 in arXiv:1807.06209"""
    PLANCK_2018_Ho = 67.32 * 1e3
    PLANCK_2018_OmegaMatter = 0.3158
    PLANCK_2018_OmegaLambda = 1.0 - PLANCK_2018_OmegaMatter
    PLANCK_2018_OmegaRadiation = 0.0

    return Cosmology(
        PLANCK_2018_Ho,
        PLANCK_2018_OmegaMatter,
        PLANCK_2018_OmegaRadiation,
        PLANCK_2018_OmegaLambda,
    )
