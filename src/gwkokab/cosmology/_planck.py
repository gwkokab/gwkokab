# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
from types import MappingProxyType
from typing import Final

from gwkokab.utils.exceptions import LoggedValueError

from ._cosmology import Cosmology


def PLANCK_2013_Cosmology() -> Cosmology:
    h_0 = 67.77 * 1e3
    omega_m = 0.30712
    return Cosmology(h_0, omega_m, 0.0, 1.0 - omega_m)


def PLANCK_2015_Cosmology() -> Cosmology:
    """Cosmology: See Table 4 in arXiv:1502.01589, OmegaMatter from astropy Planck
    2015.
    """
    h_0 = 67.74 * 1e3
    omega_m = 0.3075
    return Cosmology(h_0, omega_m, 0.0, 1.0 - omega_m)


def PLANCK_2018_Cosmology() -> Cosmology:
    """Cosmology: See Table 1 in arXiv:1807.06209."""
    h_0 = 67.66 * 1e3
    omega_m = 0.30966
    return Cosmology(h_0, omega_m, 0.0, 1.0 - omega_m)


COSMOLOGY_REGISTRY: Final = MappingProxyType(
    {
        "Planck13": PLANCK_2013_Cosmology,
        "Planck15": PLANCK_2015_Cosmology,
        "Planck18": PLANCK_2018_Cosmology,
    }
)


if (
    name := os.environ.get("GWKOKAB_DEFAULT_COSMOLOGY", "Planck15")
) not in COSMOLOGY_REGISTRY:
    raise LoggedValueError(
        f"Invalid or unavailable cosmology: GWKOKAB_DEFAULT_COSMOLOGY={name}. "
        f"Available options: {list(COSMOLOGY_REGISTRY.keys())}"
    )


def default_cosmology() -> Cosmology:
    """Returns the default cosmology based on GWKOKAB_DEFAULT_COSMOLOGY."""
    return COSMOLOGY_REGISTRY[name]()
