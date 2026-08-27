# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Concrete Planck cosmologies and the package-wide default.

Each factory returns a fresh flat :class:`~gwkokab.cosmology.Cosmology` (radiation
density neglected, :math:`\Omega_\Lambda = 1 - \Omega_m`). The default returned by
:func:`default_cosmology` is selected at import time by the
``GWKOKAB_DEFAULT_COSMOLOGY`` environment variable, which must name a key of
:data:`COSMOLOGY_REGISTRY`; an unrecognised name raises
:class:`~gwkokab.utils.exceptions.LoggedValueError` on import.
"""

import os
from types import MappingProxyType
from typing import Final

from gwkokab.utils.exceptions import LoggedValueError

from ._cosmology import Cosmology


def PLANCK_2013_Cosmology() -> Cosmology:
    r"""Planck 2013 cosmology, with :math:`\Omega_m` from astropy's ``Planck13``.

    Returns
    -------
    Cosmology
        A flat :math:`\Lambda\mathrm{CDM}` cosmology with :math:`H_0 = 67.77` km/s/Mpc and
        :math:`\Omega_m = 0.30712`.
    """
    h_0 = 67.77 * 1e3
    omega_m = 0.30712
    return Cosmology(h_0, omega_m, 0.0, 1.0 - omega_m)


def PLANCK_2015_Cosmology() -> Cosmology:
    r"""Planck 2015 cosmology.

    See Table 4 in `arXiv:1502.01589 <https://arxiv.org/abs/1502.01589>`_;
    :math:`\Omega_m` is taken from astropy's ``Planck15``.

    Returns
    -------
    Cosmology
        A flat :math:`\Lambda\mathrm{CDM}` cosmology with :math:`H_0 = 67.74` km/s/Mpc and
        :math:`\Omega_m = 0.3075`.
    """
    h_0 = 67.74 * 1e3
    omega_m = 0.3075
    return Cosmology(h_0, omega_m, 0.0, 1.0 - omega_m)


def PLANCK_2018_Cosmology() -> Cosmology:
    r"""Planck 2018 cosmology.

    See Table 1 in `arXiv:1807.06209 <https://arxiv.org/abs/1807.06209>`_.

    Returns
    -------
    Cosmology
        A flat :math:`\Lambda\mathrm{CDM}` cosmology with :math:`H_0 = 67.66` km/s/Mpc and
        :math:`\Omega_m = 0.30966`.
    """
    h_0 = 67.66 * 1e3
    omega_m = 0.30966
    return Cosmology(h_0, omega_m, 0.0, 1.0 - omega_m)


COSMOLOGY_REGISTRY: Final = MappingProxyType({
    "Planck13": PLANCK_2013_Cosmology,
    "Planck15": PLANCK_2015_Cosmology,
    "Planck18": PLANCK_2018_Cosmology,
})


if (
    name := os.environ.get("GWKOKAB_DEFAULT_COSMOLOGY", "Planck15")
) not in COSMOLOGY_REGISTRY:
    raise LoggedValueError(
        f"Invalid or unavailable cosmology: GWKOKAB_DEFAULT_COSMOLOGY={name}. "
        f"Available options: {list(COSMOLOGY_REGISTRY.keys())}"
    )


def default_cosmology() -> Cosmology:
    """Build the default cosmology named by ``GWKOKAB_DEFAULT_COSMOLOGY``.

    The environment variable is validated once, at import time; this function looks the
    validated name up in :data:`COSMOLOGY_REGISTRY` and constructs a fresh instance.

    Returns
    -------
    Cosmology
        The default cosmology, ``Planck15`` unless overridden.
    """
    return COSMOLOGY_REGISTRY[name]()
