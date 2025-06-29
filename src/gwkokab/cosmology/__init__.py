# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


#
"""Cosmology module for GWKokab, adapted from code written by
`Reed Essick <https://orcid.org/0000-0001-8196-9267>`_ included in the
`gw-distributions <https://git.ligo.org/reed.essick/gw-distributions>`_ package at
`source <https://git.ligo.org/reed.essick/gw-distributions/-/blob/master/gwdistributions/utils/cosmology.py>`_."""

from ._cosmology import Cosmology as Cosmology


PLANCK_2015_Cosmology: Cosmology
"""See Table4 in arXiv:1502.01589, OmegaMatter from astropy Planck 2015."""

PLANCK_2018_Cosmology: Cosmology
""""See Table1 in arXiv:1807.06209."""


def __getattr__(name: str) -> Cosmology:
    """Lazy load cosmology classes based on the name provided.

    Parameters
    ----------
    name : str
        The name of the cosmology class to load.

    Returns
    -------
    Cosmology
        The cosmology class corresponding to the provided name.

    Raises
    ------
    AttributeError
        If the requested cosmology class does not exist.
    """
    match name:
        case "PLANCK_2015_Cosmology":
            from ._planck import PLANCK_2015_Cosmology as _PLANCK_2015_Cosmology

            return _PLANCK_2015_Cosmology

        case "PLANCK_2018_Cosmology":
            from ._planck import PLANCK_2018_Cosmology as _PLANCK_2018_Cosmology

            return _PLANCK_2018_Cosmology

        case _:
            raise AttributeError(f"module {__name__} has no attribute {name}")
