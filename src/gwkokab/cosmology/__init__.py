# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0
"""Cosmology module for GWKokab, adapted from code written by
`Reed Essick <https://orcid.org/0000-0001-8196-9267>`_ included in the
`gw-distributions <https://git.ligo.org/reed.essick/gw-distributions>`_ package at
`source <https://git.ligo.org/reed.essick/gw-distributions/-/blob/master/gwdistributions/utils/cosmology.py>`_."""

from ._cosmology import Cosmology as Cosmology
from ._planck import (
    PLANCK_2015_Cosmology as PLANCK_2015_Cosmology,
    PLANCK_2018_Cosmology as PLANCK_2018_Cosmology,
)
