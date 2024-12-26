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
"""Cosmology module for GWKokab, adapted from code written by
`Reed Essick <https://orcid.org/0000-0001-8196-9267>`_ included in the
`gw-distributions <https://git.ligo.org/reed.essick/gw-distributions>`_ package at
`source <https://git.ligo.org/reed.essick/gw-distributions/-/blob/master/gwdistributions/utils/cosmology.py>`_."""

from ._cosmology import Cosmology as Cosmology
from ._planck import (
    PLANCK_2015_Cosmology as PLANCK_2015_Cosmology,
    PLANCK_2018_Cosmology as PLANCK_2018_Cosmology,
)
