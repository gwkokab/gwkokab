# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""All necessary constants for the package."""

### define units in SI
C_SI = 299792458.0  # m/s
r"""Speed of light in vacuum in :math:`\text{m}/\text{s}`"""
PC_SI = 3.085677581491367e16
"""Parsec in meters."""
MPC_SI = PC_SI * 1e6
"""Mega parsec in meters."""
G_SI = 6.6743e-11
r"""Gravitational constant in :math:`\text{m}^{3}\text{kg}^{-1}\text{s}^{-2}`"""
MSUN_SI = 1.9884099021470415e30
"""Solar mass in kg."""


### define units in CGS
G_CGS = G_SI * 1e3
r"""Gravitational constant in :math:`\text{cm}^{3}\text{g}^{-1}\text{s}^{-2}`"""
C_CGS = C_SI * 1e2
r"""Speed of light in vacuum in :math:`\text{cm}/\text{s}`"""
PC_CGS = PC_SI * 1e2
"""Parsec in centimeters."""
MPC_CGS = MPC_SI * 1e2
"""Mega parsec in centimeters."""
MSUN_CGS = MSUN_SI * 1e3
"""Solar mass in grams."""
