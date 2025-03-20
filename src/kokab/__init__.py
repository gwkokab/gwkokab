# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""Kokab namespace provides a comprehensive suite of models and utilities for population
analysis of compact binary coalescences (CBCs).

It includes command-line interfaces (CLIs) for generating mock posterior estimates,
conducting population analyses, and producing visualizations. A detailed list of
available models is accessible at :doc:`/autoapi/gwkokab/models/index`.
"""

from . import (
    chi_eff_q_relation as chi_eff_q_relation,
    ecc_matters as ecc_matters,
    n_pls_m_gs as n_pls_m_gs,
    n_spls_m_sgs as n_spls_m_sgs,
    one_powerlaw_one_peak as one_powerlaw_one_peak,
    utils as utils,
)
