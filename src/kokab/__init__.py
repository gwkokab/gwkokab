# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""Kokab namespace provides a comprehensive suite of models and utilities for population
analysis of compact binary coalescences (CBCs).

It includes command-line interfaces (CLIs) for generating mock posterior estimates,
conducting population analyses, and producing visualizations. A detailed list of
available models is accessible at :doc:`/autoapi/gwkokab/models/index`.
"""

import os

from . import (
    ecc_matters as ecc_matters,
    n_pls_m_gs as n_pls_m_gs,
    one_powerlaw_one_peak as one_powerlaw_one_peak,
    utils as utils,
)
from .utils.logger import log_info as _log_info, set_log_level as _set_log_level


_set_log_level(os.environ.get("GWKOKAB_LOG_LEVEL", "TRACE"))
del _set_log_level

_log_info(start=True)
del _log_info
