# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

#
"""GWKAnal namespace provides a comprehensive suite of models and utilities for
population analysis of compact binary coalescences (CBCs).

It includes command-line interfaces (CLIs) for generating mock posterior estimates,
conducting population analyses, and producing visualizations. A detailed list of
available models is accessible at :doc:`/autoapi/gwkanal/models/index`.
"""

from . import (
    core as core,
    ecc_matters as ecc_matters,
    multisource as multisource,
    n_pls_m_gs as n_pls_m_gs,
    subpopulation as subpopulation,
    utils as utils,
)
