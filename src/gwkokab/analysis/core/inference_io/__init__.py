# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the JSON configuration files an analysis reads.

An inference run is configured by four JSON files, one per command-line flag:

===================== ======================================= ==========================================
Flag                  Model                                   Purpose
===================== ======================================= ==========================================
``--data-loader-cfg`` :class:`DiscretePELoader` /             which event files, how to undo the PE prior
                      :class:`AnalyticalGWalkPELoader`
``--prior-cfg``       plain JSON                              population priors
``--pmean-cfg``       :class:`PoissonMeanEstimationLoader`    selection function
``--sampler-cfg``     :class:`SamplerConfig`                  which sampler and all its knobs
===================== ======================================= ==========================================

Every model sets ``extra="forbid"``, so a typo in a configuration file raises a
``ValidationError`` rather than being silently ignored. The ``gwk_*_cfg_template``
console scripts dump starter versions of each file.
"""

from ._analytical_gwalk import AnalyticalGWalkPELoader as AnalyticalGWalkPELoader
from ._discrete import DiscretePELoader as DiscretePELoader
from ._poisson_mean import PoissonMeanEstimationLoader as PoissonMeanEstimationLoader
from ._sampler import (
    FlowMCGlobalConfig as FlowMCGlobalConfig,
    NumpyroGlobalConfig as NumpyroGlobalConfig,
    NumpyroMCMCConfig as NumpyroMCMCConfig,
    NumpyroNUTSSamplerConfig as NumpyroNUTSSamplerConfig,
    SamplerConfig as SamplerConfig,
)
