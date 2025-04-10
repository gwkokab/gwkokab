# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import importlib.metadata
import os

from loguru import logger


__version__ = importlib.metadata.version("gwkokab")


from . import (
    constants as constants,
    cosmology as cosmology,
    errors as errors,
    inference as inference,
    models as models,
    parameters as parameters,
    poisson_mean as poisson_mean,
    population as population,
    utils as utils,
    vts as vts,
)
from .utils.logger import set_log_level as _set_log_level


_set_log_level(os.environ.get("GWKOKAB_LOG_LEVEL", "TRACE"))
del _set_log_level


logger.info("Initializing GWKokab version {version}", version=__version__)
