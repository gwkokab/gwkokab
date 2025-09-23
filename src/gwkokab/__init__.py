# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import importlib.metadata
from typing import Final


__version__: Final[str] = importlib.metadata.version("gwkokab")


from . import (
    constants as constants,
    cosmology as cosmology,
    errors as errors,
    inference as inference,
    models as models,
    parameters as parameters,
    poisson_mean as poisson_mean,
    utils as utils,
)
