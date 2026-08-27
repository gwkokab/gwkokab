# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""GWKokab: a JAX-based gravitational-wave population inference toolkit.

The top-level namespace re-exports the building blocks needed to assemble a
hierarchical population analysis:

- :mod:`~gwkokab.constants` -- physical and numerical constants.
- :mod:`~gwkokab.cosmology` -- redshift/distance conversions and volume elements.
- :mod:`~gwkokab.errors` -- measurement-error models used to synthesise mock
  parameter estimation (PE) samples.
- :mod:`~gwkokab.inference` -- inhomogeneous-Poisson log-likelihoods for the
  supported sampler backends and data representations.
- :mod:`~gwkokab.models` -- NumPyro population models, constraints and transforms.
- :mod:`~gwkokab.parameters` -- the :class:`~gwkokab.parameters.Parameters` enum and
  the relation graph used to derive one GW parameter from others.
- :mod:`~gwkokab.poisson_mean` -- selection-function (Poisson mean) estimators.
- :mod:`~gwkokab.utils` -- coordinate transformations, math helpers and logging-aware
  exceptions.

The command-line drivers live in :mod:`gwkokab.analysis`; every user-facing entry
point is declared as a console script in :file:`pyproject.toml`.
"""

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
from .version import __version__ as __version__
