# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Poisson mean (selection function) estimators.

The inhomogeneous-Poisson likelihood needs the expected number of detections
:math:`\mu(\Lambda)` for a population :math:`\Lambda` -- the integral of the population
density against the detection probability. Each function here returns a callable that
computes :math:`\log\mu` from a :class:`~gwkokab.models.utils.ScaledMixture`, differing
only in how the sensitivity enters:

- :func:`poisson_mean_from_neural_vt` -- a neural regressor for the sensitive
  volume-time :math:`VT`.
- :func:`poisson_mean_from_neural_pdet` -- a neural regressor for the detection
  probability :math:`p_{\text{det}}`.
- :func:`poisson_mean_from_sensitivity_injections` -- Monte Carlo over a found-injection
  campaign, with FAR and SNR cuts.

Estimators are selected from ``pmean_cfg.json`` through
:mod:`gwkokab.analysis.core.inference_io`, which also supports a ``custom`` loader for
an estimator supplied by file path.
"""

from ._injection_based import (
    poisson_mean_from_sensitivity_injections as poisson_mean_from_sensitivity_injections,
)
from ._neural_pdet import poisson_mean_from_neural_pdet as poisson_mean_from_neural_pdet
from ._neural_vt import poisson_mean_from_neural_vt as poisson_mean_from_neural_vt
