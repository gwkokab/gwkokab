# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the analysis drivers.

- :mod:`~gwkokab.analysis.utils.checks` -- validity predicates on hyper-parameters,
  used as ``where_fns`` by the likelihoods.
- :mod:`~gwkokab.analysis.utils.common` -- JSON I/O and the ``<name>_<index>``
  hyper-parameter name expansion.
- :mod:`~gwkokab.analysis.utils.jenks` -- bucketing events of unequal sample count into
  rectangular batches.
- :mod:`~gwkokab.analysis.utils.literals` -- the names of the datasets and groups written
  to the output HDF5.
- :mod:`~gwkokab.analysis.utils.logger` -- loguru configuration, driven by the
  ``GWKOKAB_LOG_*`` environment variables.
- :mod:`~gwkokab.analysis.utils.marginals` -- per-component marginal densities for the
  post-hoc report.
- :mod:`~gwkokab.analysis.utils.priors` -- turning ``prior_cfg.json`` into priors,
  constants, aliases and lazy priors.
- :mod:`~gwkokab.analysis.utils.regex` -- matching the regex keys of ``prior_cfg.json``
  against hyper-parameter names.
"""

from . import (
    checks as checks,
    common as common,
    jenks as jenks,
    literals as literals,
    logger as logger,
    marginals as marginals,
    priors as priors,
    regex as regex,
)
