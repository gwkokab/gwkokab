# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Redshift distributions for the merger rate.

Both models factorise the redshift density as the differential comoving volume element
times a rate evolution :math:`\psi(z)` and a :math:`1/(1+z)` time-dilation factor, and
differ only in :math:`\psi`: a pure power law for
:class:`~gwkokab.models.redshift.PowerlawRedshiftModel`, and the Madau-Dickinson star
formation rate -- which rises, peaks and falls -- for
:class:`~gwkokab.models.redshift.MadauDickinsonRedshiftModel`.
"""

from ._models import (
    MadauDickinsonRedshiftModel as MadauDickinsonRedshiftModel,
    PowerlawRedshiftModel as PowerlawRedshiftModel,
)
