# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Two-dimensional mass models for compact binary populations.

Each model is a joint distribution over a pair of mass coordinates -- :math:`(m_1, m_2)`
or :math:`(m_1, q)` -- with the ordering :math:`m_2 \leq m_1` and the mass bounds built
into its support. The ``Smoothed*`` variants taper the density above the minimum mass
with a Planck window rather than truncating it sharply, and are normalised numerically
at construction time.
"""

from ._bpls import (
    BrokenPowerlaw as BrokenPowerlaw,
    BrokenPowerlawTwoPeak as BrokenPowerlawTwoPeak,
    SmoothedBrokenPowerlawMassRatioPowerlaw as SmoothedBrokenPowerlawMassRatioPowerlaw,
)
from ._models import (
    GaussianPrimaryMassRatio as GaussianPrimaryMassRatio,
    GenericSmoothedPowerlawMassRatio as GenericSmoothedPowerlawMassRatio,
    PowerlawPrimaryMassRatio as PowerlawPrimaryMassRatio,
    SmoothedGaussianPrimaryMassRatio as SmoothedGaussianPrimaryMassRatio,
    SmoothedPowerlawPrimaryMassRatio as SmoothedPowerlawPrimaryMassRatio,
    SmoothedTwoComponentPrimaryMassRatio as SmoothedTwoComponentPrimaryMassRatio,
    Wysocki2019MassModel as Wysocki2019MassModel,
)
