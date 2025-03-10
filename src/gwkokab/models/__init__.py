# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
"""
Provides implementation of various models, constraints, and transformations using
`NumPyro <https://github.com/pyro-ppl/numpyro>`_.
"""

from . import (
    constraints as constraints,
    transformations as transformations,
    utils as utils,
)
from ._models import (
    FlexibleMixtureModel as FlexibleMixtureModel,
    MassGapModel as MassGapModel,
    PowerlawPrimaryMassRatio as PowerlawPrimaryMassRatio,
    SmoothedGaussianPrimaryMassRatio as SmoothedGaussianPrimaryMassRatio,
    SmoothedPowerlawPrimaryMassRatio as SmoothedPowerlawPrimaryMassRatio,
    Wysocki2019MassModel as Wysocki2019MassModel,
)
from .multivariate import (
    ChiEffMassRatioConstraint as ChiEffMassRatioConstraint,
    ChiEffMassRatioCorrelated as ChiEffMassRatioCorrelated,
)
from .npowerlawmgaussian import NPowerlawMGaussian as NPowerlawMGaussian
from .nsmoothedpowerlawmsmoothedgaussian import (
    NSmoothedPowerlawMSmoothedGaussian as NSmoothedPowerlawMSmoothedGaussian,
    SmoothedPowerlawAndPeak as SmoothedPowerlawAndPeak,
    SmoothedPowerlawPeakAndPowerlawRedshift as SmoothedPowerlawPeakAndPowerlawRedshift,
)
from .redshift import PowerlawRedshift as PowerlawRedshift
from .spin import (
    BetaFromMeanVar as BetaFromMeanVar,
    GaussianSpinModel as GaussianSpinModel,
    IndependentSpinOrientationGaussianIsotropic as IndependentSpinOrientationGaussianIsotropic,
)
