# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Distribution machinery shared by the population models.

- :class:`~gwkokab.models.utils.DoublyTruncatedPowerLaw` and its functional core --
  a power law on :math:`[x_\min, x_\max]` with custom JVP rules that stay finite at
  the :math:`\alpha = -1` special case.
- :class:`~gwkokab.models.utils.JointDistribution` -- the product of independent
  marginals, used to assemble one mixture component out of per-parameter models.
- :class:`~gwkokab.models.utils.LazyJointDistribution` -- the same, but where a
  marginal's own hyper-parameters are themselves sampled.
- :class:`~gwkokab.models.utils.ScaledMixture` -- a mixture whose components carry
  log rates, which is what lets rates and shapes be inferred jointly.
- :class:`~gwkokab.models.utils.ExtendedSupportTransformedDistribution` -- a
  transformed distribution whose support is not narrowed by the transform.
"""

from ._doubletruncpowerlaw import (
    doubly_truncated_power_law_cdf as doubly_truncated_power_law_cdf,
    doubly_truncated_power_law_icdf as doubly_truncated_power_law_icdf,
    doubly_truncated_power_law_log_norm_constant as doubly_truncated_power_law_log_norm_constant,
    doubly_truncated_power_law_log_prob as doubly_truncated_power_law_log_prob,
    DoublyTruncatedPowerLaw as DoublyTruncatedPowerLaw,
)
from ._extendedsupporttransformeddistribution import (
    ExtendedSupportTransformedDistribution as ExtendedSupportTransformedDistribution,
)
from ._joindistribution import JointDistribution as JointDistribution
from ._lazyjointdistribution import LazyJointDistribution as LazyJointDistribution
from ._scaledmixture import ScaledMixture as ScaledMixture
