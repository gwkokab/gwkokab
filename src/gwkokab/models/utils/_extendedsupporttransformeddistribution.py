# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from numpyro.distributions import TransformedDistribution

from ..constraints import transform_constraint


class ExtendedSupportTransformedDistribution(TransformedDistribution):
    @property
    def support(self):
        return transform_constraint(self.base_dist.support, self.transforms)
