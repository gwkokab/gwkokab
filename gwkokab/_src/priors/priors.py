#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array
from numpyro import distributions as dist
from numpyro.distributions.util import validate_sample


class UnnormalizedUniformOnRealLine(dist.Distribution):
    """This is an unnormalize uniform distribution on the real line,
    i.e. the density is 1 for all real numbers. It does not depend on any
    parameters.

    > [NOTE!]
    > You can not sample from it.
    """

    arg_constraints = {}
    support = dist.constraints.real

    def __init__(self, *, validate_args=None):
        super(UnnormalizedUniformOnRealLine, self).__init__(
            batch_shape=(), event_shape=(), validate_args=validate_args
        )

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        return jnp.ones(value.shape[:-1])
