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


from typing import Tuple

import chex
from absl.testing import parameterized
from jax import random as jrd
from jaxtyping import Array
from numpy.testing import assert_allclose
from numpyro import distributions as dist
from numpyro.distributions.distribution import DistributionLike

from gwkokab.models import (
    PowerlawPrimaryMassRatio,
)
from gwkokab.models.utils import JointDistribution


uniform = dist.Uniform(low=0.0, high=1.0, validate_args=True)
normal = dist.Normal(loc=0.0, scale=1.0, validate_args=True)
exponential = dist.Exponential(rate=1.0, validate_args=True)
powerlawprimarymassratio = PowerlawPrimaryMassRatio(
    alpha=-1.0, beta=1.0, mmin=0.000001, mmax=10.0, validate_args=True
)

_DISTRIBUTIONS = [
    # (uniform,),
    # (normal,),
    # (exponential,),
    (uniform, normal),
    (uniform, exponential),
    # (normal, exponential),
    # (uniform, normal, exponential),
    # (uniform.expand((3,)),),
    # (normal.expand((3,)),),
    # (exponential.expand((3,)),),
    # (uniform.expand((3,)), normal.expand((3,))),
    # (uniform.expand((3,)), exponential.expand((3,))),
    # (normal.expand((3,)), exponential.expand((3,))),
    # (uniform.expand((3,)), normal.expand((3,)), exponential.expand((3,))),
    # (powerlawprimarymassratio, uniform),
    # (powerlawprimarymassratio, normal),
]


class TestJointDistribution(parameterized.TestCase):
    @chex.variants(  # pyright: ignore
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.product(
        distributions=_DISTRIBUTIONS,
        shape=[
            # (),
            # (2,),
            (3, 7),
            (2, 4),
        ],
    )
    def test_log_prob(
        self, distributions: Tuple[DistributionLike, ...], shape: Tuple[int, ...]
    ) -> None:
        joint_dist = JointDistribution(*distributions, validate_args=True)

        @self.variant
        def _log_prob(value: Array) -> Array:
            return joint_dist.log_prob(value)

        values = jrd.uniform(
            jrd.PRNGKey(0),
            shape=shape + joint_dist.batch_shape + joint_dist.event_shape,
        )

        _log_prob_val = 0.0
        k = 0
        for _dist in distributions:
            if len(_dist.event_shape) == 0:
                shape = k
                k += 1
            else:
                shape = [i + k for i in _dist.event_shape]
                k += _dist.event_shape[0]
            _log_prob_val += _dist.log_prob(values[..., shape])

        assert_allclose(_log_prob(values), _log_prob_val, rtol=1e-6, atol=1e-6)
