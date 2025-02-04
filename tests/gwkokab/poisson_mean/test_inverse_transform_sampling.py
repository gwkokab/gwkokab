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


from collections.abc import Callable

import chex
from absl.testing import parameterized
from conftest import DIST_LOG_VT_VALUE
from jax import random as jrd
from jaxtyping import Array, ArrayLike
from numpy.testing import assert_allclose
from numpyro.distributions import Distribution

from gwkokab.poisson_mean import InverseTransformSamplingPoissonMean


class TestVariants(parameterized.TestCase):
    @chex.variants(  # pyright: ignore
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters(
        [
            (str(i), dist, log_vt_fn, value)
            for i, (dist, log_vt_fn, value) in enumerate(DIST_LOG_VT_VALUE)
        ]
    )
    def test_inverse_transform_sampling_poisson_mean(
        self,
        dist: type[Distribution],
        log_vt_fn: Callable[[Array], Array],
        value: ArrayLike,
    ):
        key = jrd.PRNGKey(0)

        pmean_estimator = InverseTransformSamplingPoissonMean(
            logVT_fn=log_vt_fn,
            key=key,
            num_samples=10_000,
            scale=1.0,
        )

        @self.variant  # pyright: ignore
        def pmean_estimator_fn(dist_arg):
            return pmean_estimator(dist_arg)

        assert_allclose(pmean_estimator_fn(dist), value, atol=1e-1, rtol=1e-6)
