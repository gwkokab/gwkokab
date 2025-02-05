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
from numpyro.distributions import Uniform

from gwkokab.models.utils import ScaledMixture
from gwkokab.parameters import Parameter
from gwkokab.poisson_mean import ImportanceSamplingPoissonMean


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
        dist: ScaledMixture,
        log_vt_fn: Callable[[Array], Array],
        value: ArrayLike,
    ) -> None:
        prior = Uniform(0.0, 1.0, validate_args=True)
        parameter = Parameter("dummy", prior)

        key = jrd.PRNGKey(0)
        if dist.event_shape:
            parameters = [parameter for _ in range(dist.event_shape[0])]
        else:
            parameters = [parameter]

        pmean_estimator = ImportanceSamplingPoissonMean(
            logVT_fn=log_vt_fn,
            parameters=parameters,
            key=key,
            num_samples=10_000,
            scale=1.0,
        )

        @self.variant  # pyright: ignore
        def pmean_estimator_fn(dist_arg):
            return pmean_estimator(dist_arg)

        assert_allclose(pmean_estimator_fn(dist), value, atol=1e-2, rtol=1e-6)
