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

import sys

from jax import numpy as jnp

sys.path.append("../gwkokab")
from gwkokab.models import Wysocki2019SpinModel


class TestWysocki2019SpinModel:
    model = Wysocki2019SpinModel(
        alpha=1.1,
        beta=5.5,
        chimax=1.0,
        name="test",
    )

    def test_init(self):
        assert self.model._name == "test"
        assert jnp.all(self.model._alpha - jnp.array([1.1, 2.1])) < 1e-6
        assert jnp.all(self.model._beta - jnp.array([5.5, 2.5])) < 1e-6

    def test_rvs(self):
        N = 1000
        rvs = self.model.samples(N)
        assert rvs.shape == (N,)
