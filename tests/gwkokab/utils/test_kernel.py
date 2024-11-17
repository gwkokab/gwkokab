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

from __future__ import annotations

import chex
from absl.testing import parameterized
from jax import grad, numpy as jnp

from gwkokab.utils import log_planck_taper_window


x = [
    -0.03196073,
    -0.09624577,
    -0.09714341,
    -0.10647035,
    -0.19786143,
    -0.579038,
    -0.89962137,
    -0.9902357,
    0.0,
    0.2309705,
    0.33585954,
    0.54150164,
    0.7553431,
    0.8452761,
    1.0,
    1.0090289,
    1.1652794,
    1.3518975,
    1.6708747,
    1.7008203,
    1.720163,
    1.9225303,
]


class TestPlanckTaperWindow(parameterized.TestCase):
    @chex.variants(  # pyright: ignore
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters([(str(i), x_) for i, x_ in enumerate(x)])
    def test_planck_taper_window(self, x):
        @self.variant  # pyright: ignore
        def planck_taper_window_fn(x):
            return jnp.exp(log_planck_taper_window(x))

        if x < 0.0:
            assert planck_taper_window_fn(x) == 0
        elif x > 1.0:
            assert planck_taper_window_fn(x) == 1
        else:
            assert 0 <= planck_taper_window_fn(x) <= 1

    @chex.variants(  # pyright: ignore
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters([(str(i), x_) for i, x_ in enumerate(x)])
    def test_planck_taper_window_grad(self, x):
        @self.variant  # pyright: ignore
        def planck_taper_window_grad_fn(x):
            return grad(lambda X: jnp.exp(log_planck_taper_window(X)))(x)

        grad_val = planck_taper_window_grad_fn(x)
        assert not jnp.any(jnp.isnan(grad_val))
