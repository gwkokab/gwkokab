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

from itertools import product

import chex
from absl.testing import parameterized
from jax import grad, numpy as jnp

from gwkokab.utils import log_planck_taper_window


x = [10.907955, 16.377821, 18.65535, 19.9329, 3.5127172, 9.406811]
a = [17.059471, 18.486938, 19.237825, 3.8927088, 8.82331]
b = [0.35940528, 1.4052713, 2.5084865, 4.8406796, 8.82331]


class TestPlanckTaperWindow(parameterized.TestCase):
    @chex.variants(  # pyright: ignore
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters(
        [(str(i), x_, a_, b_) for i, (x_, a_, b_) in enumerate(product(x, a, b))]
    )
    def test_planck_taper_window(self, x, a, b):
        @self.variant  # pyright: ignore
        def planck_taper_window_fn(x, a, b):
            return jnp.exp(log_planck_taper_window(x, a, b))

        if x < a:
            assert planck_taper_window_fn(x, a, b) == 0
        if x > a + b:
            assert planck_taper_window_fn(x, a, b) == 1
        if a <= x <= a + b:
            assert 0 <= planck_taper_window_fn(x, a, b) <= 1

    @chex.variants(  # pyright: ignore
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters(
        [(str(i), x_, a_, b_) for i, (x_, a_, b_) in enumerate(product(x, a, b))]
    )
    def test_planck_taper_window_grad(self, x, a, b):
        @self.variant  # pyright: ignore
        def planck_taper_window_fn(x, a, b):
            return jnp.exp(log_planck_taper_window(x, a, b))

        grad_fn = grad(planck_taper_window_fn)
        grad_val = grad_fn(x, a, b)
        assert not jnp.any(jnp.isnan(grad_val))
