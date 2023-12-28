# Copyright 2023 The Jaxtro Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import jax
from jax import numpy as jnp

sys.path.append("../jaxtro")
from jaxtro.models import Wysocki2019SpinModel


def test_init() -> None:
    model = Wysocki2019SpinModel(
        alpha_1=1.1,
        beta_1=5.5,
        alpha_2=2.1,
        beta_2=2.5,
        chimax=1.0,
        name="test",
    )
    assert model._name == "test"
    assert jnp.all(model._alpha - jnp.array([1.1, 2.1])) < 1e-6
    assert jnp.all(model._beta - jnp.array([5.5, 2.5])) < 1e-6
    assert jnp.all(model._chimax == 1.0)


def test_rvs() -> None:
    model = Wysocki2019SpinModel(
        alpha_1=1.1,
        beta_1=5.5,
        alpha_2=2.1,
        beta_2=2.5,
        chimax=1.0,
        name="test",
    )
    key = jax.random.PRNGKey(0)
    N = 100000
    rvs = model.rvs(N=N, key=key)
    assert rvs.shape == (N, 2)
