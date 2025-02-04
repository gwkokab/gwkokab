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
from typing import List, Tuple

import numpy as np
import numpyro.distributions as dist
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints

from gwkokab.models.utils import ScaledMixture


_mixture_dist = ScaledMixture(
    log_scales=np.log(np.array([2.0, 3.0])),
    component_distributions=[
        dist.Uniform(validate_args=True),
        dist.Beta(2.0, 1.0, validate_args=True),
    ],
    support=constraints.unit_interval,
)

_mixture_dist_batched_by_rates = ScaledMixture(
    log_scales=np.log(np.array([[2.0, 3.0], [4.0, 5.0]])),
    component_distributions=[
        dist.Uniform(validate_args=True),
        dist.Beta(2.0, 1.0, validate_args=True),
    ],
    support=constraints.unit_interval,
)


_mixture_dist_batched_by_dist = ScaledMixture(
    log_scales=np.log(np.array([2.0, 3.0])),
    component_distributions=[
        dist.Uniform(np.zeros((2,)), np.ones((2,)), validate_args=True),
        dist.Beta(2.0, 1.0, validate_args=True),
    ],
    support=constraints.unit_interval,
)


def _unif_moments(k: int, low: ArrayLike, high: ArrayLike) -> Array:
    return (np.power(high, k + 1) - np.power(low, k + 1)) / ((k + 1) * (high - low))


def _beta_moments(k: int, a: ArrayLike, b: ArrayLike) -> Array:
    return np.prod([(a + i) / (a + b + i) for i in range(k)])


DIST_LOG_VT_VALUE: List[
    Tuple[type[dist.Distribution], Callable[[Array], Array], ArrayLike]
] = [
    (
        _mixture_dist,
        lambda x: jnp.log(x),
        2.0 * _unif_moments(1, 0.0, 1.0) + 3.0 * _beta_moments(1, 2.0, 1.0),
    ),
    (
        _mixture_dist,
        lambda x: 2 * jnp.log(x),
        2.0 * _unif_moments(2, 0.0, 1.0) + 3.0 * _beta_moments(2, 2.0, 1.0),
    ),
    (
        _mixture_dist,
        lambda x: 3 * jnp.log(x),
        2.0 * _unif_moments(3, 0.0, 1.0) + 3.0 * _beta_moments(3, 2.0, 1.0),
    ),
    (
        _mixture_dist,
        lambda x: 4 * jnp.log(x),
        2.0 * _unif_moments(4, 0.0, 1.0) + 3.0 * _beta_moments(4, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_rates,
        lambda x: jnp.log(x),
        np.array([2.0, 4.0]) * _unif_moments(1, 0.0, 1.0)
        + np.array([3.0, 5.0]) * _beta_moments(1, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_rates,
        lambda x: 2 * jnp.log(x),
        np.array([2.0, 4.0]) * _unif_moments(2, 0.0, 1.0)
        + np.array([3.0, 5.0]) * _beta_moments(2, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_rates,
        lambda x: 3 * jnp.log(x),
        np.array([2.0, 4.0]) * _unif_moments(3, 0.0, 1.0)
        + np.array([3.0, 5.0]) * _beta_moments(3, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_rates,
        lambda x: 4 * jnp.log(x),
        np.array([2.0, 4.0]) * _unif_moments(4, 0.0, 1.0)
        + np.array([3.0, 5.0]) * _beta_moments(4, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_dist,
        lambda x: jnp.log(x),
        2.0 * _unif_moments(1, np.zeros((2,)), np.ones((2,)))
        + 3.0 * _beta_moments(1, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_dist,
        lambda x: 2 * jnp.log(x),
        2.0 * _unif_moments(2, np.zeros((2,)), np.ones((2,)))
        + 3.0 * _beta_moments(2, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_dist,
        lambda x: 3 * jnp.log(x),
        2.0 * _unif_moments(3, np.zeros((2,)), np.ones((2,)))
        + 3.0 * _beta_moments(3, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_dist,
        lambda x: 4 * jnp.log(x),
        2.0 * _unif_moments(4, np.zeros((2,)), np.ones((2,)))
        + 3.0 * _beta_moments(4, 2.0, 1.0),
    ),
]
