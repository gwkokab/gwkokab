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

import chex
import numpy as np
import numpyro.distributions as dist
import pytest
from absl.testing import parameterized
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, ArrayLike
from numpy.testing import assert_allclose
from numpyro.distributions import constraints

from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.poisson_mean import PoissonMean


_a = np.array

_scaled_dist = ScaledMixture(
    log_scales=np.log(_a([2.0])),
    component_distributions=[dist.Beta(_a([2.0]), _a([1.0]), validate_args=True)],
    support=constraints.unit_interval,
    validate_args=True,
)

_mixture_dist = ScaledMixture(
    log_scales=np.log(_a([2.0])),
    component_distributions=[
        JointDistribution(
            dist.Uniform(_a([0.0]), _a([1.0]), validate_args=True),
            dist.Beta(_a([2.0]), _a([1.0]), validate_args=True),
            validate_args=True,
        ),
    ],
    support=constraints.independent(
        constraints.interval(jnp.zeros((2,)), jnp.ones((2,))), 1
    ),
    validate_args=True,
)


_mixture_dist_batched_by_dist = ScaledMixture(
    log_scales=np.log(_a([1.0, 2.0, 3.0])),
    component_distributions=[
        JointDistribution(
            dist.Uniform(_a([0.0]), _a([1.0]), validate_args=True),
            dist.Uniform(_a([0.0]), _a([1.0]), validate_args=True),
            validate_args=True,
        ),
        JointDistribution(
            dist.Uniform(_a([0.0]), _a([1.0]), validate_args=True),
            dist.Uniform(_a([0.0]), _a([1.0]), validate_args=True),
            validate_args=True,
        ),
        JointDistribution(
            dist.Beta(_a([2.0]), _a([1.0]), validate_args=True),
            dist.Beta(_a([2.0]), _a([1.0]), validate_args=True),
            validate_args=True,
        ),
    ],
    support=constraints.independent(
        constraints.interval(jnp.zeros((2,)), jnp.ones((2,))), 1
    ),
    validate_args=True,
)


def _unif_moments(k: int, low: ArrayLike, high: ArrayLike) -> Array:
    return (np.power(high, k + 1) - np.power(low, k + 1)) / ((k + 1) * (high - low))


def _beta_moments(k: int, a: ArrayLike, b: ArrayLike) -> Array:
    return np.prod([float(a + i) / float(a + b + i) for i in range(k)])


DIST_LOG_VT_VALUE: List[
    Tuple[type[dist.Distribution], Callable[[Array], Array], ArrayLike]
] = [
    (
        _scaled_dist,
        lambda x: jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        2.0 * _beta_moments(1, 2.0, 1.0),
    ),
    (
        _scaled_dist,
        lambda x: 2.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        2.0 * _beta_moments(2, 2.0, 1.0),
    ),
    (
        _scaled_dist,
        lambda x: 3.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        2.0 * _beta_moments(3, 2.0, 1.0),
    ),
    (
        _scaled_dist,
        lambda x: 4.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        2.0 * _beta_moments(4, 2.0, 1.0),
    ),
    (
        _mixture_dist,
        lambda x: jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        2.0 * _unif_moments(1, 0.0, 1.0) * _beta_moments(1, 2.0, 1.0),
    ),
    (
        _mixture_dist,
        lambda x: 2.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        2.0 * _unif_moments(2, 0.0, 1.0) * _beta_moments(2, 2.0, 1.0),
    ),
    (
        _mixture_dist,
        lambda x: 3.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        2.0 * _unif_moments(3, 0.0, 1.0) * _beta_moments(3, 2.0, 1.0),
    ),
    (
        _mixture_dist,
        lambda x: 4.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        2.0 * _unif_moments(4, 0.0, 1.0) * _beta_moments(4, 2.0, 1.0),
    ),
    (
        _mixture_dist_batched_by_dist,
        lambda x: jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        3.0 * np.square(_unif_moments(1, 0.0, 1.0))
        + 3.0 * np.square(_beta_moments(1, 2.0, 1.0)),
    ),
    (
        _mixture_dist_batched_by_dist,
        lambda x: 2.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        3.0 * np.square(_unif_moments(2, 0.0, 1.0))
        + 3.0 * np.square(_beta_moments(2, 2.0, 1.0)),
    ),
    (
        _mixture_dist_batched_by_dist,
        lambda x: 3.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        3.0 * np.square(_unif_moments(3, 0.0, 1.0))
        + 3.0 * np.square(_beta_moments(3, 2.0, 1.0)),
    ),
    (
        _mixture_dist_batched_by_dist,
        lambda x: 4.0 * jnp.sum(jnp.log(x), axis=-1, dtype=jnp.float64),
        3.0 * np.square(_unif_moments(4, 0.0, 1.0))
        + 3.0 * np.square(_beta_moments(4, 2.0, 1.0)),
    ),
]


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
        pytest.xfail("Need to update tests in accordace to recent changes")
        key = jrd.PRNGKey(0)

        pmean_estimator = PoissonMean(
            logVT_fn=log_vt_fn,
            proposal_dists=["self" for _ in range(dist.mixture_size)],
            key=key,
            self_num_samples=50_000,
            scale=1.0,
        )

        @self.variant  # pyright: ignore
        def pmean_estimator_fn(dist_arg):
            return pmean_estimator(dist_arg)

        assert_allclose(pmean_estimator_fn(dist), value, atol=5e-3, rtol=1e-6)

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
    def test_importance_sampling_poisson_mean(
        self,
        dist: ScaledMixture,
        log_vt_fn: Callable[[Array], Array],
        value: ArrayLike,
    ) -> None:
        pytest.xfail("Need to update tests in accordace to recent changes")
        key = jrd.PRNGKey(0)

        pmean_estimator = PoissonMean(
            logVT_fn=log_vt_fn,
            proposal_dists=[
                dist.component_distributions[i] for i in range(dist.mixture_size)
            ],
            key=key,
            num_samples_per_component=[50_000 for _ in range(dist.mixture_size)],
            scale=1.0,
        )

        @self.variant  # pyright: ignore
        def pmean_estimator_fn(dist_arg):
            return pmean_estimator(dist_arg)

        assert_allclose(pmean_estimator_fn(dist), value, atol=5e-3, rtol=1e-6)

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
    def test_inverse_transform_sampling_poisson_and_importance_sampling_poisson_mean(
        self,
        dist: ScaledMixture,
        log_vt_fn: Callable[[Array], Array],
        value: ArrayLike,
    ) -> None:
        pytest.xfail("Need to update tests in accordace to recent changes")
        key = jrd.PRNGKey(0)

        pmean_estimator = PoissonMean(
            logVT_fn=log_vt_fn,
            proposal_dists=[
                dist.component_distributions[i] if i % 2 else "self"
                for i in range(dist.mixture_size)
            ],
            key=key,
            num_samples=50_000,
            self_num_samples=10_000,
            scale=1.0,
        )

        @self.variant  # pyright: ignore
        def pmean_estimator_fn(dist_arg):
            return pmean_estimator(dist_arg)

        assert_allclose(pmean_estimator_fn(dist), value, atol=1e-2, rtol=1e-6)
