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

from typing_extensions import Tuple

from jax import lax, numpy as jnp, random as jrd, tree as jtr
from jaxtyping import PRNGKeyArray
from numpyro.distributions import constraints, Distribution
from numpyro.util import is_prng_key


class JointDistribution(Distribution):
    r"""Joint distribution of multiple marginal distributions."""

    pytree_aux_fields = ("marginal_distributions", "shaped_values")
    support = constraints.real_vector

    def __init__(
        self, *marginal_distributions: Distribution, validate_args=None
    ) -> None:
        r"""
        :param marginal_distributions: A sequence of marginal distributions.
        """
        self.marginal_distributions = marginal_distributions
        self.shaped_values: Tuple[int | slice, ...] = tuple()
        batch_shape = lax.broadcast_shapes(
            *tuple(d.batch_shape for d in self.marginal_distributions)
        )
        k = 0
        for d in self.marginal_distributions:
            if d.event_shape:
                self.shaped_values += (slice(k, k + d.event_shape[0]),)
                k += d.event_shape[0]
            else:
                self.shaped_values += (k,)
                k += 1
        super(JointDistribution, self).__init__(
            batch_shape=batch_shape,
            event_shape=(k,),
            validate_args=validate_args,
        )

    def log_prob(self, value):
        log_probs = jtr.map(
            lambda d, v: d.log_prob(value[..., v]),
            self.marginal_distributions,
            self.shaped_values,
            is_leaf=lambda x: isinstance(x, Distribution),
        )
        log_probs = jtr.reduce(
            lambda x, y: x + y,
            log_probs,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )
        return log_probs

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()):
        assert is_prng_key(key)
        keys = tuple(jrd.split(key, len(self.marginal_distributions)))
        samples = jtr.map(
            lambda d, k: d.sample(k, sample_shape).reshape(*sample_shape, -1),
            self.marginal_distributions,
            keys,
            is_leaf=lambda x: isinstance(x, Distribution),
        )
        samples = jnp.concatenate(samples, axis=-1)
        return samples