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


import jax
from jax import numpy as jnp
from numpyro import distributions as dist

from gwkokab.utils import get_key


class JointDistribution(dist.Distribution):
    r"""Joint distribution of multiple marginal distributions."""

    def __init__(self, *marginal_distributions: dist.Distribution, validate_args=None) -> None:
        self.marginal_distributions = marginal_distributions
        self.shaped_values = tuple()
        k = 0
        for d in self.marginal_distributions:
            if d.event_shape:
                self.shaped_values += (slice(k, k + d.event_shape[0]),)
                k += d.event_shape[0]
            else:
                self.shaped_values += (k,)
                k += 1
        super(JointDistribution, self).__init__(
            batch_shape=(),
            event_shape=(k,),
            validate_args=validate_args,
        )

    # @partial(jit, static_argnums=(0,))
    def log_prob(self, value):
        log_probs = jax.tree_util.tree_map(
            lambda d, v: d.log_prob(value[..., v]),
            self.marginal_distributions,
            self.shaped_values,
            is_leaf=lambda x: isinstance(x, dist.Distribution),
        )
        log_probs = jnp.sum(jnp.asarray(log_probs))
        return log_probs

    # @partial(jit, static_argnums=(0, 2))
    def sample(self, key, sample_shape=()):
        if key is None or isinstance(key, int):
            key = get_key(key)
        keys = tuple(jax.random.split(key, len(self.marginal_distributions)))
        samples = jax.tree_util.tree_map(
            lambda d, k: d.sample(k, sample_shape).reshape(*sample_shape, -1),
            self.marginal_distributions,
            keys,
            is_leaf=lambda x: isinstance(x, dist.Distribution),
        )
        samples = jnp.concatenate(samples, axis=-1)
        return samples
