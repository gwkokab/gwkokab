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
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key

from gwkokab.models.constraints import all_constraint


class JointDistribution(Distribution):
    r"""Joint distribution of multiple marginal distributions."""

    pytree_data_fields = ("marginal_distributions", "shaped_values", "_support")

    def __init__(
        self, *marginal_distributions: Distribution, validate_args=None
    ) -> None:
        r"""
        :param marginal_distributions: A sequence of marginal distributions.
        """
        if not marginal_distributions:
            raise ValueError("At least one marginal distribution is required.")
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
        self._support = all_constraint(
            *[m_d.support for m_d in marginal_distributions],
            event_slices=self.shaped_values,
        )

    @constraints.dependent_property(is_discrete=False)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        def log_prob_i(d: Distribution, v: Array) -> Array:
            log_p = d.log_prob(v)
            return log_p

        log_probs = jtr.reduce(
            lambda x, y: x + log_prob_i(y[0], value[..., y[1]]),
            list(zip(self.marginal_distributions, self.shaped_values)),
            jnp.zeros(self.batch_shape),
            is_leaf=lambda x: isinstance(x, tuple),
        )
        return log_probs

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()):
        assert is_prng_key(key)
        keys = tuple(jrd.split(key, len(self.marginal_distributions)))
        samples = [
            d.sample(k, sample_shape).reshape(*sample_shape, -1)
            for d, k in zip(self.marginal_distributions, keys)
        ]
        samples = jnp.concatenate(samples, axis=-1)
        return samples
