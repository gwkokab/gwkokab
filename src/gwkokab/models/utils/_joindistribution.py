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


from typing import Optional, Tuple

from jax import lax, numpy as jnp, random as jrd, tree as jtr
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key


class JointDistribution(Distribution):
    r"""Joint distribution of multiple marginal distributions."""

    pytree_aux_fields = ("marginal_distributions", "shaped_values")
    # pytree_data_fields = ("_support",)
    support = constraints.real_vector

    def __init__(
        self,
        *marginal_distributions: Distribution,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        validate_args : _type_, optional
            Whether to validate input arguments, by default None

        Raises
        ------
        ValueError
            If no marginal distributions are provided.
        """
        if not marginal_distributions:
            raise ValueError("At least one marginal distribution is required.")
        self.marginal_distributions = list(marginal_distributions)
        self.shaped_values: Tuple[Tuple[int, Optional[int]], ...] = tuple()
        batch_shape = lax.broadcast_shapes(
            *tuple(d.batch_shape for d in self.marginal_distributions)
        )
        k = 0
        for d in self.marginal_distributions:
            if d.event_shape:
                self.shaped_values += ((k, d.event_shape[0]),)
                k += d.event_shape[0]
            else:
                self.shaped_values += ((k, None),)
                k += 1
        super(JointDistribution, self).__init__(
            batch_shape=batch_shape,
            event_shape=(k,),
            validate_args=validate_args,
        )

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        def _f(m_dist: Distribution, _slice: Tuple[int, Optional[int]]) -> Array:
            if _slice[1] is None:
                _val = value[..., _slice[0]]
            else:
                _val = value[..., _slice[0] : _slice[0] + _slice[1]]
            m_dist_log_prob = m_dist.log_prob(_val)
            return m_dist_log_prob

        return jtr.reduce(
            lambda x, y: x + _f(*y),
            list(zip(self.marginal_distributions, self.shaped_values)),
            initializer=jnp.zeros(()),
            is_leaf=lambda x: isinstance(x, tuple),
        )

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()):
        assert is_prng_key(key)
        keys = tuple(jrd.split(key, len(self.marginal_distributions)))
        samples = [
            d.sample(k, sample_shape).reshape(*sample_shape, -1)
            for d, k in zip(self.marginal_distributions, keys)
        ]
        samples = jnp.concatenate(samples, axis=-1)
        return samples
