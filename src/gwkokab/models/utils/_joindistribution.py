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


from typing import Optional

from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key

from gwkokab.models.constraints import all_constraint


class JointDistribution(Distribution):
    r"""Joint distribution of multiple marginal distributions."""

    pytree_aux_fields = ("marginal_distributions", "shaped_values")
    pytree_data_fields = ("_support",)

    def __init__(
        self,
        *marginal_distributions: Distribution,
        validate_args: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        validate_args : Optional[bool], optional
            Whether to validate input arguments, by default None

        Raises
        ------
        ValueError
            If no marginal distributions are provided.
        """
        if not marginal_distributions:
            raise ValueError("At least one marginal distribution is required.")
        assert all(isinstance(d, Distribution) for d in marginal_distributions), (
            "All marginal distributions must be instances of Distribution."
        )
        batch_shape = marginal_distributions[0].batch_shape
        assert all(d.batch_shape == batch_shape for d in marginal_distributions), (
            "All marginal distributions must have the same batch shape."
        )
        event_shape_dim = max(map(lambda d: d.event_dim, marginal_distributions))
        event_shape_per_dist = tuple(
            (event_shape_dim - d.event_dim) * (1,) + d.event_shape
            for d in marginal_distributions
        )
        event_shape = tuple(map(sum, tuple(zip(*event_shape_per_dist))))
        self.shaped_values = [[slice(i) for i in event_shape_per_dist[0]]]
        for i in range(1, len(marginal_distributions)):
            prev = self.shaped_values[i - 1]
            self.shaped_values.append(
                [
                    slice(prev[j].stop, prev[j].stop + k)
                    for j, k in enumerate(event_shape_per_dist[i])
                ]
            )

        print(event_shape_per_dist, self.shaped_values)

        self.marginal_distributions = marginal_distributions
        super(JointDistribution, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )
        self._support = all_constraint(
            constraints=[m_d.support for m_d in marginal_distributions],
            event_slices=self.shaped_values,
        )

    @constraints.dependent_property(is_discrete=False)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        print(self.batch_shape, self.event_shape, self.shaped_values)
        log_prob_stack = [
            m_dist.log_prob(value[..., *slice_])
            for m_dist, slice_ in zip(self.marginal_distributions, self.shaped_values)
        ]
        print([s.shape for s in log_prob_stack])
        return jnp.sum(jnp.stack(log_prob_stack, axis=-1), axis=-1)

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()):
        assert is_prng_key(key)
        keys = tuple(jrd.split(key, len(self.marginal_distributions)))
        samples = [
            d.sample(k, sample_shape).reshape(*sample_shape, -1)
            for d, k in zip(self.marginal_distributions, keys)
        ]
        samples = jnp.concatenate(samples, axis=-1)
        return samples
