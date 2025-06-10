# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Optional, Tuple

from jax import lax, numpy as jnp, random as jrd
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
        self.shaped_values: Sequence[int | Tuple[int, int]] = tuple()
        batch_shape = lax.broadcast_shapes(
            *tuple(d.batch_shape for d in self.marginal_distributions)
        )
        k = 0
        for d in self.marginal_distributions:
            if d.event_shape:
                self.shaped_values += ((k, k + d.event_shape[0]),)
                k += d.event_shape[0]
            else:
                self.shaped_values += (k,)
                k += 1
        self._support = all_constraint(
            [d.support for d in self.marginal_distributions], self.shaped_values
        )
        super(JointDistribution, self).__init__(
            batch_shape=batch_shape,
            event_shape=(k,),
            validate_args=validate_args,
        )

    @constraints.dependent_property(is_discrete=False)
    def support(self) -> constraints.Constraint:
        """The support of the joint distribution."""
        return self._support

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        log_prob_val = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for m_dist, event_slice in zip(self.marginal_distributions, self.shaped_values):
            if isinstance(event_slice, int):
                value_slice = lax.dynamic_index_in_dim(
                    value, event_slice, axis=-1, keepdims=False
                )
            else:
                value_slice = lax.dynamic_slice_in_dim(
                    value,
                    event_slice[0],
                    event_slice[1] - event_slice[0],
                    axis=-1,
                )
            log_prob_val += m_dist.log_prob(value_slice)
        return log_prob_val

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()):
        assert is_prng_key(key)
        keys = tuple(jrd.split(key, len(self.marginal_distributions)))
        samples = [
            d.sample(k, sample_shape).reshape(*sample_shape, -1)
            for d, k in zip(self.marginal_distributions, keys)
        ]
        samples = jnp.concatenate(samples, axis=-1)
        return samples
