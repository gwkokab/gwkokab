# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional

import jax
from jax import lax, numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import categorical, is_prng_key, validate_sample


class ScaledMixture(Distribution):
    r"""A finite mixture of component distributions from different families. This is a
    generalization of :class:`~numpyro.distributions.Mixture` where the component
    distributions are scaled by a set of rates.

    **Example**

    .. code::

       >>> import jax
       >>> import jax.random as jrd
       >>> import numpyro.distributions as dist
       >>> from gwkokab.models.utils import ScaledMixture
       >>> log_scales = jrd.uniform(jrd.PRNGKey(42), (3,), minval=0, maxval=5)
       >>> component_dists = [
       ...     dist.Normal(loc=0.0, scale=1.0),
       ...     dist.Normal(loc=-0.5, scale=0.3),
       ...     dist.Normal(loc=0.6, scale=1.2),
       ... ]
       >>> mixture = ScaledMixture(log_scales, component_dists)
       >>> mixture.sample(jax.random.PRNGKey(42)).shape
       ()
    """

    arg_constraints = {
        "log_scales": constraints.real_vector,
    }
    pytree_data_fields = ("_component_distributions", "_support", "log_scales")
    pytree_aux_fields = ("_mixture_size",)

    def __init__(
        self,
        log_scales: Array,
        component_distributions: List[Distribution],
        *,
        support: Optional[constraints.Constraint] = None,
        validate_args: Optional[bool] = None,
    ):
        self.log_scales = log_scales
        try:
            component_distributions = list(component_distributions)
        except TypeError:
            raise ValueError(
                "The 'component_distributions' argument must be a list of Distribution objects"
            )
        self._mixture_size = log_scales.shape[-1]
        for d in component_distributions:
            if not isinstance(d, Distribution):
                raise ValueError(
                    "All elements of 'component_distributions' must be instances of "
                    "numpyro.distributions.Distribution subclasses"
                )
        if len(component_distributions) != self.mixture_size:
            raise ValueError(
                "The number of elements in 'component_distributions' must match the mixture size; "
                f"expected {self._mixture_size}, got {len(component_distributions)}"
            )

        # TODO: It would be good to check that the support of all the component
        # distributions match, but for now we just check the type, since __eq__
        # isn't consistently implemented for all support types.
        self._support = support
        if support is None:
            support_type = type(component_distributions[0].support)
            if any(
                type(d.support) is not support_type for d in component_distributions[1:]
            ):
                raise ValueError(
                    "All component distributions must have the same support."
                )
        else:
            assert isinstance(support, constraints.Constraint), (
                "support must be a Constraint object"
            )

        self._component_distributions = component_distributions

        batch_shape = lax.broadcast_shapes(
            *(d.batch_shape for d in component_distributions)
        )
        event_shape = component_distributions[0].event_shape
        for d in component_distributions[1:]:
            if d.event_shape != event_shape:
                raise ValueError(
                    "All component distributions must have the same event shape"
                )

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def component_distributions(self):
        """The list of component distributions in the mixture.

        :return: The list of component distributions
        :rtype: list[Distribution]
        """
        return self._component_distributions

    @constraints.dependent_property
    def support(self):
        if self._support is not None:
            return self._support
        return self.component_distributions[0].support

    @property
    def is_discrete(self):
        return self.component_distributions[0].is_discrete

    @property
    def component_mean(self):
        return jnp.stack(
            [d.mean for d in self.component_distributions], axis=self.mixture_dim
        )

    @property
    def component_variance(self):
        return jnp.stack(
            [d.variance for d in self.component_distributions], axis=self.mixture_dim
        )

    def component_cdf(self, samples):
        return jnp.stack(
            [d.cdf(samples) for d in self.component_distributions],
            axis=self.mixture_dim,
        )

    def component_sample(self, key, sample_shape=()):
        keys = jax.random.split(key, self.mixture_size)
        samples = []
        for k, d in zip(keys, self.component_distributions):
            samples.append(d.expand(sample_shape + self.batch_shape).sample(k))
        return jnp.stack(samples, axis=self.mixture_dim)

    def component_log_probs(self, value: ArrayLike) -> ArrayLike:
        # modified implementation of numpyro.distributions.MixtureGeneral.component_log_probs
        component_log_probs = []
        for d in self.component_distributions:
            log_prob = d.log_prob(value)
            if (self._support is not None) and (not d._validate_args):
                mask = d.support(value)
                log_prob = jnp.where(mask, log_prob, -jnp.inf)
            component_log_probs.append(log_prob)
        component_log_probs = jnp.stack(component_log_probs, axis=-1)
        return self.log_scales + component_log_probs

    @property
    def mixture_size(self):
        """The number of components in the mixture."""
        return self._mixture_size

    @property
    def mixture_dim(self):
        return -self.event_dim - 1

    @property
    def mean(self):
        probs = jnp.exp(self.log_scales)
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        weighted_component_means = probs * self.component_mean
        return jnp.sum(weighted_component_means, axis=self.mixture_dim)

    @property
    def variance(self):
        # TODO(Qazalbash): Check the correctness
        probs = jnp.exp(self.log_scales)
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        mean_cond_var = jnp.sum(probs * self.component_variance, axis=self.mixture_dim)
        sq_deviation = (
            self.component_mean - jnp.expand_dims(self.mean, axis=self.mixture_dim)
        ) ** 2
        var_cond_mean = jnp.sum(probs * sq_deviation, axis=self.mixture_dim)
        return mean_cond_var + var_cond_mean

    def cdf(self, samples):
        """The cumulative distribution function.

        :param value: samples from this distribution.
        :return: output of the cumulative distribution function evaluated at
            `value`.
        :raises: NotImplementedError if the component distribution does not
            implement the cdf method.
        """
        cdf_components = self.component_cdf(samples)
        return jnp.sum(cdf_components * jnp.exp(self.log_scales), axis=-1)

    def sample_with_intermediates(self, key, sample_shape=()):
        """A version of ``sample`` that also returns the sampled component indices.

        :param jax.random.PRNGKey key: the rng_key key to be used for the distribution.
        :param tuple sample_shape: the sample shape for the distribution.
        :return: A 2-element tuple with the samples from the distribution, and the
            indices of the sampled components.
        :rtype: tuple
        """
        assert is_prng_key(key)
        key_comp, key_ind = jax.random.split(key)
        samples = self.component_sample(key_comp, sample_shape=sample_shape)

        # Sample selection indices from the categorical (shape will be sample_shape)
        indices: Array = categorical(
            key_ind,
            jax.nn.softmax(self.log_scales, axis=-1),
            shape=sample_shape + self.batch_shape,
        )
        n_expand = self.event_dim + 1
        indices_expanded = indices.reshape(indices.shape + (1,) * n_expand)

        # Select samples according to indices samples from categorical
        samples_selected = jnp.take_along_axis(
            samples, indices=indices_expanded, axis=self.mixture_dim
        )

        # Final sample shape (*sample_shape, *batch_shape, *event_shape)
        return jnp.squeeze(samples_selected, axis=self.mixture_dim), [indices]

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key=key, sample_shape=sample_shape)[0]

    @validate_sample
    def log_prob(self, value, intermediates=None):
        del intermediates
        sum_log_probs = self.component_log_probs(value)
        safe_sum_log_probs = jnp.where(
            jnp.isneginf(sum_log_probs), -jnp.inf, sum_log_probs
        )
        return jax.nn.logsumexp(
            safe_sum_log_probs,
            where=~jnp.isneginf(sum_log_probs),
            axis=-1,
        )
