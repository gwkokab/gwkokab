# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Mixture distribution whose components carry log rates rather than weights.

:class:`ScaledMixture` is the central abstraction of the model layer: because the
component scales are unnormalised log rates, the mixture integrates to the total merger
rate rather than to one. That is what lets the Poisson-mean estimators in
:mod:`gwkokab.poisson_mean` read the expected number of detections straight off the
model, and hence what lets rates and population shapes be inferred jointly.
"""

from typing import List, Optional

import jax
from jax import lax, numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import categorical, is_prng_key, validate_sample


class ScaledMixture(Distribution):
    r"""A finite mixture of component distributions from different families.

    This is a generalization of :class:`~numpyro.distributions.Mixture` where the
    component distributions are scaled by a set of rates. The scales are *not*
    normalised to sum to one, so

    .. math::
        p(x) = \sum_{k} \mathcal{R}_k \, p_k(x),
        \qquad \log \mathcal{R}_k = \texttt{log\_scales}_k

    integrates to :math:`\sum_k \mathcal{R}_k` -- the total rate -- rather than to unity.
    The Poisson-mean estimators rely on this, so they require a :class:`ScaledMixture`
    specifically.

    Parameters
    ----------
    log_scales : Array
        Log rates :math:`\log\mathcal{R}_k` of the components; the trailing axis fixes
        the mixture size.
    component_distributions : List[Distribution]
        The component distributions. All must share an event shape and a support type,
        and their number must match the mixture size.
    support : Optional[constraints.Constraint], optional
        Support of the mixture. Defaults to :data:`None`, in which case the first
        component's support is used and every component is required to have the same
        support type. Pass it explicitly when components have genuinely different
        supports; out-of-support components are then masked in
        :meth:`component_log_probs`.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Raises
    ------
    ValueError
        If ``component_distributions`` is not a list of
        :class:`~numpyro.distributions.Distribution` objects, if its length does not
        match the mixture size, or if the components disagree on support type or event
        shape.

    Examples
    --------
    .. code::

       >>> import jax
       >>> import jax.random as jrd
       >>> import numpyro.distributions as dist
       >>> from gwkokab.models.utils import ScaledMixture
       >>> log_scales = jrd.uniform(jrd.key(42), (3,), minval=0, maxval=5)
       >>> component_dists = [
       ...     dist.Normal(loc=0.0, scale=1.0),
       ...     dist.Normal(loc=-0.5, scale=0.3),
       ...     dist.Normal(loc=0.6, scale=1.2),
       ... ]
       >>> mixture = ScaledMixture(log_scales, component_dists)
       >>> mixture.sample(jax.random.key(42)).shape
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

        Returns
        -------
        List[Distribution]
            The component distributions, in the order matching ``log_scales``.
        """
        return self._component_distributions

    @constraints.dependent_property
    def support(self):
        """The support of the mixture.

        Returns
        -------
        constraints.Constraint
            The support passed to the constructor, or the first component's support when
            none was given.
        """
        if self._support is not None:
            return self._support
        return self.component_distributions[0].support

    @property
    def is_discrete(self):
        """Whether the mixture is discrete.

        Returns
        -------
        bool
            The first component's ``is_discrete`` flag; all components are required to
            share a support type.
        """
        return self.component_distributions[0].is_discrete

    @property
    def component_mean(self):
        """Means of the component distributions.

        Returns
        -------
        Array
            The component means stacked along :attr:`mixture_dim`.
        """
        return jnp.stack(
            [d.mean for d in self.component_distributions], axis=self.mixture_dim
        )

    @property
    def component_variance(self):
        """Variances of the component distributions.

        Returns
        -------
        Array
            The component variances stacked along :attr:`mixture_dim`.
        """
        return jnp.stack(
            [d.variance for d in self.component_distributions], axis=self.mixture_dim
        )

    def component_cdf(self, samples):
        """Cumulative distribution functions of the components.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the component CDFs.

        Returns
        -------
        Array
            The component CDF values stacked along :attr:`mixture_dim`.

        Raises
        ------
        NotImplementedError
            If any component does not implement ``cdf``.
        """
        return jnp.stack(
            [d.cdf(samples) for d in self.component_distributions],
            axis=self.mixture_dim,
        )

    def component_sample(self, key, sample_shape=()):
        """Draw one sample batch from every component.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key, split one way per component.
        sample_shape : tuple[int, ...]
            Shape of the sample batch to draw. Defaults to ``()``.

        Returns
        -------
        Array
            Samples of shape ``sample_shape + batch_shape + (mixture_size,) + event_shape``,
            with the component axis at :attr:`mixture_dim`.
        """
        keys = jax.random.split(key, self.mixture_size)
        samples = []
        for k, d in zip(keys, self.component_distributions):
            samples.append(d.expand(sample_shape + self.batch_shape).sample(k))
        return jnp.stack(samples, axis=self.mixture_dim)

    def component_log_probs(self, value: ArrayLike) -> ArrayLike:
        # modified implementation of numpyro.distributions.MixtureGeneral.component_log_probs
        r"""Log densities of the components, offset by their log rates.

        When an explicit ``support`` was given, each component's own support is checked and
        out-of-support values are masked to :math:`-\infty`, so a component contributes
        nothing outside the region where it is defined.

        Parameters
        ----------
        value : ArrayLike
            Points at which to evaluate the components.

        Returns
        -------
        ArrayLike
            Array of shape ``batch_shape + (mixture_size,)`` holding
            :math:`\log \mathcal{R}_k + \log p_k(x)`.
        """
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
        """The number of components in the mixture.

        Returns
        -------
        int
            Size of the trailing axis of ``log_scales``.
        """
        return self._mixture_size

    @property
    def mixture_dim(self):
        """Axis along which components are stacked.

        Returns
        -------
        int
            The negative axis index ``-event_dim - 1``, i.e. the axis just before the event
            axes.
        """
        return -self.event_dim - 1

    @property
    def mean(self):
        r"""Rate-weighted mean of the mixture.

        Returns
        -------
        Array
            :math:`\sum_k \mathcal{R}_k \mu_k`. Note that because the scales are rates
            rather than normalised weights, this is the first moment of the *unnormalised*
            density, not the mean of a probability distribution.
        """
        probs = jnp.exp(self.log_scales)
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        weighted_component_means = probs * self.component_mean
        return jnp.sum(weighted_component_means, axis=self.mixture_dim)

    @property
    def variance(self):
        # TODO(Qazalbash): Check the correctness
        """Rate-weighted variance of the mixture.

        Computed by the law of total variance, as the rate-weighted mean of the component
        variances plus the rate-weighted variance of the component means.

        Returns
        -------
        Array
            The mixture variance, in the same unnormalised sense as :attr:`mean`.
        """
        probs = jnp.exp(self.log_scales)
        probs = probs.reshape(probs.shape + (1,) * self.event_dim)
        mean_cond_var = jnp.sum(probs * self.component_variance, axis=self.mixture_dim)
        sq_deviation = (
            self.component_mean - jnp.expand_dims(self.mean, axis=self.mixture_dim)
        ) ** 2
        var_cond_mean = jnp.sum(probs * sq_deviation, axis=self.mixture_dim)
        return mean_cond_var + var_cond_mean

    def cdf(self, samples):
        r"""The cumulative distribution function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the CDF.

        Returns
        -------
        Array
            :math:`\sum_k \mathcal{R}_k F_k(x)`, the rate-weighted sum of the component
            CDFs.

        Raises
        ------
        NotImplementedError
            If any component distribution does not implement ``cdf``.
        """
        cdf_components = self.component_cdf(samples)
        return jnp.sum(cdf_components * jnp.exp(self.log_scales), axis=-1)

    def sample_with_intermediates(self, key, sample_shape=()):
        """Sample, also returning the indices of the components each sample came from.

        The component is chosen from a categorical over ``softmax(log_scales)``, i.e. the
        rates normalised to weights.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key.
        sample_shape : tuple[int, ...]
            Shape of the sample batch to draw. Defaults to ``()``.

        Returns
        -------
        tuple[Array, list[Array]]
            The samples, of shape ``sample_shape + batch_shape + event_shape``, and a
            single-element list holding the sampled component indices.
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
        """Draw samples from the mixture.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key.
        sample_shape : tuple[int, ...]
            Shape of the sample batch to draw. Defaults to ``()``.

        Returns
        -------
        Array
            Samples of shape ``sample_shape + batch_shape + event_shape``.
        """
        return self.sample_with_intermediates(key=key, sample_shape=sample_shape)[0]

    @validate_sample
    def log_prob(self, value, intermediates=None):
        r"""Log of the rate-weighted density.

        .. math::
            \log p(x) = \log \sum_k \exp\left(\log\mathcal{R}_k + \log p_k(x)\right)

        The reduction masks out components that contribute :math:`-\infty`, so a value
        outside one component's support does not poison the whole sum with a NaN gradient.

        Parameters
        ----------
        value : Array
            Points at which to evaluate the mixture.
        intermediates : Any, optional
            Accepted for API compatibility with
            :class:`~numpyro.distributions.MixtureGeneral` and ignored.

        Returns
        -------
        Array
            The log density, of shape ``batch_shape``.
        """
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
