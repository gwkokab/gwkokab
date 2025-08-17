# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Literal, Optional, Tuple

from jax import lax, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key

from ...utils.tools import error_if
from ..constraints import all_constraint


class JointDistribution(Distribution):
    pytree_aux_fields = ("shaped_values",)
    pytree_data_fields = ("_support", "marginal_distributions")

    def __init__(
        self,
        *marginal_distributions: Distribution,
        flatten_method: Optional[Literal["deep", "shallow"]] = None,
        support: Optional[constraints.Constraint] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """Construct a joint distribution from one or more marginal distributions.

        You may pass individual `Distribution` instances or nest them inside
        :class:`JointDistribution`s. The `flatten_method` argument allows flattening of nested
        joints into a single flat list of marginals.

        Parameters
        ----------
        marginal_distributions : *Distribution
            One or more `Distribution` objects (or nested :class:`JointDistribution`s) that form
            the components of the joint distribution.
        flatten_method : Optional[Literal[&quot;deep&quot;, &quot;shallow&quot;]], optional
            If "shallow", one level of nested :class:`JointDistributions` will be flattened.
            If "deep", all levels of nested :class:`JointDistributions` will be recursively flattened.
            If None (default), the nesting is preserved as-is.
        support : Optional[constraints.Constraint], optional
            The constraint object representing the support of the joint distribution.
            If not provided, it is computed from the support of the marginals.
        validate_args : Optional[bool], optional
            Whether to validate distribution parameters and inputs. Default is None.

        Raises
        ------
        ValueError
             If no marginal distributions are provided.


        Example
        -------
        .. code::

            >>> from numpyro.distributions import Normal
            >>> from gwkokab.models.utils import JointDistribution

            >>> A = Normal(0, 1)
            >>> B = Normal(1, 1)
            >>> C = Normal(2, 1)
            >>> D = Normal(3, 1)
            >>> E = Normal(4, 1)

            >>> jd = JointDistribution(
            ...     A, JointDistribution(B, JointDistribution(C, D)), E
            ... )

            >>> len(jd.marginal_distributions)  # No flattening (default)
            3

            >>> jd = JointDistribution(
            ...     A,
            ...     JointDistribution(B, JointDistribution(C, D)),
            ...     E,
            ...     flatten_method="shallow",
            ... )
            >>> len(jd.marginal_distributions)  # Shallow flattening
            4

            >>> jd = JointDistribution(
            ...     A,
            ...     JointDistribution(B, JointDistribution(C, D)),
            ...     E,
            ...     flatten_method="deep",
            ... )
            >>> len(jd.marginal_distributions)  # Deep flattening
            5
        """
        error_if(
            not marginal_distributions,
            msg="At least one marginal distribution is required.",
        )

        marginal_flatten = _flatten_marginal_distributions(
            marginal_distributions, flatten_method
        )
        self.marginal_distributions: Sequence[Distribution] = tuple(marginal_flatten)
        self.shaped_values: Sequence[int | Tuple[int, int]] = tuple()
        batch_shape = lax.broadcast_shapes(
            *tuple(d.batch_shape for d in marginal_flatten)
        )
        k = 0
        for d in marginal_flatten:
            if d.event_shape:
                self.shaped_values += ((k, k + d.event_shape[0]),)
                k += d.event_shape[0]
            else:
                self.shaped_values += (k,)
                k += 1
        if support is None:
            self._support = all_constraint(
                [d.support for d in marginal_flatten], self.shaped_values
            )
        else:
            self._support = support
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
        samples_concatenated = jnp.concatenate(samples, axis=-1)
        return samples_concatenated


def _flatten_marginal_distributions(
    marginal_distributions: Sequence[Distribution | JointDistribution],
    flatten_method: Optional[Literal["deep", "shallow"]] = None,
) -> Sequence[Distribution | JointDistribution]:
    """Flatten marginal distributions based on the specified method.

    This function converts nested :class:`JointDistribution` structures into a flat sequence
    of `Distribution` objects, depending on the chosen `flatten_method`

    Parameters
    ----------
    marginal_distributions : Sequence[Distribution  |  JointDistribution]
        Sequence of Distribution or JointDistribution instances.
    flatten_method : Optional[Literal[&quot;deep&quot;, &quot;shallow&quot;]], optional
        If "shallow", flattens one level of nesting.\n
        If "deep", recursively flattens all levels.\n
        If None, returns the input as-is., by default None

    Returns
    -------
    Sequence[Distribution | JointDistribution]
        A flattened sequence of distributions.

    Example
    -------
        Given:
            JointDistribution(A, JointDistribution(B, JointDistribution(C, D)), E)

        - flatten_method=None
            => (A, JointDistribution(B, JointDistribution(C, D)), E)

        - flatten_method='shallow'
            => (A, B, JointDistribution(C, D), E)

        - flatten_method='deep'
            => (A, B, C, D, E)
    """
    error_if(
        flatten_method not in (None, "deep", "shallow"),
        msg=f"Unknown flatten method: {flatten_method}",
    )
    if flatten_method is None:
        return marginal_distributions
    flatten_dists: list[Distribution] = []
    for m_dist in marginal_distributions:
        if isinstance(m_dist, JointDistribution):
            if flatten_method == "shallow":
                m_dist_marginal_distributions = m_dist.marginal_distributions
            else:  # deep case
                m_dist_marginal_distributions = _flatten_marginal_distributions(
                    m_dist.marginal_distributions, flatten_method
                )
            flatten_dists.extend(m_dist_marginal_distributions)
        else:
            flatten_dists.append(m_dist)
    return flatten_dists
