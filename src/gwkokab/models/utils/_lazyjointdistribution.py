# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Dict, List, Literal, Optional, Tuple, Union

import jax
from jax import lax, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro._typing import ConstraintT, DistributionT
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key

from ...utils.tools import error_if, warn_if


class _LazyConstraint(constraints.Constraint):
    """A constraint that checks if a value is in the support of a
    :class:`LazyJointDistribution`.
    """

    def __init__(
        self,
        *marginal_distributions: Union[DistributionT, jax.tree_util.Partial],
        dependencies: Dict[int, Dict[str, int]],
        event_slices: Sequence[int | Tuple[int, int]],
    ) -> None:
        self.marginal_distributions = tuple(marginal_distributions)
        self.dependencies = dependencies
        self.event_slices = event_slices

    def __call__(self, x: Array) -> Array:
        marginal_dists = list(self.marginal_distributions)
        for i, dep in self.dependencies.items():
            kwargs = {
                k: jax.lax.dynamic_index_in_dim(x, v, axis=-1, keepdims=False)
                for k, v in dep.items()
            }
            mdist = marginal_dists[i]
            marginal_dists[i] = mdist.func(*mdist.args, **mdist.keywords, **kwargs)  # type: ignore
        mask = None
        for mdist, event_slice in zip(marginal_dists, self.event_slices, strict=True):
            constraint: ConstraintT = mdist.support  # type: ignore
            if isinstance(event_slice, int):
                x_slice = lax.dynamic_index_in_dim(
                    x, event_slice, axis=-1, keepdims=False
                )
            else:
                x_slice = lax.dynamic_slice_in_dim(
                    x,
                    event_slice[0],
                    event_slice[1] - event_slice[0],
                    axis=-1,
                )
            if mask is None:
                mask = constraint.check(x_slice)
            else:
                mask = jnp.logical_and(mask, constraint.check(x_slice))
        return mask  # type: ignore

    def tree_flatten(self):
        return (self.marginal_distributions, self.dependencies, self.event_slices), (
            ("marginal_distributions", "dependencies", "event_slices"),
            dict(),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LazyConstraint):
            return False
        return (
            all(
                constraint == other_constraint
                for constraint, other_constraint in zip(
                    self.marginal_distributions,
                    other.marginal_distributions,
                    strict=True,
                )
            )
            and self.dependencies == other.dependencies
            and self.event_slices == other.event_slices
        )


class LazyJointDistribution(Distribution):
    pytree_aux_fields = ("shaped_values", "partial_order", "dependencies")
    pytree_data_fields = ("_support", "marginal_distributions")

    def __init__(
        self,
        *marginal_distributions: Union[DistributionT, jax.tree_util.Partial],
        dependencies: Dict[int, Dict[str, int]],
        partial_order: List[int],
        dependencies_event_shape: Optional[List[Tuple[int, ...]]] = None,
        flatten_method: Optional[Literal["deep", "shallow"]] = None,
        support: Optional[ConstraintT] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        """Construct a joint distribution from one or more marginal distributions.

        You may pass individual `Distribution` instances or nest them inside
        :class:`LazyJointDistribution`s. The `flatten_method` argument allows flattening of nested
        joints into a single flat list of marginals.

        Parameters
        ----------
        marginal_distributions : *Union[DistributionT, jax.tree_util.Partial]
            One or more marginal distributions. Each marginal distribution can be an instance of
            `numpyro.distributions.Distribution` or a `jax.tree_util.Partial` that returns a
            `numpyro.distributions.Distribution` when called with its arguments.
        dependencies : Dict[int, Dict[str, int]]
            A dictionary mapping the index of each marginal distribution that is a
            `jax.tree_util.Partial` to another dictionary that maps the names of its dependency
            parameters to the indices of the marginal distributions they depend on.
            This is used to specify which variables each lazy variable depends on.
        partial_order : Optional[Tuple[str, int, int]], optional
            A tuple defining a partial order for the lazy variables.
            Each entry in the tuple should be of the form (var_name, event_index, marginal_index),
            such that elements coming earlier in the tuple are not dependent on elements coming later.
            This is used to ensure that when sampling from the joint distribution, the lazy variables
            are sampled in an order that respects their dependencies.
        dependencies_event_shape : Optional[List[Tuple[int, ...]]], optional
            A list of event shapes for the dependencies of each marginal distribution.
            This is used to validate the shapes of the dependency variables when constructing
        flatten_method : Optional[Literal[&quot;deep&quot;, &quot;shallow&quot;]], optional
            Currently not used.
            If "shallow", one level of nested :class:`LazyJointDistributions` will be flattened.
            If "deep", all levels of nested :class:`LazyJointDistributions` will be recursively flattened.
            If None (default), the nesting is preserved as-is.
        support : Optional[ConstraintT], optional
            The constraint object representing the support of the joint distribution.
            If not provided, it is computed from the support of the marginals.
        validate_args : Optional[bool], optional
            Whether to validate distribution parameters and inputs. Default is None.

        Raises
        ------
        ValueError
             If no marginal distributions are provided.
        """
        error_if(
            not marginal_distributions,
            msg="At least one marginal distribution is required.",
        )
        error_if(
            len(partial_order) == 0,
            msg="`partial_order` must be provided. If there is not lazy "
            "variable then use a standard `gwkokab.models.JointDistribution` instead.",
        )
        error_if(
            dependencies is None,
            msg="`dependencies` must be provided. If there is not lazy "
            "variable then use a standard `gwkokab.models.JointDistribution` instead.",
        )
        error_if(
            len(partial_order) != len(dependencies),
            msg="`partial_order` and `dependencies` must have the same length.",
        )
        warn_if(
            flatten_method is not None,
            msg="The `flatten_method` argument is not used with in "
            "`LazyJointDistribution`. It will be ignored.",
        )

        # TODO(Qazalbash): Implement flattening logic
        # marginal_flatten = _flatten_marginal_distributions(
        #     marginal_distributions, flatten_method
        # )
        marginal_flatten = marginal_distributions
        self.marginal_distributions: Sequence[
            Union[DistributionT, jax.tree_util.Partial]
        ] = tuple(marginal_flatten)
        self.shaped_values: Sequence[Union[int, Tuple[int, int]]] = tuple()
        dependencies = dependencies or {}  # for type checker
        k = 0
        for i, d in enumerate(marginal_flatten):
            error_if(
                not isinstance(d, (Distribution, jax.tree_util.Partial)),
                msg="All marginals must be instances of "
                "`numpyro.distributions.Distribution` and `jax.tree_util.Partial`. "
                f"Got {type(d)}",
            )
            if isinstance(d, jax.tree_util.Partial):
                event_shape = (
                    dependencies_event_shape[i] if dependencies_event_shape else ()
                )
            else:
                event_shape = d.event_shape
            if event_shape:
                self.shaped_values += ((k, k + event_shape[0]),)
                k += event_shape[0]
            else:
                self.shaped_values += (k,)
                k += 1
        batch_shape = ()
        if support is None:
            self._support = _LazyConstraint(
                *self.marginal_distributions,
                dependencies=dependencies,
                event_slices=self.shaped_values,
            )
        else:
            self._support = support

        self.partial_order = partial_order
        self.dependencies = dependencies

        super(LazyJointDistribution, self).__init__(
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
        marginal_dists: List[DistributionT] = list(self.marginal_distributions)  # type: ignore
        for i, dep in self.dependencies.items():
            kwargs = {
                k: jax.lax.dynamic_index_in_dim(value, v, axis=-1, keepdims=False)
                for k, v in dep.items()
            }
            mdist: jax.tree_util.Partial = marginal_dists[i]
            marginal_dists[i] = mdist.func(*mdist.args, **mdist.keywords, **kwargs)  # type: ignore
        log_prob_val = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for m_dist, event_slice in zip(marginal_dists, self.shaped_values, strict=True):
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
        n_total = len(self.marginal_distributions)
        independent = set(range(n_total)) - set(self.partial_order)

        samples = [jnp.empty((*sample_shape, 1)) for _ in range(n_total)]

        for i in independent:
            mdist: DistributionT = self.marginal_distributions[i]  # type: ignore
            key, subkey = jrd.split(key)
            samples[i] = mdist.sample(subkey, sample_shape).reshape(*sample_shape, -1)

        for i in self.partial_order:
            key, subkey = jrd.split(key)
            kwargs = {k: samples[v] for k, v in self.dependencies[i].items()}
            dist = self.marginal_distributions[i]
            if isinstance(dist, jax.tree_util.Partial):
                dist = dist.func(*dist.args, **dist.keywords, **kwargs)  # type: ignore
            samples[i] = dist.sample(
                subkey, sample_shape[dist.event_dim + 1 :]
            ).reshape(*sample_shape, -1)

        samples_concatenated = jnp.concatenate(samples, axis=-1)
        return samples_concatenated
