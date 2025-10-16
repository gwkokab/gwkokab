# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Tuple, Union

import jax
from jax import numpy as jnp
from jax.scipy.special import xlog1py
from jaxtyping import Array, ArrayLike, PRNGKeyArray
from numpyro import distributions as dist
from numpyro._typing import DistributionT
from numpyro.distributions import constraints, Delta, Distribution
from numpyro.distributions.util import validate_sample

from gwkokab.utils.tools import error_if
from kokab.utils.regex import match_all


class _DirichletElement(Distribution):
    """DirichletElement distribution for order < n_dimensions - 1 and n_dimensions > 1."""

    arg_constraints = {
        "order": constraints.nonnegative_integer,
        "n_dimensions": constraints.positive_integer,
    }
    pytree_data_fields = ("order", "n_dimensions", "sum_of_concentrations")

    def __init__(
        self,
        order: float,
        n_dimensions: float,
        sum_of_concentrations: ArrayLike,
        validate_args: Optional[bool] = None,
    ):
        self.order = order
        self.n_dimensions = n_dimensions
        self.sum_of_concentrations = sum_of_concentrations
        batch_shape = jnp.shape(jnp.array(self.sum_of_concentrations))
        super(_DirichletElement, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(0.0, 1.0 - self.sum_of_concentrations)

    @validate_sample
    def log_prob(self, value: Array) -> Array:
        constant = self.n_dimensions - self.order - 1
        log_prob_unnorm = xlog1py(constant - 1, -self.sum_of_concentrations - value)  # type: ignore
        log_norm = xlog1py(constant, -self.sum_of_concentrations) - jnp.log(constant)  # type: ignore
        return log_prob_unnorm - log_norm

    def sample(self, key: PRNGKeyArray, sample_shape: Tuple[int, ...] = ()) -> Array:
        u = jax.random.uniform(key, shape=sample_shape)
        inv_constant = 1.0 / (self.n_dimensions - self.order - 1)
        return (1.0 - self.sum_of_concentrations) * (1.0 - jnp.power(u, inv_constant))  # type: ignore


def DirichletElement(
    order: int,
    n_dimensions: int,
    validate_args: Optional[bool] = None,
    **kwargs,
) -> DistributionT:
    r"""Conditional Dirichlet Distribution.

    A `DirichletElement` is a distribution over the :math:`N`-dimensional simplex, where
    :math:`N` is the :code:`n_dimensions` parameter. The :math:`n`-th order Dirichlet
    distribution given by the :code:`order`. The distribution is defined as,

    .. math::
        \forall n \in \{0, N-2\}, p(x_n \mid x_0, \cdots, x_{n-1}) =
        \frac{(N-n-1)(1-\alpha_n-x_n)^{N-n-1}}{(1-\alpha_n)^{N-n-1}},
        \qquad x_n \in [0, \alpha_n]

    where :math:`\alpha_n = \sum_{i=0}^{n-1} \alpha_i` and :math:`x_n` is the
    :code:`order`-th element of the simplex.

    .. math::
        p(x_{N-1}\mid x_0, \cdots, x_{N-2}) = \frac{1}{1-\alpha_{N-1}},
        \qquad x_{N-1} \in [0,\alpha_{N-1}]

    Parameters
    ----------
    order : int
        order of the DirichletElement
    n_dimensions : int
        number of dimensions
    validate_args : Optional[bool], optional
        whether to validate the arguments, by default None

    Returns
    -------
    DistributionT
        DirichletElement distribution

    Raises
    ------
    KeyError
        Missing concentration parameters for `DirichletElement` of order {order}
    """
    error_if(
        isinstance(order, int) is False or order < 0,
        msg="`order` must be a non-negative integer.",
    )
    error_if(
        isinstance(n_dimensions, int) is False or n_dimensions < 2,
        msg="`n_dimensions` must be an integer greater than 1. "
        "If your problem is 1D, use a Uniform prior instead.",
    )
    try:
        concentrations = [kwargs["concentration" + str(i)] for i in range(order)]
    except KeyError as e:
        raise KeyError(
            f"Missing concentration parameters for `DirichletElement` of order {order}"
        ) from e

    sum_of_concentrations = sum(concentrations, start=0.0)
    high = 1.0 - sum_of_concentrations

    if order == n_dimensions - 1:
        return Delta(v=high, log_density=0.0, validate_args=validate_args)

    return _DirichletElement(
        order=float(order),
        n_dimensions=float(n_dimensions),
        sum_of_concentrations=sum_of_concentrations,
        validate_args=validate_args,
    )


def _available_prior(name: str) -> DistributionT:
    """Get the available prior from numpyro distributions.

    Parameters
    ----------
    name : str
        name of the prior

    Returns
    -------
    DistributionT
        prior distribution

    Raises
    ------
    AttributeError
        If the prior is not found in numpyro distributions
    """
    gwkokab_priors = {"DirichletElement": DirichletElement}
    error_if(
        name not in dist.__all__ + list(gwkokab_priors.keys()),
        AttributeError,
        f"Prior {name} not found. Available priors are: "
        + ", ".join(dist.__all__ + list(gwkokab_priors.keys())),
    )
    if name in gwkokab_priors:
        return gwkokab_priors[name]
    return getattr(dist, name)


def _is_lazy_prior(prior_dict: dict[str, Union[str, float]]) -> bool:
    """Check if the prior is a lazy prior.

    Parameters
    ----------
    prior_dict : dict[str, Union[str, float]]
        dictionary of prior

    Returns
    -------
    bool
        True if the prior is a lazy prior, False otherwise
    """
    for value in prior_dict.values():
        if isinstance(value, str):
            return True
    return False


def get_processed_priors(params: List[str], priors: dict) -> dict:
    """Get the processed priors from a list of parameters. A processed prior is either
    an instantiated prior or a tuple of :code:`(jax.tree_util.Partial, lazy_vars)` where
    :code:`lazy_vars` is a dictionary of lazy variables.

    Parameters
    ----------
    params : List[str]
        list of parameters
    priors : dict
        dictionary of priors

    Returns
    -------
    dict
        dictionary of processed priors

    Raises
    ------
    ValueError
        if the prior value is invalid
    """
    matched_prior_params = match_all(params, priors)
    for param in params:
        error_if(param not in matched_prior_params, msg=f"Missing prior for {param}")

    for key, value in matched_prior_params.items():
        # if the value is not a dict, means its a constant/duplicate value
        if not isinstance(value, dict):
            continue
        value_cpy = value.copy()
        dist_type = value_cpy.pop("dist", None)
        error_if(
            not isinstance(dist_type, str) or dist_type == "",
            msg=f"Prior for '{key}' must specify a 'dist' string field.",
        )
        prior = _available_prior(dist_type)

        # if there are no lazy variables, instantiate the prior
        if not _is_lazy_prior(value_cpy):
            matched_prior_params[key] = prior(**value_cpy, validate_args=True)
            continue

        # separate the lazy variables and the non-lazy variables
        lazy_vars = {k: v for k, v in value_cpy.items() if isinstance(v, str)}
        non_lazy_vars = {k: v for k, v in value_cpy.items() if not isinstance(v, str)}
        matched_prior_params[key] = (
            jax.tree_util.Partial(prior, **non_lazy_vars, validate_args=True),
            lazy_vars,
        )
    return matched_prior_params
