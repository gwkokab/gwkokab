# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Union

from jax import numpy as jnp
from jaxtyping import Array


def check_min_concentration_for_beta_dist(
    loc: Array,
    var: Array,
    /,
    *,
    alpha_min: Union[Array, float] = 1.0,
    beta_min: Union[Array, float] = 1.0,
) -> Array:
    r"""Check if the mean and variance are valid for a given minimum alpha and beta of a
    Beta distribution.

    .. math::
        \alpha > \alpha_{\mathrm{min}} \iff \mu^2 (1 - \mu) > \sigma^2 (\alpha_{\mathrm{min}} + \mu)

    .. math::
        \beta > \beta_{\mathrm{min}} \iff \mu (1 - \mu)^2 > \sigma^2 (\beta_{\mathrm{min}} + 1 - \mu)

    Parameters
    ----------
    loc : Array
        The location parameter (mean).
    var : Array
        The variance parameter.
    alpha_min : Union[Array, float]
        The minimum allowed value for alpha, default is 1.0.
    beta_min : Union[Array, float]
        The minimum allowed value for beta, default is 1.0.

    Returns
    -------
    Array
        A boolean array indicating whether the constraints are satisfied.
    """
    ensure_positive_alpha = jnp.greater(
        jnp.square(loc) * (1.0 - loc),
        var * (alpha_min + loc),
    )
    ensure_positive_beta = jnp.greater(
        loc * jnp.square(1.0 - loc),
        var * (beta_min + 1.0 - loc),
    )
    valid_var = jnp.logical_and(ensure_positive_alpha, ensure_positive_beta)
    return valid_var
