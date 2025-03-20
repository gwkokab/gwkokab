# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from quadax import cumulative_trapezoid


def beta_dist_concentrations_to_mean_variance(
    alpha: ArrayLike,
    beta: ArrayLike,
    loc: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
) -> Tuple[ArrayLike, ArrayLike]:
    r"""Let :math:`\alpha` and :math:`\beta` be the shape parameters of a beta
    distribution, :math:`a` being the location and :math:`b` being the scale. This
    function returns the mean and variance of the distribution. Then concentrations are
    given by:

    .. math::
        \mu = a+b\frac{\alpha}{\alpha + \beta}\qquad
        \sigma^2 = b^2\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}

    Parameters
    ----------
    alpha : ArrayLike
        The shape parameter :math:`\alpha`.
    beta : ArrayLike
        The shape parameter :math:`\beta`.
    loc : ArrayLike
        The location :math:`a` of the beta distribution.
    scale : ArrayLike
        The scale :math:`b` of the beta distribution.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        The mean :math:`\mu` and variance :math:`\sigma^2` of the beta distribution.
    """
    sum_of_concentrations = jnp.add(alpha, beta)
    product_of_concentrations = jnp.multiply(alpha, beta)

    mean = jnp.divide(alpha, sum_of_concentrations)

    variance = jnp.add(sum_of_concentrations, 1)
    variance = jnp.multiply(jnp.square(sum_of_concentrations), variance)
    variance = jnp.divide(product_of_concentrations, variance)

    return loc + scale * mean, scale**2 * variance


def beta_dist_mean_variance_to_concentrations(
    mean: ArrayLike,
    variance: ArrayLike,
    loc: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
) -> Tuple[ArrayLike, ArrayLike]:
    r"""Let :math:`\mu` and :math:`\sigma^2` be the mean and variance of a beta
    distribution, :math:`a` being the location and :math:`b` being the scale. This
    function returns the shape parameters :math:`\alpha` and :math:`\beta` of the
    distribution. Then concentrations are given by:

    .. math::
        \alpha = -\frac{\mu-a}{b} \left(\left(\frac{\mu-a}{\sigma}\right)\left(\frac{\mu-a-b}{\sigma}\right)+1\right)\qquad
        \beta = \alpha\left(\frac{b}{\mu-a}-1\right)

    Parameters
    ----------
    mean : ArrayLike
        The mean :math:`\mu` of the beta distribution.
    variance : ArrayLike
        The variance :math:`\sigma^2` of the beta distribution.
    loc : ArrayLike
        The location :math:`a` of the beta distribution.
    scale : ArrayLike
        The scale :math:`b` of the beta distribution.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        The shape parameters :math:`\alpha` and :math:`\beta` of the beta distribution.
    """
    low = loc
    high = loc + scale
    mean_shifted_by_low = mean - low
    mean_shifted_by_high = mean - high
    shared_term = (mean_shifted_by_low * mean_shifted_by_high + variance) / (
        scale * variance
    )
    alpha = -mean_shifted_by_low * shared_term
    beta = mean_shifted_by_high * shared_term
    return alpha, beta


def cumtrapz(y: Array, x: Array) -> Array:
    """Calculate the cumulative trapezoidal integration of y with respect to x.

    Parameters
    ----------
    y : Array
        array to integrate
    x : Array
        array to integrate over

    Returns
    -------
    Array
        The result of the cumulative trapezoidal integration of y with respect to x.
    """
    y_cum = y
    for index in range(x.shape[-1]):
        y_cum = cumulative_trapezoid(y=y_cum, x=x[..., index], axis=index, initial=0.0)
    return y_cum
