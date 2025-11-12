# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple

import jax
import numpy as np
from jax import numpy as jnp
from jax._src.lax import lax
from jax._src.numpy.util import promote_args_inexact
from jax.scipy import special as scs
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


@jax.jit
def logsubexp(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Compute :math:`\log(\exp(a) - \exp(b))` in a numerically stable way.

    Parameters
    ----------
    a : ArrayLike
        input array
    b : ArrayLike
        input array

    Returns
    -------
    ArrayLike
        The value of :math:`\log(\exp(a) - \exp(b))`.
    """
    a, b = promote_args_inexact("logsubexp", a, b)
    return lax.add(a, lax.log1p(lax.neg(lax.exp(lax.sub(b, a)))))


@jax.jit
def truncnorm_logpdf(
    xx: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    low: ArrayLike,
    high: ArrayLike,
) -> ArrayLike:
    """Compute the log probability density function of a truncated normal distribution.

    Parameters
    ----------
    xx : ArrayLike
        The input array.
    loc : ArrayLike
        The mean of the distribution.
    scale : ArrayLike
        The standard deviation of the distribution.
    low : ArrayLike
        The lower bound of the distribution.
    high : ArrayLike
        The upper bound of the distribution.

    Returns
    -------
    ArrayLike
        The log probability density function of the truncated normal distribution.
    """
    xx, loc, scale, low, high = promote_args_inexact(
        "truncnorm_logpdf", xx, loc, scale, low, high
    )
    safe_scale = jnp.where(scale <= 0, 1.0, scale)
    zz = lax.div(lax.sub(xx, loc), safe_scale)
    aa = lax.div(lax.sub(low, loc), safe_scale)
    bb = lax.div(lax.sub(high, loc), safe_scale)
    constant = lax._const(xx, np.log(2.0 * np.pi))
    neg_half = lax._const(xx, -0.5)
    log_pdf = lax.sub(
        lax.mul(neg_half, lax.add(lax.square(zz), constant)), lax.log(safe_scale)
    )

    # cf https://github.com/scipy/scipy/blob/v1.15.1/scipy/stats/_continuous_distns.py#L10189
    log_norm = jnp.select(
        [bb <= 0, aa > 0, bb > 0],
        [
            logsubexp(scs.log_ndtr(bb), scs.log_ndtr(aa)),
            logsubexp(scs.log_ndtr(-aa), scs.log_ndtr(-bb)),
            lax.log1p(-scs.ndtr(aa) - scs.ndtr(-bb)),
        ],
        np.nan,
    )
    log_pdf -= log_norm
    return jnp.where((xx < low) | (xx > high) | (scale <= 0), -np.inf, log_pdf)
