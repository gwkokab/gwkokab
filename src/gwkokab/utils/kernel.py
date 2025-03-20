# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import jax
from jax import nn as jnn, numpy as jnp
from jaxtyping import ArrayLike


@jax.jit
def log_planck_taper_window(x: ArrayLike) -> ArrayLike:
    r"""If :math:`x` is the point at which to evaluate the window, then the Planck taper
    window is defined as,

    .. math::

        S(x)=\begin{cases}
            0                                                                   & \text{if } x < 0,           \\
            \displaystyle\frac{1}{1+e^{\left(\frac{1}{x}+\frac{1}{x-1}\right)}} & \text{if } 0 \leq x \leq 1, \\
            1                                                                   & \text{if } x > 1,           \\
        \end{cases}

    This function evaluates the log of the Planck taper window :math:`\ln{S(x)}`.

    Parameters
    ----------
    x: ArrayLike
        point at which to evaluate the window

    Returns
    -------
    ArrayLike
        window value
    """
    inv_1 = jnp.where(x == 0.0, 0.0, 1.0 / jnp.where(x == 0.0, 1.0, x))
    inv_2 = jnp.where(x == 1.0, 0.0, 1.0 / jnp.where(x == 1.0, 1.0, x - 1.0))
    return jnp.where(
        x <= 0,
        -jnp.inf,
        jnp.where(x < 1.0, -jnn.softplus(inv_1 + inv_2), 0.0),
    )
