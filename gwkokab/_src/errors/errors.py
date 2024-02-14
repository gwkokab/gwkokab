#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

from jax import numpy as jnp, vmap
from jax.random import normal, truncated_normal, uniform
from jaxtyping import Array

from ..utils import chirp_mass, get_key, symmetric_mass_ratio


def error_factory(error_type: str, **kwargs) -> Array:
    """Factory function to create different types of errors.

    :param error_type: name of the error
    :raises ValueError: if the error type is unknown
    :return: error values for the given error type
    """
    if error_type == "normal":
        return normal_error(**kwargs)
    elif error_type == "truncated_normal":
        return truncated_normal_error(**kwargs)
    elif error_type == "uniform":
        return uniform_error(**kwargs)
    elif error_type == "banana":
        return banana_error(**kwargs)
    else:
        raise ValueError(f"Unknown error type: {error_type}")


def normal_error(x: Array, size: int, *, scale: float) -> Array:
    r"""Add normal error to the given values.

    .. math::
        x' = \mathcal{N}(\mu=x, \sigma=scale)

    :param x: given values
    :param size: number of samples
    :param scale: standard deviation of the normal distribution
    :return: error values
    """
    return vmap(
        lambda x_: normal(
            key=get_key(),
            shape=(size,),
            dtype=x.dtype,
        )
        * scale
        + x_
    )(x)


def truncated_normal_error(x: Array, size: int, *, scale: float, lower: float, upper: float) -> Array:
    r"""Add truncated normal error to the given values.

    .. math::
        x' = \mathcal{N}(\mu=x, \sigma=scale) \cap [lower, upper]

    :param x: given values
    :param size: number of samples
    :param scale: standard deviation of the normal distribution
    :param lower: lower bound of the truncated normal distribution
    :param upper: upper bound of the truncated normal distribution
    :return: error values
    """
    return vmap(
        lambda x_: truncated_normal(
            key=get_key(),
            lower=lower,
            upper=upper,
            shape=(size,),
            dtype=x.dtype,
        )
        * scale
        + x_
    )(x)


def uniform_error(x: Array, size: int, *, lower: float, upper: float) -> Array:
    r"""Add uniform error to the given values.

    .. math::
        x' = x+\mathcal{U}(a=lower, b=upper)

    :param x: given values
    :param size: number of samples
    :param scale:
    :param lower: _description_
    :param upper: _description_
    :return: _description_
    """
    return vmap(
        lambda x_: uniform(
            key=get_key(),
            shape=(size,),
            dtype=x.dtype,
            minval=lower,
            maxval=upper,
        )
        + x_
    )(x)


def banana_error(x: Array, size: int) -> Array:
    r"""Add banana error to the given values. Section 3 of the following paper
    https://doi.org/10.1093/mnras/stw2883 discusses the banana error.

    It adds errors in the chirp mass and symmetric mass ratio and then converts back to masses.

    .. math::
        \mathbf{M}_{c} = M_{c}^{T}\left[1+\alpha\frac{12}{\rho}\left(r_{0}+r\right)\right]
        \mathbf{\eta} = \eta^{T}\left[1+0.03\frac{12}{\rho}\left(r_{0}^{'}+r^{'}\right)\right]

    :param x: given values as m1 and m2
    :param size: number of samples
    :return: error values
    """
    m1 = x[0]
    m2 = x[1]

    r0 = normal(key=get_key(), shape=(), dtype=x.dtype)
    r0p = normal(key=get_key(), shape=(), dtype=x.dtype)
    r = normal(key=get_key(), shape=(size,), dtype=x.dtype)
    rp = normal(key=get_key(), shape=(size,), dtype=x.dtype)

    rho = 8.0 * jnp.power(
        uniform(
            key=get_key(),
            shape=(size,),
            dtype=x.dtype,
            minval=0.0,
            maxval=1.0,
        ),
        -1.0 / 3.0,
    )

    Mc_true = chirp_mass(m1, m2)
    eta_true = symmetric_mass_ratio(m1, m2)

    alpha = jnp.zeros_like(r)
    alpha = jnp.where(eta_true >= 0.1, 0.01, alpha)
    alpha = jnp.where((0.1 > eta_true) & (eta_true >= 0.05), 0.03, alpha)
    alpha = jnp.where(0.05 > eta_true, 0.1, alpha)

    twelve_over_rho = 12.0 / rho

    Mc = Mc_true * (1.0 + alpha * twelve_over_rho * (r0 + r))
    eta = eta_true * (1.0 + 0.03 * twelve_over_rho * (r0p + rp))

    mask = 0.25 >= eta
    mask &= eta >= 0.01

    mtot = Mc * jnp.power(eta, -0.6)
    m1m2 = eta * jnp.power(mtot, 2)

    m2_final = 0.5 * (mtot - jnp.sqrt(mtot**2 - 4 * m1m2))
    m1_final = 0.5 * (mtot + jnp.sqrt(mtot**2 - 4 * m1m2))

    m1_final = jnp.where(mask, m1_final, jnp.nan)
    m2_final = jnp.where(mask, m2_final, jnp.nan)

    return jnp.column_stack([m1_final, m2_final])
