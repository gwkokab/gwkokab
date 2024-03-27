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

from typing_extensions import Optional

import RIFT.lalsimutils as lalsimutils
from jax import numpy as jnp, vmap
from jax.random import normal, uniform
from jaxtyping import Array
from numpyro import distributions as dist

from ..utils import chirp_mass, get_key, symmetric_mass_ratio


def error_factory(error_type: str, key: Optional[int | Array] = None, **kwargs) -> Array:
    """Factory function to create different types of errors.

    :param error_type: name of the error
    :raises ValueError: if the error type is unknown
    :return: error values for the given error type
    """
    if key is None or isinstance(key, int):
        key = get_key(key)

    if error_type == "normal":
        return normal_error(key=key, **kwargs)
    elif error_type == "truncated_normal":
        return truncated_normal_error(key=key, **kwargs)
    elif error_type == "uniform":
        return uniform_error(key=key, **kwargs)
    elif error_type == "banana":
        return banana_error(key=key, **kwargs)
    else:
        raise ValueError(f"Unknown error type: {error_type}")


def normal_error(x: Array, size: int, key: Array, *, scale: float) -> Array:
    r"""Add normal error to the given values.

    .. math::
        x' \sim \mathcal{N}(\mu=x, \sigma=\text{scale})

    :param x: given values
    :param size: number of samples
    :param scale: standard deviation of the normal distribution
    :return: error values
    """
    return vmap(
        lambda x_: normal(
            key=key,
            shape=(size,),
            dtype=x.dtype,
        )
        * scale
        + x_
    )(x)


def truncated_normal_error(
    x: Array,
    size: int,
    key: Array,
    *,
    scale: float,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> Array:
    r"""Add truncated normal error to the given values.

    .. math::
        x' \sim \mathcal{N}(\mu=x, \sigma=\text{scale}) \cap [lower, upper]

    :param x: given values
    :param size: number of samples
    :param scale: standard deviation of the normal distribution
    :param lower: lower bound of the truncated normal distribution
    :param upper: upper bound of the truncated normal distribution
    :return: error values
    """
    return vmap(
        lambda x_: dist.TruncatedNormal(
            loc=x_,
            scale=scale,
            low=lower,
            high=upper,
        ).sample(key=key, sample_shape=(size,))
    )(x)


def uniform_error(x: Array, size: int, key: Array, *, lower: float, upper: float) -> Array:
    r"""Add uniform error to the given values.

    .. math::
        x' \sim x+\mathcal{U}(a=lower, b=upper)

    :param x: given values
    :param size: number of samples
    :param scale:
    :param lower: _description_
    :param upper: _description_
    :return: _description_
    """
    return vmap(lambda x_: dist.Uniform(low=lower, high=upper).sample(key=key, sample_shape=(size,)) + x_)(x)


def banana_error(x: Array, size: int, key: Array, *, scale_Mc: float = 1.0, scale_eta: float = 1.0) -> Array:
    r"""Add banana error to the given values. Section 3 of the
    `paper <https://doi.org/10.1093/mnras/stw2883>`__ discusses the banana
    error. It adds errors in the chirp mass and symmetric mass ratio and then
    converts back to masses.

    .. math::
        M_{c} = M_{c}^{T}\left[1+\alpha\frac{12}{\rho}\left(r_{0}+r\right)\right]

        \eta = \eta^{T}\left[1+0.03\frac{12}{\rho}\left(r_{0}^{'}+r^{'}\right)\right]

    :param x: given values as m1 and m2
    :param size: number of samples
    :return: error values
    """
    m1 = x[..., 0]
    m2 = x[..., 1]

    key = get_key(key)
    r0 = normal(key=key)

    key = get_key(key)
    r0p = normal(key=key)

    key = get_key(key)
    r = normal(key=key, shape=(size,)) * scale_Mc

    key = get_key(key)
    rp = normal(key=key, shape=(size,)) * scale_eta

    key = get_key(key)
    rho = 9.0 * jnp.power(uniform(key=key), -1.0 / 3.0)

    Mc_true = chirp_mass(m1, m2)
    eta_true = symmetric_mass_ratio(m1, m2)

    v_PN_param = (jnp.pi * Mc_true * 20 * lalsimutils.MsunInSec) ** (1.0 / 3.0)  # 'v' parameter
    v_PN_param_max = 0.2
    v_PN_param = jnp.min(jnp.array([v_PN_param, v_PN_param_max]))
    snr_fac = rho / 15.0
    # this ignores range due to redshift / distance, based on a low-order est
    ln_mc_error_pseudo_fisher = 1.5 * 0.3 * (v_PN_param / v_PN_param_max) ** (7.0) / snr_fac

    alpha = jnp.min(jnp.array([0.07 / snr_fac, ln_mc_error_pseudo_fisher]))

    Mc = Mc_true * (1.0 + alpha * (r0 + r))
    eta = eta_true * (1.0 + (0.36 / rho) * (r0p + rp))

    etaV = 1.0 - 4.0 * eta
    etaV_sqrt = jnp.where(etaV >= 0, jnp.sqrt(etaV), jnp.nan)

    factor = 0.5 * Mc * jnp.power(eta, -0.6)
    m1_final = factor * (1.0 + etaV_sqrt)
    m2_final = factor * (1.0 - etaV_sqrt)

    return jnp.column_stack([m1_final, m2_final])
