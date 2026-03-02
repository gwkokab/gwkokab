# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import RIFT.lalsimutils as lalsimutils
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray

from .utils.transformations import chirp_mass, symmetric_mass_ratio


__all__ = [
    "banana_error_m1_m2",
    "truncated_normal_error",
]


def banana_error_m1_m2(
    x: Array,
    size: int,
    key: PRNGKeyArray,
    *,
    scale_Mc: float = 1.0,
    scale_eta: float = 1.0,
) -> Array:
    r"""Add banana error to the given values. Section 3 of the `Model-independent
    inference on compact-binary observations <https://doi.org/10.1093/mnras/stw2883>`_
    discusses the banana error. It adds errors in the chirp mass and symmetric mass
    ratio and then converts back to masses.

    .. math::

        M_{c} = M_{c}^{T}
        \left[1+\beta\frac{12}{\rho}\left(r_{0}+r\right)\right]

        \eta = \eta^{T}
        \left[1+0.03\frac{12}{\rho}\left(r_{0}^{'}+r^{'}\right)\right]

    x : Array
        given values as m1 and m2
    size : int
        number of samples
    key : PRNGKeyArray
        jax random key
    scale_Mc : float
        scale of the chirp mass error, defaults to 1.0
    scale_eta : float
        scale of the symmetric mass ratio error, defaults to 1.0

    Returns
    -------
    Array
        m1 and m2 with banana
    """
    m1 = x[..., 0]
    m2 = x[..., 1]

    keys = jrd.split(key, 5)

    r0 = jrd.normal(key=keys[0])
    r0p = jrd.normal(key=keys[1])
    r = jrd.normal(key=keys[2], shape=(size,)) * scale_Mc
    rp = jrd.normal(key=keys[3], shape=(size,)) * scale_eta
    rho = 9.0 * jnp.power(jrd.uniform(key=keys[4]), -1.0 / 3.0)

    Mc_true = chirp_mass(m1=m1, m2=m2)
    eta_true = symmetric_mass_ratio(m1=m1, m2=m2)

    v_PN_param = (jnp.pi * Mc_true * 20 * lalsimutils.MsunInSec) ** (
        1.0 / 3.0
    )  # 'v' parameter
    v_PN_param_max = 0.2
    v_PN_param = jnp.min(jnp.array([v_PN_param, v_PN_param_max]))
    snr_fac = rho / 12.0
    # this ignores range due to redshift / distance, based on a low-order est
    ln_mc_error_pseudo_fisher = (
        1.5 * 0.3 * (v_PN_param / v_PN_param_max) ** (7.0) / snr_fac
    )

    beta = jnp.min(jnp.array([0.07 / snr_fac, ln_mc_error_pseudo_fisher]))

    Mc = Mc_true * (1.0 + beta * (r0 + r))
    eta = eta_true * (1.0 + 0.03 * (12.0 / rho) * (r0p + rp))

    etaV = 1.0 - 4.0 * eta
    etaV_sqrt = jnp.where(etaV >= 0, jnp.sqrt(etaV), jnp.nan)

    factor = 0.5 * Mc * jnp.power(eta, -0.6)
    m1_final = factor * (1.0 + etaV_sqrt)
    m2_final = factor * (1.0 - etaV_sqrt)

    return jnp.column_stack([m1_final, m2_final])


def truncated_normal_error(
    x: Array,
    size: int,
    key: PRNGKeyArray,
    *,
    scale: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> Array:
    """Adds truncated normal error to the given values. The error is sampled from a
    truncated normal distribution with the given parameters. The function will resample
    until all values are within the allowed range.

    .. note::

        If :code:`cut_low` and :code:`cut_high` are both None, the function will return
        the sampled values without any truncation.

    .. note::

        if :code:`low` or :code:`high` is None, the :code:`TruncatedNormal` distribution
        will not be truncated at those bounds.

    Parameters
    ----------
    x : Array
        Given values to which the error will be added.
    size : int
        Number of samples to generate.
    key : PRNGKeyArray
        JAX random key for sampling.
    scale : float
        Scale parameter for the truncated normal distribution.
    low : Optional[float], optional
        Lower bound for the truncation, defaults to None (no lower bound).
    high : Optional[float], optional
        Upper bound for the truncation, defaults to None (no upper bound).

    Returns
    -------
    Array
        Array of values with added truncated normal error.
    """
    key_r0, key_r, key_rho = jrd.split(key, 3)

    r0 = jrd.normal(key=key_r0)
    r = jrd.normal(key=key_r, shape=(size,))
    rho = 9.0 * jnp.power(
        jrd.uniform(key=key_rho, minval=jnp.finfo(jnp.result_type(float)).tiny),
        -1.0 / 3.0,
    )

    samples = x + scale * (r0 + r) * (12.0 / rho)

    # reflect samples that are out of bounds back into the allowed range
    if low is not None:
        samples = jnp.where(samples < low, 2.0 * low - samples, samples)
    if high is not None:
        samples = jnp.where(samples > high, 2.0 * high - samples, samples)

    return samples
