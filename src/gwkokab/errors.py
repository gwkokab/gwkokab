# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import RIFT.lalsimutils as lalsimutils
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro import distributions as dist

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
    scale: Optional[float] = None,
    low: Optional[float] = None,
    high: Optional[float] = None,
    cut_low: Optional[float] = None,
    cut_high: Optional[float] = None,
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
    scale : Optional[float], optional
        Scale parameter for the truncated normal distribution, by default None
    low : Optional[float], optional
        Lower bound for the truncated normal distribution, by default None
    high : Optional[float], optional
        Upper bound for the truncated normal distribution, by default None
    cut_low : Optional[float], optional
        Lower bound for the final values, by default None
    cut_high : Optional[float], optional
        Upper bound for the final values, by default None

    Returns
    -------
    Array
        Array of values with added truncated normal error.

    Raises
    ------
    ValueError
        If the scale parameter is not provided.
    """
    if scale is None:
        raise ValueError("Scale parameter is required.")

    err_dist = dist.TruncatedNormal(loc=x, scale=scale, low=low, high=high)

    # Initial sampling from the truncated normal distribution.
    err_x = err_dist.sample(key=key, sample_shape=(size,))

    if cut_low is None and cut_high is None:
        return err_x

    # Resample until all values are within the allowed range
    while True:
        mask = jnp.zeros_like(err_x, dtype=bool)
        if cut_low is not None:
            mask = mask | (err_x < cut_low)
        if cut_high is not None:
            mask = mask | (err_x > cut_high)

        # If no values are outside the allowed range, break the loop
        # This is important to avoid infinite loops
        if not jnp.any(mask).item():
            break

        # Split the key for a new random seed
        key, _ = jrd.split(key)
        num_invalid = int(jnp.sum(mask))
        new_samples = err_dist.sample(key=key, sample_shape=(num_invalid,))
        invalid_indices = jnp.where(mask)[0]
        err_x = err_x.at[invalid_indices].set(new_samples)

    return err_x
