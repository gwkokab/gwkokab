# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import numpy as np
import RIFT.lalsimutils as lalsimutils
from jax import numpy as jnp, random as jrd
from jaxtyping import PRNGKeyArray


__all__ = [
    "banana_error",
    "truncated_normal_error",
]


def banana_error(
    x: np.ndarray,
    size: int,
    key: PRNGKeyArray,
    *,
    scale_Mc: float = 1.0,
    scale_eta: float = 1.0,
) -> np.ndarray:
    r"""Add banana error to the given values. Section 3 of the `Model-independent
    inference on compact-binary observations <https://doi.org/10.1093/mnras/stw2883>`_
    discusses the banana error. It adds errors in the chirp mass and symmetric mass
    ratio and then converts back to masses.

    .. math::

        M_{c} = M_{c}^{T}
        \left[1+\beta\frac{12}{\rho}\left(r_{0}+r\right)\right]

        \eta = \eta^{T}
        \left[1+0.03\frac{12}{\rho}\left(r_{0}^{'}+r^{'}\right)\right]

    x : np.ndarray
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
    np.ndarray
        array of values with added banana error
    """
    keys = jrd.split(key, 5)

    r0 = np.asarray(jrd.normal(key=keys[0]))
    r0p = np.asarray(jrd.normal(key=keys[1]))
    r = np.asarray(jrd.normal(key=keys[2], shape=(size,))) * scale_Mc
    rp = np.asarray(jrd.normal(key=keys[3], shape=(size,))) * scale_eta
    rho = 9.0 * np.power(np.asarray(jrd.uniform(key=keys[4])), -1.0 / 3.0)

    Mc_true, eta_true = np.unstack(x, axis=-1)

    v_PN_param = (np.pi * Mc_true * 20 * lalsimutils.MsunInSec) ** (
        1.0 / 3.0
    )  # 'v' parameter
    v_PN_param_max = 0.2
    v_PN_param = np.minimum(v_PN_param, v_PN_param_max)
    snr_fac = rho / 12.0
    # this ignores range due to redshift / distance, based on a low-order est
    ln_mc_error_pseudo_fisher = (
        1.5 * 0.3 * (v_PN_param / v_PN_param_max) ** (7.0) / snr_fac
    )

    beta = np.min(np.array([0.07 / snr_fac, ln_mc_error_pseudo_fisher]))

    Mc = Mc_true * (1.0 + beta * (r0 + r))
    eta = eta_true * (1.0 + 0.03 * (12.0 / rho) * (r0p + rp))

    Mc = np.where(Mc <= 0.0, np.nan, Mc)
    eta = np.where((eta <= 0.25) & (eta >= 0.0), eta, np.nan)

    return np.stack((Mc, eta), axis=-1)


def truncated_normal_error(
    x: np.ndarray,
    size: int,
    key: PRNGKeyArray,
    *,
    scale: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> np.ndarray:
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
    x : np.ndarray
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
    np.ndarray
        Array of values with added truncated normal error, with all values within the
        specified bounds.
    """
    key_r0, key_r, key_rho = jrd.split(key, 3)

    r0 = np.asarray(jrd.normal(key=key_r0))
    r = np.asarray(jrd.normal(key=key_r, shape=(size,)))
    rho = 9.0 * np.power(
        np.asarray(
            jrd.uniform(key=key_rho, minval=jnp.finfo(jnp.result_type(float)).tiny)
        ),
        -1.0 / 3.0,
    )

    samples = x + scale * (r0 + r) * (12.0 / rho)

    # reflect samples that are out of bounds back into the allowed range
    if low is not None and high is not None:
        samples = low + np.mod(samples - low, 2 * (high - low))
        samples = np.where(samples > high, 2.0 * high - samples, samples)
    elif low is not None:
        samples = np.where(samples < low, 2.0 * low - samples, samples)
    elif high is not None:
        samples = np.where(samples > high, 2.0 * high - samples, samples)

    return samples
