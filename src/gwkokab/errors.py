# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import numpy as np
import RIFT.lalsimutils as lalsimutils
from jax import random as jrd
from jaxtyping import PRNGKeyArray
from numpyro.distributions.truncated import TruncatedNormal

from gwkokab.parameters import Parameters as P
from gwkokab.utils.exceptions import LoggedValueError


__all__ = [
    "banana_error",
    "mock_spin_error",
    "truncated_normal_error",
]


def banana_error(
    Mc_true: np.ndarray,
    eta_true: np.ndarray,
    size: int,
    key: PRNGKeyArray,
    *,
    estimates: dict[str | P, np.ndarray],
    rho: np.ndarray,
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

    Mc_true : np.ndarray
        True chirp mass
    eta_true : np.ndarray
        True symmetric mass ratio
    size : int
        number of samples
    key : PRNGKeyArray
        jax random key
    scale_Mc : float
        scale of the chirp mass error, defaults to 1.0
    scale_eta : float
        scale of the symmetric mass ratio error, defaults to 1.0
    estimates : dict[str | P, np.ndarray]
        Parameter estimates performed so far
    rho : np.ndarray
        SNR of the event, used to scale the error

    Returns
    -------
    np.ndarray
        array of values with added banana error
    """
    r0_key, r0p_key, r_key, rp_key = jrd.split(key, 4)

    r0 = np.asarray(jrd.normal(key=r0_key))
    r0p = np.asarray(jrd.normal(key=r0p_key))
    r = np.asarray(jrd.normal(key=r_key, shape=(size,))) * scale_Mc
    rp = np.asarray(jrd.normal(key=rp_key, shape=(size,))) * scale_eta

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

    beta = np.minimum(0.07 / snr_fac, ln_mc_error_pseudo_fisher)

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
    estimates: dict[str | P, np.ndarray],
    rho: np.ndarray,
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
    estimates : dict[str | P, np.ndarray]
        Parameter estimates performed so far
    rho : np.ndarray
        SNR of the event, used to scale the error
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
    key_r0, key_r = jrd.split(key)

    r0 = np.asarray(jrd.normal(key=key_r0))
    r = np.asarray(jrd.normal(key=key_r, shape=(size,)))

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


# Copyright 2023 Amanda Farah
# SPDX-License-Identifier: CC0-1.0
def dpsi_from_dXeff_neglect_Xa(dXeff, n):
    """Returns calculation of delta psi, which is a function of n (eta) and delta Xeff
    obtained by rearranging eq A2 of arxiv:1805.03046 (Ng et al.

    2018), neglecting chi_a
    psi: float or array-like of floats, the 1.5 PN phase term coefficient
    n: float or array-like of floats, the symmetric mass ratio
    """
    A = np.power(n, -3 / 5)
    B = 113 - (76 * n)
    C = 128
    return A * (B / C) * dXeff


# Copyright 2023 Amanda Farah
# SPDX-License-Identifier: CC0-1.0
def psi_from_chi_eff_and_eta_neglect_chi_a(chi_eff, n):
    """Returns psi coefficient, neglecting chi_a term."""
    return np.power(n, -3 / 5) * (
        (((113 - (76 * n)) * chi_eff) / 128) - (3 * np.pi / 8)
    )


# Copyright 2023 Amanda Farah
# SPDX-License-Identifier: CC0-1.0
def chi_eff_from_psi_and_eta_neglect_chi_a(psi, n):
    """Returns calculation of chi_eff from psi, which is a function of n (eta) obtained
    by rearranging eq A2 of arxiv:1805.03046 (Ng et al.

    2018), assuming chi_2 = 0
    psi: float or array-like of floats, the 1.5 PN phase term coefficient
    n: float or array-like of floats, the symmetric mass ratio
    """
    A = 128
    B = (psi * np.power(n, 3 / 5)) + (3 * np.pi / 8)
    C = 113 - (76 * n)
    return A * (B / C)


# This is a refactored implementation of https://git.ligo.org/amanda.farah/GWMockCat/-/blob/main/GWMockCat/posterior_utils.py?ref_type=heads#L61
#
# Copyright 2023 Amanda Farah
# SPDX-License-Identifier: CC0-1.0
def mock_spin_error(
    chi_eff: np.ndarray,
    eta: np.ndarray,
    size: int,
    key: PRNGKeyArray,
    *,
    estimates: dict[str | P, np.ndarray],
    rho: np.ndarray,
    scale_chi_eff: np.ndarray,
) -> np.ndarray:
    if (etaobs := estimates.get(P.SYMMETRIC_MASS_RATIO, None)) is None:
        raise LoggedValueError(
            "Parameter estimation of Symmetric Mass Ratio is not available."
        )
    threshold_snr = 12.0
    uncert_psi = dpsi_from_dXeff_neglect_Xa(scale_chi_eff, eta)
    psi = psi_from_chi_eff_and_eta_neglect_chi_a(chi_eff, eta)
    spsi = threshold_snr / rho * uncert_psi
    psiobs = TruncatedNormal(
        loc=psi,
        scale=spsi,
        low=-4.2,
        high=-1.2,
        validate_args=True,
    ).sample(key, (size,))
    Xeffobs = chi_eff_from_psi_and_eta_neglect_chi_a(psiobs, etaobs)
    return Xeffobs
