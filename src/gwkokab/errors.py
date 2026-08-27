# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Measurement-error models for synthesising mock parameter estimation samples.

The synthetic-data CLIs draw a true population with
``synthetic_events_<family>`` and then blur each true event into a cloud of mock PE
samples using the functions here. Every error model shares the same call signature --
the true value(s), the number of samples to draw, a PRNG key, the ``estimates``
dictionary of parameters blurred so far, and the event SNR ``rho`` -- so that
:attr:`~gwkokab.analysis.core.synthetic_pe.SyntheticDiscretePE.error_function_registry` can wire them to
parameters by name. Errors scale as :math:`1/\rho`, so louder events get tighter
posteriors.

The ``psi``/``chi_eff`` helpers and :func:`mock_spin_error` are refactored from
`GWMockCat <https://git.ligo.org/amanda.farah/GWMockCat>`_ and carry their own
copyright and CC0 licence notice.
"""

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
    r"""Add banana error to the given values.

    Section 3 of `Model-independent inference on compact-binary observations
    <https://doi.org/10.1093/mnras/stw2883>`_ discusses the banana error. It adds errors
    in the chirp mass and symmetric mass ratio and then converts back to masses.

    .. math::

        M_{c} = M_{c}^{T}
        \left[1+\beta\frac{12}{\rho}\left(r_{0}+r\right)\right]

        \eta = \eta^{T}
        \left[1+0.03\frac{12}{\rho}\left(r_{0}^{'}+r^{'}\right)\right]

    Parameters
    ----------
    Mc_true : np.ndarray
        True chirp mass.
    eta_true : np.ndarray
        True symmetric mass ratio.
    size : int
        Number of samples to draw.
    key : PRNGKeyArray
        JAX random key.
    estimates : dict[str | P, np.ndarray]
        Parameter estimates performed so far. Unused here, but part of the common error
        model signature.
    rho : np.ndarray
        SNR of the event, used to scale the error.
    scale_Mc : float
        Scale of the chirp mass error. Defaults to ``1.0``.
    scale_eta : float
        Scale of the symmetric mass ratio error. Defaults to ``1.0``.

    Returns
    -------
    np.ndarray
        Array of shape ``(size, 2)`` holding the blurred
        :math:`(M_c, \eta)` pairs. Samples that leave the physical region
        (:math:`M_c \leq 0`, or :math:`\eta` outside :math:`[0, 0.25]`) are set to NaN
        rather than resampled, and are dropped downstream.
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
    r"""Add normal error to the given values, reflected back into the allowed range.

    The error has a per-event offset ``r0`` shared by all samples plus a per-sample
    term ``r``, both scaled by ``scale`` and by :math:`12/\rho`. Samples that fall
    outside ``[low, high]`` are reflected back in rather than resampled, which keeps the
    returned array exactly ``size`` long and avoids an unbounded rejection loop.

    .. note::

        If ``low`` and ``high`` are both :data:`None` the samples are returned
        untruncated. If only one is given, reflection is applied at that bound only.

    Parameters
    ----------
    x : np.ndarray
        Given values to which the error will be added.
    size : int
        Number of samples to generate.
    key : PRNGKeyArray
        JAX random key for sampling.
    scale : float
        Scale of the error, before the :math:`12/\rho` SNR factor.
    estimates : dict[str | P, np.ndarray]
        Parameter estimates performed so far. Unused here, but part of the common error
        model signature.
    rho : np.ndarray
        SNR of the event, used to scale the error.
    low : Optional[float], optional
        Lower bound for the reflection. Defaults to :data:`None` (no lower bound).
    high : Optional[float], optional
        Upper bound for the reflection. Defaults to :data:`None` (no upper bound).

    Returns
    -------
    np.ndarray
        Array of ``size`` values with added error, all within the specified bounds.
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
    r"""Propagate an uncertainty in :math:`\chi_{\text{eff}}` into one in :math:`\psi`.

    Obtained by rearranging Eq. (A2) of `arXiv:1805.03046
    <https://arxiv.org/abs/1805.03046>`_ (Ng et al. 2018), neglecting :math:`\chi_a`:

    .. math::
        \delta\psi = \eta^{-3/5}\,\frac{113 - 76\eta}{128}\,\delta\chi_{\text{eff}}

    Parameters
    ----------
    dXeff : ArrayLike
        Uncertainty in the effective spin :math:`\chi_{\text{eff}}`.
    n : ArrayLike
        Symmetric mass ratio :math:`\eta`.

    Returns
    -------
    np.ndarray
        The corresponding uncertainty in the 1.5 PN phase coefficient :math:`\psi`.
    """
    A = np.power(n, -3 / 5)
    B = 113 - (76 * n)
    C = 128
    return A * (B / C) * dXeff


# Copyright 2023 Amanda Farah
# SPDX-License-Identifier: CC0-1.0
def psi_from_chi_eff_and_eta_neglect_chi_a(chi_eff, n):
    r"""Compute the 1.5 PN phase coefficient, neglecting the :math:`\chi_a` term.

    .. math::
        \psi = \eta^{-3/5}\left(\frac{(113 - 76\eta)\chi_{\text{eff}}}{128}
               - \frac{3\pi}{8}\right)

    Parameters
    ----------
    chi_eff : ArrayLike
        Effective spin :math:`\chi_{\text{eff}}`.
    n : ArrayLike
        Symmetric mass ratio :math:`\eta`.

    Returns
    -------
    np.ndarray
        The 1.5 PN phase term coefficient :math:`\psi`.
    """
    return np.power(n, -3 / 5) * (
        (((113 - (76 * n)) * chi_eff) / 128) - (3 * np.pi / 8)
    )


# Copyright 2023 Amanda Farah
# SPDX-License-Identifier: CC0-1.0
def chi_eff_from_psi_and_eta_neglect_chi_a(psi, n):
    r"""Recover :math:`\chi_{\text{eff}}` from the 1.5 PN phase coefficient.

    Obtained by rearranging Eq. (A2) of `arXiv:1805.03046
    <https://arxiv.org/abs/1805.03046>`_ (Ng et al. 2018), assuming
    :math:`\chi_2 = 0`:

    .. math::
        \chi_{\text{eff}} = \frac{128}{113 - 76\eta}
        \left(\psi\eta^{3/5} + \frac{3\pi}{8}\right)

    Parameters
    ----------
    psi : ArrayLike
        The 1.5 PN phase term coefficient :math:`\psi`.
    n : ArrayLike
        Symmetric mass ratio :math:`\eta`.

    Returns
    -------
    np.ndarray
        The effective spin :math:`\chi_{\text{eff}}`.
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
    r"""Blur an effective spin by propagating the error through the 1.5 PN phase.

    :math:`\chi_{\text{eff}}` is not measured directly: what the waveform constrains is
    the 1.5 PN phase coefficient :math:`\psi`. This model therefore maps the true
    :math:`\chi_{\text{eff}}` to :math:`\psi`, draws from a truncated normal on
    :math:`\psi` whose width scales as :math:`12/\rho`, and maps back -- using the
    *already blurred* symmetric mass ratio from ``estimates`` for the inverse map, so the
    resulting :math:`\chi_{\text{eff}}`--:math:`\eta` posterior carries the correct
    correlation.

    Parameters
    ----------
    chi_eff : np.ndarray
        True effective spin.
    eta : np.ndarray
        True symmetric mass ratio.
    size : int
        Number of samples to generate.
    key : PRNGKeyArray
        JAX random key for sampling.
    estimates : dict[str | P, np.ndarray]
        Parameter estimates performed so far. Must already contain
        :attr:`~gwkokab.parameters.Parameters.SYMMETRIC_MASS_RATIO`, so the mass error
        model has to run before this one.
    rho : np.ndarray
        SNR of the event, used to scale the error.
    scale_chi_eff : np.ndarray
        Uncertainty in :math:`\chi_{\text{eff}}` at the reference SNR of 12.

    Returns
    -------
    np.ndarray
        Array of ``size`` blurred effective spin values.

    Raises
    ------
    LoggedValueError
        If ``estimates`` does not yet hold the symmetric mass ratio.

    Notes
    -----
    The truncation of :math:`\psi` to :math:`[-4.2, -1.2]` is inherited from
    `GWMockCat <https://git.ligo.org/amanda.farah/GWMockCat>`_.
    """
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
