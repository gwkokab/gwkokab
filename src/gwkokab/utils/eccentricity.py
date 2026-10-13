# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Approximate quadrupole transport of model-native eccentricity.

This is a quick global ansatz, not a waveform-based eccentricity conversion.
Frequencies are quadrupole GW frequencies in the same frame. See Peters,
Phys. Rev. 136, B1224 (1964), ADS key ``1964PhRv..136.1224P``.
These NumPy routines run while loading data, outside the JAX likelihood.
"""

from typing import Literal

import numpy as np

from .exceptions import LoggedValueError


def _log_peters_g(log_e: np.ndarray) -> np.ndarray:
    e2 = np.exp(2.0 * log_e)
    return (
        (12.0 / 19.0) * log_e
        - np.log1p(-e2)
        + (870.0 / 2299.0) * np.log1p((121.0 / 304.0) * e2)
    )


def _log_peters_slope(log_e: np.ndarray) -> np.ndarray:
    """Log of d log(g) / d log(e), finite in the circular limit."""
    e2 = np.exp(2.0 * log_e)
    return np.log(
        12.0 / 19.0
        + 2.0 * e2 / (1.0 - e2)
        + (870.0 / 2299.0) * (242.0 / 304.0) * e2 / (1.0 + (121.0 / 304.0) * e2)
    )


def evolve_eccentricity(
    eccentricity,
    frequency_ratio,
    method: Literal["peters", "small-e"] = "peters",
) -> tuple[np.ndarray, np.ndarray]:
    """Transport eccentricity and return the log inverse Jacobian.

    Parameters
    ----------
    eccentricity : array_like
        Native bound-orbit eccentricities, 0 <= e < 1.
    frequency_ratio : array_like
        Target/native quadrupole frequency ratio in a common frame. Broadcast
        against ``eccentricity``. For source frequency f* and detector native
        frequency f0, this is f* / ((1 + z) f0).
    method : {"peters", "small-e"}
        Full orbit-averaged quadrupole inverse or its small-e power law.

    Returns
    -------
    eccentricity_reference, log_inverse_jacobian : numpy.ndarray
        Transported eccentricity and log |de_native/de_reference| at fixed z.
        This Jacobian belongs in the PE reference prior, not sample weights.

    Raises
    ------
    LoggedValueError
        Inputs are invalid, or small-e evolution leaves bound-orbit support.
        Values are never clipped or silently discarded.
    """
    e, ratio = np.broadcast_arrays(
        np.asarray(eccentricity, dtype=float),
        np.asarray(frequency_ratio, dtype=float),
    )
    if np.any(~np.isfinite(e)) or np.any((e < 0) | (e >= 1)):
        raise LoggedValueError("Native eccentricity must be finite with 0 <= e < 1.")
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0):
        raise LoggedValueError("Frequency ratios must be finite and positive.")
    log_ratio = np.log(ratio)
    if method == "small-e":
        log_jacobian = (19.0 / 18.0) * log_ratio
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.where(
                e == 0,
                0.0,
                np.exp(np.log(np.where(e == 0, 1.0, e)) - log_jacobian),
            )
        if np.any(~np.isfinite(result)) or np.any(result >= 1):
            raise LoggedValueError(
                "Small-e transport produces e >= 1; use the Peters backstop."
            )
        if np.any((e > 0) & (result == 0)):
            raise LoggedValueError(
                "Eccentricity transport underflows; change reference."
            )
        return result, log_jacobian
    if method != "peters":
        raise LoggedValueError("Unknown eccentricity evolution method: {}", method)

    # Solve in log(e), retaining precision even for extremely small e.
    circular = e == 0
    log_e = np.log(np.where(circular, 1.0e-300, e))
    target = _log_peters_g(log_e) - (2.0 / 3.0) * log_ratio
    lower = np.minimum(log_e, (19.0 / 12.0) * target) - 2.0
    upper = np.full(e.shape, np.log(np.nextafter(1.0, 0.0)))
    if np.any((~circular) & (target > _log_peters_g(upper))):
        raise LoggedValueError("Peters solution is too close to unity to resolve.")
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        below = _log_peters_g(midpoint) < target
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    log_result = (lower + upper) / 2.0
    result = np.where(circular, 0.0, np.exp(log_result))
    log_jacobian = np.where(
        circular,
        (19.0 / 18.0) * log_ratio,
        _log_peters_slope(log_result) - _log_peters_slope(log_e) + log_e - log_result,
    )
    if np.any((~circular) & (result == 0)) or np.any(result >= 1):
        raise LoggedValueError("Peters solution is outside floating-point resolution.")
    # Check the returned floating-point value, not just the bisection interval.
    # Near unity (or subnormal zero) rounding e can materially change g and J.
    returned_log_e = np.log(np.where(circular, 1.0e-300, result))
    residual = np.abs(_log_peters_g(returned_log_e) - target)
    tolerance = 1.0e-10 * np.maximum(1.0, np.abs(target))
    if np.any((~circular) & (residual > tolerance)):
        raise LoggedValueError("Peters solution exceeds floating-point resolution.")
    return result, log_jacobian
