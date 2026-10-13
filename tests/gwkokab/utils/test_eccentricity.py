# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical tests of the global eccentricity-transport ansatz."""

import numpy as np
import pytest

from gwkokab.utils.eccentricity import evolve_eccentricity
from gwkokab.utils.exceptions import LoggedValueError


def log_g(e):
    return (
        12 / 19 * np.log(e)
        - np.log1p(-(e**2))
        + 870 / 2299 * np.log1p(121 / 304 * e**2)
    )


@pytest.mark.parametrize("method", ["peters", "small-e"])
def test_identity_circular_redshift_and_roundtrip(method):
    e = np.array([0.0, 1e-100, 1e-6, 0.1, 0.4])
    identity, jac = evolve_eccentricity(e, 1.0, method)
    np.testing.assert_allclose(identity, e, rtol=1e-12, atol=0)
    np.testing.assert_allclose(jac, 0.0, atol=1e-12)
    ratio = np.array([0.1, 0.5, 2.0, 0.5, 2.0])
    out, log_jac = evolve_eccentricity(e, ratio, method)
    back, inverse_jac = evolve_eccentricity(out, 1.0 / ratio, method)
    np.testing.assert_allclose(back, e, rtol=1e-12, atol=0)
    np.testing.assert_allclose(log_jac + inverse_jac, 0, atol=1e-12)
    # 20Hz detector at z=1 -> 10Hz source, ratio 1/4 (not 1/2).
    out, log_jac = evolve_eccentricity(1e-7, 10 / (20 * (1 + 1)), method)
    np.testing.assert_allclose(out, 1e-7 * 4 ** (19 / 18), rtol=1e-10)
    np.testing.assert_allclose(log_jac, -(19 / 18) * np.log(4), atol=1e-10)


def test_full_peters_relation_and_extremely_low_e():
    e = np.array([1e-300, 1e-100, 1e-10, 0.1, 0.5, 0.9])
    ratio = np.array([0.1, 5.0, 0.5, 10.0, 0.2, 2.0])
    out, jac = evolve_eccentricity(e, ratio)
    np.testing.assert_allclose(
        log_g(out) - log_g(e), -2 / 3 * np.log(ratio), atol=1e-12
    )
    assert np.all((out > 0) & (out < 1))
    assert np.all(np.isfinite(jac))
    circular, circular_jac = evolve_eccentricity(np.zeros(2), [1e-300, 1e300])
    np.testing.assert_array_equal(circular, 0)
    np.testing.assert_allclose(circular_jac, 19 / 18 * np.log([1e-300, 1e300]))


@pytest.mark.parametrize("method", ["peters", "small-e"])
def test_inverse_jacobian_finite_difference(method):
    e = np.array([0.01, 0.1, 0.4])
    ratio = np.array([0.4, 2.0, 1.5])
    h = 1e-7
    _, log_jac = evolve_eccentricity(e, ratio, method)
    plus, _ = evolve_eccentricity(e + h, ratio, method)
    minus, _ = evolve_eccentricity(e - h, ratio, method)
    np.testing.assert_allclose(np.exp(-log_jac), (plus - minus) / (2 * h), rtol=1e-6)


@pytest.mark.parametrize("method", ["peters", "small-e"])
def test_transformed_uniform_prior_normalizes(method):
    e = np.linspace(0.0, 0.5, 10001)
    out, jac = evolve_eccentricity(e, 0.8, method)
    assert np.trapezoid(np.exp(jac) / 0.5, out) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize(
    ("e", "ratio"),
    [(np.nan, 1), (-0.1, 1), (1.0, 1), (0.1, 0), (0.1, np.inf)],
)
def test_invalid_coordinates_are_rejected(e, ratio):
    with pytest.raises(LoggedValueError):
        evolve_eccentricity(e, ratio)


def test_small_e_failure_is_not_clipped_and_peters_is_backstop():
    with pytest.raises(LoggedValueError, match="Peters backstop"):
        evolve_eccentricity(0.5, 0.1, "small-e")
    out, _ = evolve_eccentricity(0.5, 0.1, "peters")
    assert 0.5 < out < 1
    with pytest.raises(LoggedValueError, match="Unknown"):
        evolve_eccentricity(0.1, 1, "unknown")


def test_near_unity_solution_with_inaccurate_relation_is_rejected():
    with pytest.raises(LoggedValueError, match="resolution"):
        evolve_eccentricity(0.9, 1e-20)


def test_small_e_extreme_ratio_does_not_overflow_intermediate_multiplier():
    out, jac = evolve_eccentricity(1e-320, 1e-300, "small-e")
    expected_log = np.log(1e-320) - 19 / 18 * np.log(1e-300)
    assert np.isfinite(out) and 0 < out < 1
    assert np.log(out) == pytest.approx(expected_log, abs=1e-12)
    assert jac == pytest.approx(19 / 18 * np.log(1e-300))
