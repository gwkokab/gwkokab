# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the measurement-error models used to fake parameter estimation.

Each of these takes true event values and returns *posterior samples* for one event, so
the properties worth pinning down are structural rather than numerical: the shape of
what comes back, the support it respects, that a bigger SNR means a tighter posterior,
and that the same key reproduces the same draw.
"""

import numpy as np
import pytest
from jax import random as jrd

from gwkokab.errors import (
    banana_error,
    chi_eff_from_psi_and_eta_neglect_chi_a,
    dpsi_from_dXeff_neglect_Xa,
    mock_spin_error,
    psi_from_chi_eff_and_eta_neglect_chi_a,
    truncated_normal_error,
)
from gwkokab.parameters import Parameters as P
from gwkokab.utils.exceptions import LoggedValueError


SIZE = 512


@pytest.fixture
def key():
    return jrd.PRNGKey(0)


class TestTruncatedNormalError:
    def test_shape(self, key):
        samples = truncated_normal_error(
            np.float64(0.5), SIZE, key, scale=0.1, estimates={}, rho=np.float64(12.0)
        )

        assert samples.shape == (SIZE,)

    def test_is_deterministic_in_the_key(self, key):
        kwargs = dict(scale=0.1, estimates={}, rho=np.float64(12.0))

        first = truncated_normal_error(np.float64(0.5), SIZE, key, **kwargs)
        second = truncated_normal_error(np.float64(0.5), SIZE, key, **kwargs)

        np.testing.assert_array_equal(first, second)

    def test_a_different_key_gives_a_different_draw(self, key):
        kwargs = dict(scale=0.1, estimates={}, rho=np.float64(12.0))

        first = truncated_normal_error(np.float64(0.5), SIZE, key, **kwargs)
        second = truncated_normal_error(np.float64(0.5), SIZE, jrd.PRNGKey(1), **kwargs)

        assert not np.array_equal(first, second)

    def test_unbounded_samples_are_centred_on_the_truth(self, key):
        """Without bounds the model is a plain shift, so the mean of many samples sits
        near the true value up to the shared per-event offset ``r0``.
        """
        samples = truncated_normal_error(
            np.float64(1.0), 20_000, key, scale=0.01, estimates={}, rho=np.float64(12.0)
        )

        assert abs(np.mean(samples) - 1.0) < 0.1

    def test_a_louder_event_gets_a_tighter_posterior(self, key):
        """The width scales as ``12 / rho``."""
        loud = truncated_normal_error(
            np.float64(1.0), 20_000, key, scale=0.1, estimates={}, rho=np.float64(48.0)
        )
        quiet = truncated_normal_error(
            np.float64(1.0), 20_000, key, scale=0.1, estimates={}, rho=np.float64(12.0)
        )

        assert np.std(loud) < np.std(quiet)
        np.testing.assert_allclose(4.0 * np.std(loud), np.std(quiet), rtol=1e-6)

    def test_two_sided_bounds_are_respected(self, key):
        samples = truncated_normal_error(
            np.float64(0.5),
            SIZE,
            key,
            scale=2.0,
            estimates={},
            rho=np.float64(3.0),
            low=0.0,
            high=1.0,
        )

        assert np.all(samples >= 0.0) and np.all(samples <= 1.0)

    def test_a_lower_bound_reflects(self, key):
        samples = truncated_normal_error(
            np.float64(0.05),
            SIZE,
            key,
            scale=0.05,
            estimates={},
            rho=np.float64(12.0),
            low=0.0,
        )

        assert np.all(samples >= 0.0)

    def test_an_upper_bound_reflects(self, key):
        samples = truncated_normal_error(
            np.float64(0.95),
            SIZE,
            key,
            scale=0.05,
            estimates={},
            rho=np.float64(12.0),
            high=1.0,
        )

        assert np.all(samples <= 1.0)

    def test_reflection_only_moves_out_of_bounds_samples(self, key):
        """Samples already inside the range must come back untouched."""
        kwargs = dict(scale=0.01, estimates={}, rho=np.float64(12.0))

        free = truncated_normal_error(np.float64(0.5), SIZE, key, **kwargs)
        bounded = truncated_normal_error(
            np.float64(0.5), SIZE, key, low=0.0, high=1.0, **kwargs
        )

        assert np.all((free > 0.0) & (free < 1.0))
        np.testing.assert_allclose(bounded, free)

    def test_vector_valued_truth(self, key):
        """``x`` may already be a vector of samples, one per draw."""
        x = np.linspace(0.0, 1.0, SIZE)

        samples = truncated_normal_error(
            x, SIZE, key, scale=0.01, estimates={}, rho=np.float64(12.0)
        )

        assert samples.shape == (SIZE,)
        assert np.corrcoef(samples, x)[0, 1] > 0.9


class TestBananaError:
    def test_shape_is_two_columns(self, key):
        samples = banana_error(
            np.float64(30.0),
            np.float64(0.2),
            SIZE,
            key,
            estimates={},
            rho=np.float64(12.0),
        )

        assert samples.shape == (SIZE, 2)

    def test_is_deterministic_in_the_key(self, key):
        kwargs = dict(estimates={}, rho=np.float64(12.0))

        first = banana_error(np.float64(30.0), np.float64(0.2), SIZE, key, **kwargs)
        second = banana_error(np.float64(30.0), np.float64(0.2), SIZE, key, **kwargs)

        np.testing.assert_array_equal(first, second)

    def test_samples_scatter_around_the_truth(self, key):
        samples = banana_error(
            np.float64(30.0),
            np.float64(0.2),
            20_000,
            key,
            estimates={},
            rho=np.float64(20.0),
        )

        chirp_mass, eta = samples[:, 0], samples[:, 1]
        assert np.nanstd(chirp_mass) > 0.0
        np.testing.assert_allclose(np.nanmean(chirp_mass), 30.0, rtol=0.2)
        np.testing.assert_allclose(np.nanmean(eta), 0.2, rtol=0.2)

    def test_unphysical_draws_become_nan(self, key):
        """A symmetric mass ratio above 1/4 does not exist, and the caller drops those
        rows; the model must mark them rather than return them.
        """
        samples = banana_error(
            np.float64(30.0),
            np.float64(0.249),
            4096,
            key,
            estimates={},
            rho=np.float64(4.0),
        )

        eta = samples[:, 1]
        assert np.any(np.isnan(eta))
        assert np.all(eta[~np.isnan(eta)] <= 0.25)

    def test_a_louder_event_gets_a_tighter_posterior(self, key):
        def spread(rho):
            samples = banana_error(
                np.float64(30.0),
                np.float64(0.2),
                20_000,
                key,
                estimates={},
                rho=np.float64(rho),
            )
            return np.nanstd(samples[:, 0]), np.nanstd(samples[:, 1])

        loud_mc, loud_eta = spread(40.0)
        quiet_mc, quiet_eta = spread(10.0)

        assert loud_mc < quiet_mc
        assert loud_eta < quiet_eta

    def test_scales_widen_the_posterior(self, key):
        def spread(scale_Mc, scale_eta):
            samples = banana_error(
                np.float64(30.0),
                np.float64(0.2),
                20_000,
                key,
                estimates={},
                rho=np.float64(20.0),
                scale_Mc=scale_Mc,
                scale_eta=scale_eta,
            )
            return np.nanstd(samples[:, 0]), np.nanstd(samples[:, 1])

        narrow_mc, narrow_eta = spread(1.0, 1.0)
        wide_mc, wide_eta = spread(4.0, 4.0)

        assert wide_mc > narrow_mc
        assert wide_eta > narrow_eta


class TestSpinPhaseHelpers:
    @pytest.mark.parametrize("eta", [0.05, 0.15, 0.25])
    @pytest.mark.parametrize("chi_eff", [-0.5, 0.0, 0.3])
    def test_psi_and_chi_eff_are_inverses(self, eta, chi_eff):
        psi = psi_from_chi_eff_and_eta_neglect_chi_a(chi_eff, eta)

        np.testing.assert_allclose(
            chi_eff_from_psi_and_eta_neglect_chi_a(psi, eta), chi_eff, atol=1e-12
        )

    def test_psi_is_increasing_in_chi_eff(self):
        chi_eff = np.linspace(-1.0, 1.0, 32)

        psi = psi_from_chi_eff_and_eta_neglect_chi_a(chi_eff, 0.2)

        assert np.all(np.diff(psi) > 0.0)

    def test_the_phase_uncertainty_is_linear_in_the_spin_uncertainty(self):
        eta = 0.2

        single = dpsi_from_dXeff_neglect_Xa(0.1, eta)
        double = dpsi_from_dXeff_neglect_Xa(0.2, eta)

        np.testing.assert_allclose(double, 2.0 * single)


class TestMockSpinError:
    @pytest.fixture
    def estimates(self):
        return {P.SYMMETRIC_MASS_RATIO: np.full(SIZE, 0.2)}

    def test_shape(self, key, estimates):
        samples = mock_spin_error(
            np.float64(0.1),
            np.float64(0.2),
            SIZE,
            key,
            estimates=estimates,
            rho=np.float64(12.0),
            scale_chi_eff=np.float64(0.1),
        )

        assert samples.shape == (SIZE,)

    def test_is_deterministic_in_the_key(self, key, estimates):
        kwargs = dict(
            estimates=estimates, rho=np.float64(12.0), scale_chi_eff=np.float64(0.1)
        )

        first = mock_spin_error(np.float64(0.1), np.float64(0.2), SIZE, key, **kwargs)
        second = mock_spin_error(np.float64(0.1), np.float64(0.2), SIZE, key, **kwargs)

        np.testing.assert_array_equal(first, second)

    def test_samples_lie_inside_the_truncation(self, key, estimates):
        """The phase is drawn on ``[-4.2, -1.2]``; mapping that back through a monotone
        transform bounds the effective spin too.
        """
        eta = 0.2
        samples = mock_spin_error(
            np.float64(0.1),
            np.float64(eta),
            SIZE,
            key,
            estimates=estimates,
            rho=np.float64(6.0),
            scale_chi_eff=np.float64(0.3),
        )

        low = chi_eff_from_psi_and_eta_neglect_chi_a(-4.2, eta)
        high = chi_eff_from_psi_and_eta_neglect_chi_a(-1.2, eta)
        assert np.all(samples >= low - 1e-6) and np.all(samples <= high + 1e-6)

    def test_a_louder_event_gets_a_tighter_posterior(self, key, estimates):
        def spread(rho):
            return np.std(
                mock_spin_error(
                    np.float64(0.1),
                    np.float64(0.2),
                    SIZE,
                    key,
                    estimates=estimates,
                    rho=np.float64(rho),
                    scale_chi_eff=np.float64(0.1),
                )
            )

        assert spread(48.0) < spread(12.0)

    def test_a_missing_mass_ratio_estimate_is_rejected(self, key):
        """The effective spin is recovered with the *observed* symmetric mass ratio, so
        that column must already have been estimated.
        """
        with pytest.raises(LoggedValueError, match="Symmetric Mass Ratio"):
            mock_spin_error(
                np.float64(0.1),
                np.float64(0.2),
                SIZE,
                key,
                estimates={},
                rho=np.float64(12.0),
                scale_chi_eff=np.float64(0.1),
            )
