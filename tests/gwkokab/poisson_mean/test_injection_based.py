# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from jax import numpy as jnp, random as jrd
from numpyro.distributions import Uniform

from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean import poisson_mean_from_sensitivity_injections


PARAMETERS = [
    P.PRIMARY_MASS_SOURCE.value,
    P.SECONDARY_MASS_SOURCE.value,
    P.REDSHIFT.value,
]
# a flat population over a box that contains every injection the fixture writes
BOX = (60.0, 60.0, 2.0)
LOG_RATE = float(np.log(3.0))


def _flat_mixture(*, masked: bool = False) -> ScaledMixture:
    r"""A one component mixture whose ``log_prob`` is the constant
    :math:`\log R - \sum_d \log \Delta_d` inside the box.

    When `masked` is set, the mixture is given an explicit support so that
    :meth:`ScaledMixture.component_log_probs` clamps out-of-box injections to
    :math:`-\infty` (plain :class:`~numpyro.distributions.Uniform` does not).
    """
    component = JointDistribution(*(Uniform(0.0, high) for high in BOX))
    support = component.support if masked else None
    return ScaledMixture(jnp.asarray([LOG_RATE]), [component], support=support)


def _constant_log_prob() -> float:
    return LOG_RATE - float(np.sum(np.log(BOX)))


def test_first_estimator_is_none(injections_file):
    """Unlike the neural estimators there is no per-point sensitivity function."""
    log_vt, poisson_mean, kwargs = poisson_mean_from_sensitivity_injections(
        jrd.key(0), PARAMETERS, injections_file()
    )
    assert log_vt is None
    assert callable(poisson_mean)
    assert set(kwargs) == {"samples", "log_weights", "T_obs"}


def test_samples_follow_the_requested_parameter_order(injections_file):
    mass_1 = np.linspace(10.0, 50.0, 5)
    redshift = np.linspace(0.05, 0.9, 5)
    filename = injections_file(mass_1=mass_1, redshift=redshift)

    _, _, kwargs = poisson_mean_from_sensitivity_injections(
        jrd.key(0), [P.REDSHIFT.value, P.PRIMARY_MASS_SOURCE.value], filename
    )

    assert kwargs["samples"].shape == (5, 2)
    np.testing.assert_allclose(kwargs["samples"][:, 0], redshift, rtol=1e-6)
    np.testing.assert_allclose(kwargs["samples"][:, 1], mass_1, rtol=1e-6)


@pytest.mark.parametrize("prior_value", [1.0, 4.0])
def test_analytic_mean_and_variance(injections_file, prior_value):
    r"""With a constant integrand the estimator reduces to

    .. math::

        \mu = \frac{T n s}{N}, \qquad
        \sigma^2 = \frac{T^2 n s^2 (N - n)}{N^3},

    where :math:`s = e^{\log p - \log w}`, :math:`n` is the number of found injections
    and :math:`N` the number generated.
    """
    n_found, total_generated, analysis_time = 8, 100, 2.0
    filename = injections_file(
        total_generated=total_generated,
        analysis_time_years=analysis_time,
        prior_value=prior_value,
    )

    _, poisson_mean, kwargs = poisson_mean_from_sensitivity_injections(
        jrd.key(0), PARAMETERS, filename
    )
    mean, variance = poisson_mean(
        _flat_mixture(), kwargs["samples"], kwargs["log_weights"], kwargs["T_obs"]
    )

    s = np.exp(_constant_log_prob()) / prior_value
    expected_mean = analysis_time * n_found * s / total_generated
    expected_variance = (
        analysis_time**2
        * n_found
        * s**2
        * (total_generated - n_found)
        / total_generated**3
    )

    np.testing.assert_allclose(kwargs["T_obs"], analysis_time, rtol=1e-9)
    np.testing.assert_allclose(
        kwargs["log_weights"], np.full(n_found, np.log(prior_value)), rtol=1e-6
    )
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-4)


def test_out_of_support_injections_are_dropped(injections_file):
    """Injections where the population density vanishes must not enter the sum, while
    still counting towards the total number generated.
    """
    total_generated = 100
    analysis_time = 2.0
    # the last three redshifts sit outside the population box (z_max = 2)
    redshift = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 5.0, 6.0, 7.0])
    filename = injections_file(
        redshift=redshift,
        total_generated=total_generated,
        analysis_time_years=analysis_time,
    )

    _, poisson_mean, kwargs = poisson_mean_from_sensitivity_injections(
        jrd.key(0), PARAMETERS, filename
    )
    mean, variance = poisson_mean(
        _flat_mixture(masked=True),
        kwargs["samples"],
        kwargs["log_weights"],
        kwargs["T_obs"],
    )

    n_inside = 5
    s = np.exp(_constant_log_prob())
    np.testing.assert_allclose(
        mean, analysis_time * n_inside * s / total_generated, rtol=1e-5
    )
    np.testing.assert_allclose(
        variance,
        analysis_time**2
        * n_inside
        * s**2
        * (total_generated - n_inside)
        / total_generated**3,
        rtol=1e-4,
    )


def test_far_cut_filters_injections(injections_file):
    """``far_cut`` is applied as an inverse-FAR threshold of ``1 / far_cut`` years."""
    ifar = np.asarray([0.1, 0.5, 2.0, 1e3, 1e3, 0.9, 3.0, 1e4])
    filename = injections_file(ifar=ifar)

    _, _, kwargs = poisson_mean_from_sensitivity_injections(
        jrd.key(0), PARAMETERS, filename, far_cut=1.0
    )
    assert kwargs["samples"].shape[0] == int(np.sum(ifar > 1.0))

    _, _, loose_kwargs = poisson_mean_from_sensitivity_injections(
        jrd.key(0), PARAMETERS, filename, far_cut=10.0
    )
    assert loose_kwargs["samples"].shape[0] == int(np.sum(ifar > 0.1))


def test_no_injection_passes_the_threshold(injections_file):
    filename = injections_file(ifar=np.full(8, 0.01))
    with pytest.raises(ValueError, match="No sensitivity injections pass threshold"):
        poisson_mean_from_sensitivity_injections(jrd.key(0), PARAMETERS, filename)


@pytest.mark.parametrize("batch_size", [1, 3, None])
def test_batch_size_does_not_change_the_estimate(injections_file, batch_size):
    filename = injections_file()

    def _estimate(size):
        _, poisson_mean, kwargs = poisson_mean_from_sensitivity_injections(
            jrd.key(0), PARAMETERS, filename, batch_size=size
        )
        return poisson_mean(
            _flat_mixture(), kwargs["samples"], kwargs["log_weights"], kwargs["T_obs"]
        )

    np.testing.assert_allclose(_estimate(batch_size), _estimate(None), rtol=1e-5)


def test_key_is_unused(injections_file):
    """The injection estimator is deterministic; the PRNG key is accepted and
    dropped.
    """
    filename = injections_file()

    def _estimate(key):
        _, poisson_mean, kwargs = poisson_mean_from_sensitivity_injections(
            key, PARAMETERS, filename
        )
        return poisson_mean(
            _flat_mixture(), kwargs["samples"], kwargs["log_weights"], kwargs["T_obs"]
        )

    np.testing.assert_array_equal(_estimate(jrd.key(0)), _estimate(jrd.key(12345)))


def test_scaling_with_observation_time(injections_file):
    """The mean is linear and the variance quadratic in the analysis time."""

    def _estimate(analysis_time):
        filename = injections_file(analysis_time_years=analysis_time)
        _, poisson_mean, kwargs = poisson_mean_from_sensitivity_injections(
            jrd.key(0), PARAMETERS, filename
        )
        return poisson_mean(
            _flat_mixture(), kwargs["samples"], kwargs["log_weights"], kwargs["T_obs"]
        )

    mean, variance = _estimate(1.0)
    scaled_mean, scaled_variance = _estimate(3.0)

    np.testing.assert_allclose(scaled_mean, 3.0 * mean, rtol=1e-5)
    np.testing.assert_allclose(scaled_variance, 9.0 * variance, rtol=1e-4)
