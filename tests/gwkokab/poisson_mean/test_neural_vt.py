# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from jax import grad, numpy as jnp, random as jrd
from numpyro.distributions import Uniform

from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean import poisson_mean_from_neural_vt
from gwkokab.utils.exceptions import LoggedTypeError, LoggedValueError


NAMES = [P.PRIMARY_MASS_SOURCE.value, P.REDSHIFT.value]
PARAMETERS = [P.REDSHIFT.value, P.PRIMARY_MASS_SOURCE.value]


def _mixture(log_scales):
    """Two component mixture over ``(redshift, mass_1_source)``."""
    return ScaledMixture(
        log_scales,
        [
            JointDistribution(Uniform(0.0, 1.0), Uniform(10.0, 50.0)),
            JointDistribution(Uniform(0.0, 0.5), Uniform(20.0, 30.0)),
        ],
    )


def test_invalid_parameters_empty(linear_model_file):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedValueError, match="cannot be empty"):
        poisson_mean_from_neural_vt(jrd.key(0), [], filename)


@pytest.mark.parametrize("parameters", [{"a", "b"}, 3, iter(PARAMETERS)])
def test_invalid_parameters_not_a_sequence(linear_model_file, parameters):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedTypeError, match="must be a Sequence"):
        poisson_mean_from_neural_vt(jrd.key(0), parameters, filename)


def test_invalid_parameters_not_strings(linear_model_file):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedTypeError, match="must be strings"):
        poisson_mean_from_neural_vt(jrd.key(0), [P.REDSHIFT.value, 1], filename)


@pytest.mark.parametrize("batch_size", [1.5, "8"])
def test_invalid_batch_size_type(linear_model_file, batch_size):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedTypeError, match="must be an integer"):
        poisson_mean_from_neural_vt(
            jrd.key(0), PARAMETERS, filename, batch_size=batch_size
        )


@pytest.mark.parametrize("batch_size", [0, -3])
def test_invalid_batch_size_value(linear_model_file, batch_size):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedValueError, match="positive integer"):
        poisson_mean_from_neural_vt(
            jrd.key(0), PARAMETERS, filename, batch_size=batch_size
        )


def test_missing_parameter_for_model(linear_model_file):
    """The model's own parameters must all be present in ``parameters``."""
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedValueError, match="Missing"):
        poisson_mean_from_neural_vt(
            jrd.key(0), [P.PRIMARY_MASS_SOURCE.value, P.ECCENTRICITY.value], filename
        )


def test_log_vt_reorders_columns(linear_model_file):
    """``log_vt`` must feed the network its own parameter order, not the caller's."""
    coefficients = [1.0, 10.0]
    bias = 0.5
    filename = linear_model_file(NAMES, coefficients, bias)
    log_vt, _, _ = poisson_mean_from_neural_vt(jrd.key(0), PARAMETERS, filename)

    # columns are (redshift, mass_1_source); the model wants (mass_1_source, redshift)
    x = jnp.asarray([[0.2, 30.0], [0.7, 12.0], [0.0, -4.0]])
    expected = coefficients[0] * x[:, 1] + coefficients[1] * x[:, 0] + bias
    np.testing.assert_allclose(log_vt(x), expected, rtol=1e-6)


def test_extra_parameters_are_ignored(linear_model_file):
    """Columns the network was not trained on must not influence ``log_vt``."""
    filename = linear_model_file(NAMES, [1.0, 10.0], 0.5)
    parameters = [*PARAMETERS, P.ECCENTRICITY.value]
    log_vt, _, _ = poisson_mean_from_neural_vt(jrd.key(0), parameters, filename)

    x = jnp.asarray([[0.2, 30.0, 0.0], [0.2, 30.0, 1e3]])
    np.testing.assert_allclose(log_vt(x)[0], log_vt(x)[1], rtol=1e-6)


def test_time_scale_is_returned(linear_model_file):
    filename = linear_model_file(NAMES, [0.0, 0.0])
    _, _, kwargs = poisson_mean_from_neural_vt(
        jrd.key(0), PARAMETERS, filename, time_scale=7.5
    )
    assert kwargs == {"T_obs": 7.5}


@pytest.mark.parametrize("num_samples", [8, 128])
def test_constant_vt_gives_exact_mean_and_zero_variance(linear_model_file, num_samples):
    r"""For a constant :math:`VT \equiv v_0` the estimator collapses to
    :math:`\mu = T v_0 \sum_k R_k` with zero Monte-Carlo variance, independently of the
    number of samples drawn.
    """
    v_0 = 0.3
    time_scale = 3.0
    filename = linear_model_file(NAMES, [0.0, 0.0], float(np.log(v_0)))
    _, poisson_mean, kwargs = poisson_mean_from_neural_vt(
        jrd.key(0),
        PARAMETERS,
        filename,
        num_samples=num_samples,
        time_scale=time_scale,
    )

    log_scales = jnp.log(jnp.asarray([2.0, 5.0]))
    mean, variance = poisson_mean(_mixture(log_scales), kwargs["T_obs"])

    np.testing.assert_allclose(
        mean, time_scale * v_0 * np.sum(np.exp(log_scales)), rtol=1e-5
    )
    np.testing.assert_allclose(variance, 0.0, atol=1e-5)


def test_matches_reference_implementation(linear_model_file):
    """Cross-check against a plain-`numpy` transcription of equations 9 and 11 of
    `arXiv:2406.16813 <https://arxiv.org/abs/2406.16813>`_, which avoids the
    ``logsumexp`` bookkeeping of the implementation.
    """
    num_samples = 32
    time_scale = 1.7
    filename = linear_model_file(NAMES, [-0.05, -1.0], -1.0)
    key = jrd.key(7)
    log_vt, poisson_mean, kwargs = poisson_mean_from_neural_vt(
        key, PARAMETERS, filename, num_samples=num_samples, time_scale=time_scale
    )

    log_scales = jnp.log(jnp.asarray([2.0, 0.5]))
    mixture = _mixture(log_scales)
    mean, variance = poisson_mean(mixture, kwargs["T_obs"])

    samples = mixture.component_sample(key, (num_samples,))
    rates = np.exp(np.asarray(log_scales, dtype=np.float64))
    expected_mean = 0.0
    expected_variance = 0.0
    for k, rate in enumerate(rates):
        vt = np.exp(np.asarray(log_vt(samples[:, k, :]), dtype=np.float64))
        expected_mean += (time_scale / num_samples) * rate * vt.sum()
        expected_variance += (rate * time_scale) ** 2 * (
            np.square(vt).sum() / num_samples**2 - vt.sum() ** 2 / num_samples**3
        )

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-4)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("time_scale", [0.5, 4.0])
def test_scaling_with_observation_time(linear_model_file, time_scale):
    """The mean is linear and the variance quadratic in ``T_obs``."""
    filename = linear_model_file(NAMES, [-0.05, -1.0], -1.0)
    log_scales = jnp.log(jnp.asarray([2.0, 0.5]))

    def _estimate(t):
        _, poisson_mean, kwargs = poisson_mean_from_neural_vt(
            jrd.key(3), PARAMETERS, filename, num_samples=32, time_scale=t
        )
        return poisson_mean(_mixture(log_scales), kwargs["T_obs"])

    mean, variance = _estimate(1.0)
    scaled_mean, scaled_variance = _estimate(time_scale)

    np.testing.assert_allclose(scaled_mean, time_scale * mean, rtol=1e-5)
    np.testing.assert_allclose(scaled_variance, time_scale**2 * variance, rtol=1e-4)


def test_scaling_with_rates(linear_model_file):
    """The mean is linear in the component rates :math:`\\exp(\\log R_k)`."""
    filename = linear_model_file(NAMES, [-0.05, -1.0], -1.0)
    _, poisson_mean, kwargs = poisson_mean_from_neural_vt(
        jrd.key(3), PARAMETERS, filename, num_samples=32
    )

    log_scales = jnp.log(jnp.asarray([2.0, 0.5]))
    mean, variance = poisson_mean(_mixture(log_scales), kwargs["T_obs"])
    doubled_mean, doubled_variance = poisson_mean(
        _mixture(log_scales + jnp.log(2.0)), kwargs["T_obs"]
    )

    np.testing.assert_allclose(doubled_mean, 2.0 * mean, rtol=1e-5)
    np.testing.assert_allclose(doubled_variance, 4.0 * variance, rtol=1e-4)


def test_variance_is_non_negative(linear_model_file):
    """Cauchy-Schwarz guarantees ``term1 >= term2`` for every component."""
    filename = linear_model_file(NAMES, [-0.2, -3.0], 1.0)
    _, poisson_mean, kwargs = poisson_mean_from_neural_vt(
        jrd.key(11), PARAMETERS, filename, num_samples=64
    )
    _, variance = poisson_mean(
        _mixture(jnp.log(jnp.asarray([2.0, 0.5]))), kwargs["T_obs"]
    )
    assert variance >= 0.0


@pytest.mark.parametrize("batch_size", [1, 4, None])
def test_batch_size_does_not_change_the_estimate(linear_model_file, batch_size):
    """``batch_size`` only controls how ``jax.lax.map`` chunks the network calls."""
    filename = linear_model_file(NAMES, [-0.05, -1.0], -1.0)
    log_scales = jnp.log(jnp.asarray([2.0, 0.5]))

    def _estimate(size):
        _, poisson_mean, kwargs = poisson_mean_from_neural_vt(
            jrd.key(5), PARAMETERS, filename, batch_size=size, num_samples=32
        )
        return poisson_mean(_mixture(log_scales), kwargs["T_obs"])

    np.testing.assert_allclose(_estimate(batch_size), _estimate(None), rtol=1e-5)


def test_mean_is_differentiable_in_log_scales(linear_model_file):
    r"""Since :math:`\mu = \sum_k e^{\log R_k} a_k`, Euler's identity gives
    :math:`\sum_k \partial\mu/\partial\log R_k = \mu`.
    """
    filename = linear_model_file(NAMES, [-0.05, -1.0], -1.0)
    _, poisson_mean, kwargs = poisson_mean_from_neural_vt(
        jrd.key(5), PARAMETERS, filename, num_samples=32
    )

    def _mean(log_scales):
        return poisson_mean(_mixture(log_scales), kwargs["T_obs"])[0]

    log_scales = jnp.log(jnp.asarray([2.0, 0.5]))
    gradient = grad(_mean)(log_scales)

    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(jnp.sum(gradient), _mean(log_scales), rtol=1e-5)
