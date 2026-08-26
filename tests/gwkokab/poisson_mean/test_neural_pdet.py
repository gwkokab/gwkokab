# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
from jax import numpy as jnp, random as jrd
from numpyro.distributions import Independent, Uniform

from gwkokab.models import PowerlawRedshiftModel
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean import poisson_mean_from_neural_pdet
from gwkokab.utils.exceptions import LoggedTypeError, LoggedValueError


NAMES = [P.PRIMARY_MASS_SOURCE.value, P.REDSHIFT.value]
PARAMETERS = [P.REDSHIFT.value, P.PRIMARY_MASS_SOURCE.value]
LOG_SCALES = jnp.log(jnp.asarray([2.0, 5.0]))


def _joint_mixture():
    """Two ``JointDistribution`` components, neither carrying a redshift model."""
    return ScaledMixture(
        LOG_SCALES,
        [
            JointDistribution(Uniform(0.0, 1.0), Uniform(10.0, 50.0)),
            JointDistribution(Uniform(0.0, 0.5), Uniform(20.0, 30.0)),
        ],
    )


def _redshift_mixture():
    """Two ``JointDistribution`` components, each carrying a distinct redshift model."""
    return ScaledMixture(
        LOG_SCALES,
        [
            JointDistribution(
                PowerlawRedshiftModel(z_max=jnp.asarray(1.5), kappa=jnp.asarray(2.0)),
                Uniform(10.0, 50.0),
            ),
            JointDistribution(
                PowerlawRedshiftModel(z_max=jnp.asarray(0.5), kappa=jnp.asarray(1.0)),
                Uniform(20.0, 30.0),
            ),
        ],
    )


def _independent_mixture():
    """Two components that are *not* ``JointDistribution``s."""
    return ScaledMixture(
        LOG_SCALES,
        [
            Independent(Uniform(jnp.zeros(2), jnp.asarray([1.0, 50.0])), 1),
            Independent(Uniform(jnp.zeros(2), jnp.asarray([0.5, 30.0])), 1),
        ],
    )


def test_invalid_parameters_empty(linear_model_file):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedValueError, match="cannot be empty"):
        poisson_mean_from_neural_pdet(jrd.key(0), [], filename)


@pytest.mark.parametrize("parameters", [{"a", "b"}, 3, iter(PARAMETERS)])
def test_invalid_parameters_not_a_sequence(linear_model_file, parameters):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedTypeError, match="must be a Sequence"):
        poisson_mean_from_neural_pdet(jrd.key(0), parameters, filename)


def test_invalid_parameters_not_strings(linear_model_file):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedTypeError, match="must be strings"):
        poisson_mean_from_neural_pdet(jrd.key(0), [P.REDSHIFT.value, 1], filename)


@pytest.mark.parametrize("batch_size", [1.5, "8"])
def test_invalid_batch_size_type(linear_model_file, batch_size):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedTypeError, match="must be an integer"):
        poisson_mean_from_neural_pdet(
            jrd.key(0), PARAMETERS, filename, batch_size=batch_size
        )


@pytest.mark.parametrize("batch_size", [0, -3])
def test_invalid_batch_size_value(linear_model_file, batch_size):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedValueError, match="positive integer"):
        poisson_mean_from_neural_pdet(
            jrd.key(0), PARAMETERS, filename, batch_size=batch_size
        )


def test_missing_parameter_for_model(linear_model_file):
    filename = linear_model_file(NAMES, [1.0, 1.0])
    with pytest.raises(LoggedValueError, match="Missing"):
        poisson_mean_from_neural_pdet(
            jrd.key(0), [P.PRIMARY_MASS_SOURCE.value, P.ECCENTRICITY.value], filename
        )


def test_log_pdet_takes_the_log_and_reorders_columns(linear_model_file):
    """Unlike the VT network, this network outputs :math:`p_{det}` itself."""
    coefficients = [0.001, 0.01]
    bias = 0.05
    filename = linear_model_file(NAMES, coefficients, bias)
    log_pdet, _, _ = poisson_mean_from_neural_pdet(jrd.key(0), PARAMETERS, filename)

    # columns are (redshift, mass_1_source); the model wants (mass_1_source, redshift)
    x = jnp.asarray([[0.2, 30.0], [0.7, 12.0]])
    expected = np.log(coefficients[0] * x[:, 1] + coefficients[1] * x[:, 0] + bias)
    np.testing.assert_allclose(log_pdet(x), expected, rtol=1e-6)


def test_log_pdet_is_neginf_on_non_positive_output(linear_model_file):
    """A non-positive network output means "never detected", not ``nan``."""
    filename = linear_model_file(NAMES, [0.01, 0.0], -0.2)
    log_pdet, _, _ = poisson_mean_from_neural_pdet(jrd.key(0), PARAMETERS, filename)

    # 0.01 * mass_1_source - 0.2, so mass_1_source = 20 sits exactly on the boundary
    x = jnp.asarray([[0.0, 10.0], [0.0, 20.0], [0.0, 50.0]])
    values = log_pdet(x)
    assert jnp.isneginf(values[0])
    assert jnp.isneginf(values[1])
    np.testing.assert_allclose(values[2], np.log(0.3), rtol=1e-6)


def test_time_scale_is_returned(linear_model_file):
    filename = linear_model_file(NAMES, [0.0, 0.0], 0.5)
    _, _, kwargs = poisson_mean_from_neural_pdet(
        jrd.key(0), PARAMETERS, filename, time_scale=7.5
    )
    assert kwargs == {"T_obs": 7.5}


@pytest.mark.parametrize("mixture_fn", [_joint_mixture, _independent_mixture])
def test_constant_pdet_without_redshift_model(linear_model_file, mixture_fn):
    r"""Without a :class:`PowerlawRedshiftModel` the log normalisation is zero, so a
    constant :math:`p_{det} \equiv p_0` gives :math:`\mu = T p_0 \sum_k R_k` exactly.
    """
    p_0 = 0.4
    time_scale = 3.0
    filename = linear_model_file(NAMES, [0.0, 0.0], p_0)
    _, poisson_mean, kwargs = poisson_mean_from_neural_pdet(
        jrd.key(0), PARAMETERS, filename, num_samples=16, time_scale=time_scale
    )

    mean, variance = poisson_mean(mixture_fn(), kwargs["T_obs"])

    np.testing.assert_allclose(
        mean, time_scale * p_0 * np.sum(np.exp(LOG_SCALES)), rtol=1e-5
    )
    np.testing.assert_allclose(variance, 0.0, atol=1e-5)


def test_constant_pdet_applies_per_component_redshift_normalisation(linear_model_file):
    """Each component's redshift normalisation must multiply *that* component's rate."""
    p_0 = 0.4
    time_scale = 3.0
    filename = linear_model_file(NAMES, [0.0, 0.0], p_0)
    _, poisson_mean, kwargs = poisson_mean_from_neural_pdet(
        jrd.key(0), PARAMETERS, filename, num_samples=16, time_scale=time_scale
    )

    mixture = _redshift_mixture()
    mean, variance = poisson_mean(mixture, kwargs["T_obs"])

    log_norms = jnp.asarray([
        component.marginal_distributions[0].log_norm()
        for component in mixture.component_distributions
    ])
    expected = time_scale * p_0 * jnp.sum(jnp.exp(LOG_SCALES + log_norms))

    assert not jnp.allclose(log_norms[0], log_norms[1])
    np.testing.assert_allclose(mean, expected, rtol=1e-5)
    np.testing.assert_allclose(variance, 0.0, atol=1e-5)


def test_undetectable_population_gives_zero_mean(linear_model_file):
    """If the network reports :math:`p_{det} \\leq 0` everywhere the mean vanishes."""
    filename = linear_model_file(NAMES, [0.0, 0.0], -1.0)
    _, poisson_mean, kwargs = poisson_mean_from_neural_pdet(
        jrd.key(0), PARAMETERS, filename, num_samples=16
    )
    mean, variance = poisson_mean(_joint_mixture(), kwargs["T_obs"])

    np.testing.assert_allclose(mean, 0.0, atol=1e-12)
    np.testing.assert_allclose(variance, 0.0, atol=1e-12)


def test_matches_reference_implementation(linear_model_file):
    """Cross-check against a plain-`numpy` transcription of equations 9 and 11 of
    `arXiv:2406.16813 <https://arxiv.org/abs/2406.16813>`_.
    """
    num_samples = 32
    time_scale = 1.7
    filename = linear_model_file(NAMES, [0.001, 0.01], 0.05)
    key = jrd.key(7)
    log_pdet, poisson_mean, kwargs = poisson_mean_from_neural_pdet(
        key, PARAMETERS, filename, num_samples=num_samples, time_scale=time_scale
    )

    mixture = _redshift_mixture()
    mean, variance = poisson_mean(mixture, kwargs["T_obs"])

    samples = mixture.component_sample(key, (num_samples,))
    rates = np.exp(np.asarray(LOG_SCALES, dtype=np.float64)) * np.exp([
        np.asarray(component.marginal_distributions[0].log_norm(), dtype=np.float64)
        for component in mixture.component_distributions
    ])
    expected_mean = 0.0
    expected_variance = 0.0
    for k, rate in enumerate(rates):
        pdet = np.exp(np.asarray(log_pdet(samples[:, k, :]), dtype=np.float64))
        expected_mean += (time_scale / num_samples) * rate * pdet.sum()
        expected_variance += (rate * time_scale) ** 2 * (
            np.square(pdet).sum() / num_samples**2 - pdet.sum() ** 2 / num_samples**3
        )

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-4)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("time_scale", [0.5, 4.0])
def test_scaling_with_observation_time(linear_model_file, time_scale):
    """The mean is linear and the variance quadratic in ``T_obs``."""
    filename = linear_model_file(NAMES, [0.001, 0.01], 0.05)

    def _estimate(t):
        _, poisson_mean, kwargs = poisson_mean_from_neural_pdet(
            jrd.key(3), PARAMETERS, filename, num_samples=32, time_scale=t
        )
        return poisson_mean(_joint_mixture(), kwargs["T_obs"])

    mean, variance = _estimate(1.0)
    scaled_mean, scaled_variance = _estimate(time_scale)

    np.testing.assert_allclose(scaled_mean, time_scale * mean, rtol=1e-5)
    np.testing.assert_allclose(scaled_variance, time_scale**2 * variance, rtol=1e-4)


def test_variance_is_non_negative(linear_model_file):
    filename = linear_model_file(NAMES, [0.001, 0.01], 0.05)
    _, poisson_mean, kwargs = poisson_mean_from_neural_pdet(
        jrd.key(11), PARAMETERS, filename, num_samples=64
    )
    _, variance = poisson_mean(_redshift_mixture(), kwargs["T_obs"])
    assert variance >= 0.0


@pytest.mark.parametrize("batch_size", [1, 4, None])
def test_batch_size_does_not_change_the_estimate(linear_model_file, batch_size):
    filename = linear_model_file(NAMES, [0.001, 0.01], 0.05)

    def _estimate(size):
        _, poisson_mean, kwargs = poisson_mean_from_neural_pdet(
            jrd.key(5), PARAMETERS, filename, batch_size=size, num_samples=32
        )
        return poisson_mean(_joint_mixture(), kwargs["T_obs"])

    np.testing.assert_allclose(_estimate(batch_size), _estimate(None), rtol=1e-5)
