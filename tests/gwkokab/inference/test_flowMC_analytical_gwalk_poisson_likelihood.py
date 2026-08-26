# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flowMC analytical-GWalk log-posterior wrapper.

This wrapper is the analytical-GWalk twin of the discrete one, but with two deliberate
differences that the tests pin down: it takes its arguments positionally with
``constant_params`` in the middle, and it has no ``where_fns`` guard at all.
"""

import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpyro.distributions import Normal, Uniform

from gwkokab.inference import flowMC_analytical_gwalk_poisson_likelihood
from gwkokab.models.utils import JointDistribution


def _build(
    dist_fn,
    est,
    *,
    priors=None,
    constant_params=None,
    variables_index=None,
    variance_cut_threshold=None,
):
    priors = JointDistribution(Uniform(0.0, 2.0)) if priors is None else priors
    return (
        flowMC_analytical_gwalk_poisson_likelihood(
            dist_fn,
            priors,
            None,
            {} if constant_params is None else constant_params,
            {"log_scale": 0} if variables_index is None else variables_index,
            est,
            variance_cut_threshold,
        ),
        priors,
    )


@pytest.mark.parametrize("log_scale", [-1.0, 0.0, 0.75, 1.9])
@pytest.mark.parametrize(("n_events", "n_samples"), [(1, 3), (3, 5)])
def test_log_posterior_is_prior_plus_likelihood(
    model_factory, estimator, analytical_data, log_scale, n_events, n_samples
):
    log_posterior, priors = _build(model_factory(), estimator())
    data = analytical_data(n_events=n_events, n_samples=n_samples)
    x = jnp.asarray([log_scale])
    expected = n_events * (log_scale + np.log(n_samples)) + priors.log_prob(x)
    chex.assert_trees_all_close(log_posterior(x, data), expected)


def test_variables_index_selects_the_right_component(
    model_factory, estimator, analytical_data
):
    """A permuted index must change which slot the model reads."""
    dist_fn = model_factory()
    priors = JointDistribution(Uniform(0.0, 2.0), Uniform(0.0, 2.0))
    data = analytical_data(n_events=3, n_samples=5)
    x = jnp.asarray([0.25, 1.5])
    for variables_index, log_scale in (
        ({"unused": 0, "log_scale": 1}, 1.5),
        ({"log_scale": 0, "unused": 1}, 0.25),
    ):
        log_posterior, _ = _build(
            dist_fn, estimator(), priors=priors, variables_index=variables_index
        )
        expected = 3 * (log_scale + np.log(5)) + priors.log_prob(x)
        chex.assert_trees_all_close(log_posterior(x, data), expected)
    assert set(dist_fn.last_call) == {"unused", "log_scale"}


def test_constant_params_are_forwarded(model_factory, estimator, analytical_data):
    """The fourth positional argument is ``constant_params``, not ``constants``."""
    dist_fn = model_factory()
    log_posterior, _ = _build(
        dist_fn, estimator(), constant_params={"mmin": jnp.asarray(5.0)}
    )
    log_posterior(jnp.asarray([0.75]), analytical_data())
    assert dist_fn.last_call["mmin"] == 5.0


def test_variables_argument_is_unused(model_factory, estimator, analytical_data):
    data = analytical_data()
    x = jnp.asarray([0.75])
    priors = JointDistribution(Uniform(0.0, 2.0))
    results = [
        flowMC_analytical_gwalk_poisson_likelihood(
            model_factory(),
            priors,
            variables,
            {},
            {"log_scale": 0},
            estimator(),
            None,
        )(x, data)
        for variables in (None, {}, {"log_scale": Normal(0.0, 1.0)}, "junk")
    ]
    chex.assert_trees_all_close(results[0], *results[1:])


def test_estimator_receives_the_pmean_kwargs(model_factory, estimator, analytical_data):
    est = estimator()
    log_posterior, _ = _build(model_factory(), est)
    log_posterior(jnp.asarray([0.75]), analytical_data(T_obs=3.0))
    assert len(est.calls) == 1
    assert set(est.calls[0][1]) == {"T_obs"}


###############################################################################
# sanitisation
###############################################################################


@pytest.mark.parametrize("expected_rate", [jnp.nan, -jnp.inf, jnp.inf])
def test_non_finite_log_posterior_becomes_minus_inf(
    model_factory, estimator, analytical_data, expected_rate
):
    """NaN and both infinities collapse to ``-inf`` so flowMC simply rejects."""
    log_posterior, _ = _build(model_factory(), estimator(mean=expected_rate))
    out = log_posterior(jnp.asarray([0.75]), analytical_data())
    assert bool(jnp.isneginf(out))


def test_no_support_guard(model_factory, estimator, analytical_data):
    """This wrapper has no ``where_fns`` parameter, so nothing rejects an out-of-support
    position; the prior's own log-density is all there is.
    """
    log_posterior, priors = _build(model_factory(), estimator())
    x = jnp.asarray([5.0])
    data = analytical_data(n_events=3, n_samples=5)
    expected = 3 * (5.0 + np.log(5)) + priors.log_prob(x)
    chex.assert_trees_all_close(log_posterior(x, data), expected)


###############################################################################
# transformations
###############################################################################


def test_variance_cut_threshold_is_applied(model_factory, estimator, analytical_data):
    data = analytical_data(n_events=3, n_samples=5)
    x = jnp.asarray([0.75])
    est_variance, threshold = 4.0, 1.0
    plain, _ = _build(model_factory(), estimator(variance=est_variance))
    tapered, _ = _build(
        model_factory(),
        estimator(variance=est_variance),
        variance_cut_threshold=threshold,
    )
    chex.assert_trees_all_close(
        plain(x, data) - tapered(x, data),
        jnp.asarray(100.0 * (est_variance - threshold) ** 2),
    )


def test_is_jittable(model_factory, estimator, analytical_data):
    """Unlike the discrete wrapper this one is returned unjitted, but the caller still
    has to be able to compile it.
    """
    log_posterior, priors = _build(model_factory(), estimator())
    data = analytical_data(n_events=3, n_samples=5)
    x = jnp.asarray([0.75])
    chex.assert_trees_all_close(jax.jit(log_posterior)(x, data), log_posterior(x, data))


def test_is_vmappable_over_walkers(model_factory, estimator, analytical_data):
    log_posterior, priors = _build(model_factory(), estimator())
    data = analytical_data(n_events=3, n_samples=5)
    positions = jnp.asarray([[0.25], [0.75], [1.5]])
    out = jax.vmap(log_posterior, in_axes=(0, None))(positions, data)
    expected = jnp.asarray([
        3 * (float(p) + np.log(5)) + float(priors.log_prob(jnp.asarray([p])))
        for p in positions[:, 0]
    ])
    chex.assert_trees_all_close(out, expected)


def test_is_differentiable(model_factory, estimator, analytical_data):
    log_posterior, _ = _build(model_factory(), estimator())
    data = analytical_data(n_events=3, n_samples=5)
    grad = jax.grad(log_posterior)(jnp.asarray([0.75]), data)
    chex.assert_trees_all_close(grad, jnp.asarray([3.0]))


def test_result_is_a_scalar(model_factory, estimator, analytical_data):
    log_posterior, _ = _build(model_factory(), estimator())
    chex.assert_shape(log_posterior(jnp.asarray([0.75]), analytical_data()), ())
