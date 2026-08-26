# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NumPyro analytical-GWalk likelihood wrapper.

Three differences from its discrete sibling drive most of these tests: the signature is
positional with ``constant_params`` in the middle, there is no ``where_fns`` argument,
and the NaN guard is unconditional rather than tied to ``where_fns``.
"""

import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpyro.distributions import Normal, Uniform
from numpyro.infer.util import log_density

from gwkokab.inference import numpyro_analytical_gwalk_poisson_likelihood
from gwkokab.models.utils import JointDistribution, LazyJointDistribution


def _model_args(data):
    return (data["samples_stack"], data["ln_offsets"], data["pmean_kwargs"])


def _build(
    dist_fn,
    est,
    *,
    priors=None,
    variables=None,
    constant_params=None,
    variables_index=None,
    variance_cut_threshold=None,
):
    return numpyro_analytical_gwalk_poisson_likelihood(
        dist_fn,
        JointDistribution(Uniform(0.0, 2.0)) if priors is None else priors,
        {"log_scale": Uniform(0.0, 2.0)} if variables is None else variables,
        {} if constant_params is None else constant_params,
        {"log_scale": 0} if variables_index is None else variables_index,
        est,
        variance_cut_threshold,
    )


@pytest.mark.parametrize("log_scale", [0.1, 0.75, 1.9])
@pytest.mark.parametrize(("n_events", "n_samples"), [(1, 3), (3, 5)])
def test_log_density_is_prior_plus_likelihood(
    model_factory, estimator, analytical_data, log_scale, n_events, n_samples
):
    model = _build(model_factory(), estimator())
    data = analytical_data(n_events=n_events, n_samples=n_samples)
    value = jnp.asarray(log_scale)
    density, _ = log_density(model, _model_args(data), {}, {"log_scale": value})
    expected = n_events * (log_scale + np.log(n_samples)) + Uniform(0.0, 2.0).log_prob(
        value
    )
    chex.assert_trees_all_close(density, expected)


def test_trace_has_one_site_per_variable_plus_the_factor(
    model_factory, estimator, analytical_data
):
    variables = {"z_param": Uniform(0.0, 2.0), "a_param": Uniform(0.0, 2.0)}
    model = _build(model_factory(), estimator(), variables=variables)
    _, trace = log_density(
        model,
        _model_args(analytical_data()),
        {},
        {"a_param": jnp.asarray(0.75), "z_param": jnp.asarray(1.25)},
    )
    assert list(trace) == ["a_param", "z_param", "log_likelihood"]


def test_variables_index_points_into_the_sorted_variable_list(
    model_factory, estimator, analytical_data
):
    dist_fn = model_factory()
    variables = {"z_param": Uniform(0.0, 2.0), "a_param": Uniform(0.0, 2.0)}
    data = analytical_data(n_events=3, n_samples=5)
    params = {"a_param": jnp.asarray(0.25), "z_param": jnp.asarray(1.5)}
    prior_density = sum(
        float(Uniform(0.0, 2.0).log_prob(value)) for value in params.values()
    )
    for index, log_scale in (({"log_scale": 0}, 0.25), ({"log_scale": 1}, 1.5)):
        model = _build(dist_fn, estimator(), variables=variables, variables_index=index)
        density, _ = log_density(model, _model_args(data), {}, params)
        expected = 3 * (log_scale + np.log(5)) + prior_density
        chex.assert_trees_all_close(density, jnp.asarray(expected))


def test_constant_params_are_forwarded(model_factory, estimator, analytical_data):
    dist_fn = model_factory()
    model = _build(dist_fn, estimator(), constant_params={"mmin": jnp.asarray(5.0)})
    log_density(
        model, _model_args(analytical_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert dist_fn.last_call["mmin"] == 5.0


def test_model_is_built_without_validate_args(
    model_factory, estimator, analytical_data
):
    """Unlike the discrete NumPyro path, this one does not force ``validate_args=True``
    on the model.

    The resampled points are already filtered against the support inside the likelihood,
    so validation would be redundant.
    """
    dist_fn = model_factory()
    model = _build(dist_fn, estimator())
    log_density(
        model, _model_args(analytical_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert "validate_args" not in dist_fn.last_call


def test_estimator_receives_pmean_kwargs_from_the_model_args(
    model_factory, estimator, analytical_data
):
    est = estimator()
    model = _build(model_factory(), est)
    log_density(
        model,
        _model_args(analytical_data(T_obs=3.0)),
        {},
        {"log_scale": jnp.asarray(0.75)},
    )
    assert len(est.calls) == 1
    assert est.calls[0][1] == {"T_obs": 3.0}


###############################################################################
# sanitisation
###############################################################################


@pytest.mark.parametrize("expected_rate", [jnp.nan, -jnp.inf, jnp.inf])
def test_non_finite_likelihood_becomes_minus_inf(
    model_factory, estimator, analytical_data, expected_rate
):
    """The guard here is unconditional -- there is no ``where_fns`` to gate it."""
    model = _build(model_factory(), estimator(mean=expected_rate))
    density, _ = log_density(
        model, _model_args(analytical_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert bool(jnp.isneginf(density))


###############################################################################
# variance tapering
###############################################################################


def test_variance_cut_threshold_is_applied(model_factory, estimator, analytical_data):
    data = analytical_data(n_events=3, n_samples=5)
    params = {"log_scale": jnp.asarray(0.75)}
    est_variance, threshold = 4.0, 1.0
    plain, _ = log_density(
        _build(model_factory(), estimator(variance=est_variance)),
        _model_args(data),
        {},
        params,
    )
    tapered, _ = log_density(
        _build(
            model_factory(),
            estimator(variance=est_variance),
            variance_cut_threshold=threshold,
        ),
        _model_args(data),
        {},
        params,
    )
    chex.assert_trees_all_close(
        plain - tapered, jnp.asarray(100.0 * (est_variance - threshold) ** 2)
    )


def test_no_taper_below_threshold(model_factory, estimator, analytical_data):
    data = analytical_data(n_events=3, n_samples=5)
    params = {"log_scale": jnp.asarray(0.75)}
    plain, _ = log_density(
        _build(model_factory(), estimator(variance=0.25)),
        _model_args(data),
        {},
        params,
    )
    tapered, _ = log_density(
        _build(model_factory(), estimator(variance=0.25), variance_cut_threshold=1.0),
        _model_args(data),
        {},
        params,
    )
    chex.assert_trees_all_close(plain, tapered)


###############################################################################
# lazy priors
###############################################################################


def _lazy_priors():
    return LazyJointDistribution(
        Uniform(0.0, 1.0),
        jax.tree_util.Partial(Normal, scale=1.0),
        dependencies={1: {"loc": 0}},
        partial_order=[1],
    )


def _lazy_variables():
    return {
        "a": Uniform(0.0, 1.0),
        "b": jax.tree_util.Partial(Normal, scale=1.0),
    }


def test_lazy_prior_samples_the_dependent_distribution(
    model_factory, estimator, analytical_data
):
    model = _build(
        model_factory(),
        estimator(),
        priors=_lazy_priors(),
        variables=_lazy_variables(),
        variables_index={"log_scale": 1},
    )
    data = analytical_data(n_events=3, n_samples=5)
    params = {"a": jnp.asarray(0.3), "b": jnp.asarray(0.7)}
    density, trace = log_density(model, _model_args(data), {}, params)
    expected = (
        3 * (0.7 + np.log(5))
        + float(Uniform(0.0, 1.0).log_prob(jnp.asarray(0.3)))
        + float(Normal(0.3, 1.0).log_prob(jnp.asarray(0.7)))
    )
    chex.assert_trees_all_close(density, jnp.asarray(expected))
    assert list(trace) == ["a", "b", "log_likelihood"]


def test_lazy_prior_dependency_actually_moves_with_its_parent(
    model_factory, estimator, analytical_data
):
    model = _build(
        model_factory(),
        estimator(),
        priors=_lazy_priors(),
        variables=_lazy_variables(),
        variables_index={"log_scale": 1},
    )
    args = _model_args(analytical_data(n_events=3, n_samples=5))
    deltas = []
    for a in (0.1, 0.9):
        density, _ = log_density(
            model, args, {}, {"a": jnp.asarray(a), "b": jnp.asarray(0.7)}
        )
        deltas.append(density - float(Uniform(0.0, 1.0).log_prob(jnp.asarray(a))))
    expected = [
        3 * (0.7 + np.log(5)) + float(Normal(a, 1.0).log_prob(jnp.asarray(0.7)))
        for a in (0.1, 0.9)
    ]
    chex.assert_trees_all_close(jnp.asarray(deltas), jnp.asarray(expected))
