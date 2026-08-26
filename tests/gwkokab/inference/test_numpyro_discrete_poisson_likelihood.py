# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NumPyro discrete likelihood wrapper.

Unlike the flowMC wrapper this one returns a NumPyro *model*: it draws the population
hyper-parameters as sample sites and adds the Poisson log-likelihood as a factor. The
tests drive it through :func:`numpyro.infer.util.log_density`, which returns the total
log-density (priors + factor) together with the execution trace, so both the numbers and
the site structure can be checked.
"""

import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpyro.distributions import Normal, Uniform
from numpyro.infer.util import log_density

from gwkokab.inference import numpyro_discrete_poisson_likelihood
from gwkokab.models.utils import JointDistribution, LazyJointDistribution


def _model_args(data):
    return (
        data["data_group"],
        data["log_ref_priors_group"],
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
    )


def _build(
    dist_fn,
    est,
    *,
    priors=None,
    variables=None,
    variables_index=None,
    where_fns=None,
    constants=None,
    variance_cut_threshold=None,
):
    return numpyro_discrete_poisson_likelihood(
        dist_fn=dist_fn,
        priors=JointDistribution(Uniform(0.0, 2.0)) if priors is None else priors,
        variables={"log_scale": Uniform(0.0, 2.0)} if variables is None else variables,
        variables_index={"log_scale": 0}
        if variables_index is None
        else variables_index,
        poisson_mean_estimator=est,
        where_fns=where_fns,
        constants={} if constants is None else constants,
        variance_cut_threshold=variance_cut_threshold,
    )


@pytest.mark.parametrize("log_scale", [0.1, 0.75, 1.9])
@pytest.mark.parametrize(("n_events", "n_samples"), [(1, 3), (3, 5)])
def test_log_density_is_prior_plus_likelihood(
    model_factory, estimator, discrete_data, log_scale, n_events, n_samples
):
    model = _build(model_factory(), estimator())
    data = discrete_data(n_events=n_events, n_samples=n_samples)
    value = jnp.asarray(log_scale)
    density, _ = log_density(model, _model_args(data), {}, {"log_scale": value})
    expected = n_events * log_scale + Uniform(0.0, 2.0).log_prob(value)
    chex.assert_trees_all_close(density, expected)


def test_trace_has_one_site_per_variable_plus_the_factor(
    model_factory, estimator, discrete_data
):
    """Every entry of ``variables`` becomes a sample site; the likelihood is a single
    ``numpyro.factor`` named ``log_likelihood``.
    """
    variables = {"z_param": Uniform(0.0, 2.0), "a_param": Uniform(0.0, 2.0)}
    model = _build(
        model_factory(),
        estimator(),
        variables=variables,
        variables_index={"log_scale": 0},
    )
    _, trace = log_density(
        model,
        _model_args(discrete_data()),
        {},
        {"a_param": jnp.asarray(0.75), "z_param": jnp.asarray(1.25)},
    )
    assert list(trace) == ["a_param", "z_param", "log_likelihood"]


def test_variables_index_points_into_the_sorted_variable_list(
    model_factory, estimator, discrete_data
):
    """Sample sites are drawn in ``sorted(variables)`` order, and ``variables_index``
    indexes *that* list -- the same ordering the output HDF5 records.

    Inserting the variables in reverse order must not change anything.
    """
    dist_fn = model_factory()
    variables = {"z_param": Uniform(0.0, 2.0), "a_param": Uniform(0.0, 2.0)}
    data = discrete_data(n_events=3, n_samples=5)
    params = {"a_param": jnp.asarray(0.25), "z_param": jnp.asarray(1.5)}
    prior_density = sum(
        float(Uniform(0.0, 2.0).log_prob(value)) for value in params.values()
    )
    for index, log_scale in (({"log_scale": 0}, 0.25), ({"log_scale": 1}, 1.5)):
        model = _build(dist_fn, estimator(), variables=variables, variables_index=index)
        density, _ = log_density(model, _model_args(data), {}, params)
        chex.assert_trees_all_close(density, jnp.asarray(3 * log_scale + prior_density))


def test_constants_are_forwarded_to_the_model(model_factory, estimator, discrete_data):
    dist_fn = model_factory()
    model = _build(dist_fn, estimator(), constants={"mmin": jnp.asarray(5.0)})
    log_density(
        model, _model_args(discrete_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert dist_fn.last_call["mmin"] == 5.0


def test_model_is_built_with_validate_args(model_factory, estimator, discrete_data):
    """The discrete NumPyro path opts into NumPyro's argument validation.

    Sampling can wander into parameter regions a distribution cannot represent, and
    validation is what turns those into rejected proposals instead of silent NaNs.
    """
    dist_fn = model_factory()
    model = _build(dist_fn, estimator())
    log_density(
        model, _model_args(discrete_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert dist_fn.last_call["validate_args"] is True


def test_estimator_receives_pmean_kwargs_from_the_model_args(
    model_factory, estimator, discrete_data
):
    est = estimator()
    model = _build(model_factory(), est)
    log_density(
        model,
        _model_args(discrete_data(T_obs=3.0)),
        {},
        {"log_scale": jnp.asarray(0.75)},
    )
    assert len(est.calls) == 1
    assert set(est.calls[0][1]) == {"T_obs"}
    assert est.calls[0][1]["T_obs"] == 3.0


def test_multiple_buckets(model_factory, estimator):
    counts = [(2, 4), (3, 7)]
    data = {
        "data_group": tuple(jnp.full((e, k, 2), 0.5) for e, k in counts),
        "log_ref_priors_group": tuple(jnp.zeros((e, k)) for e, k in counts),
        "masks_group": tuple(jnp.ones((e, k), dtype=bool) for e, k in counts),
        "pmean_kwargs": {"T_obs": 2.0},
        "N_pes": tuple(jnp.full((e,), k) for e, k in counts),
    }
    model = _build(model_factory(), estimator())
    value = jnp.asarray(0.75)
    density, _ = log_density(model, _model_args(data), {}, {"log_scale": value})
    n_events = sum(e for e, _ in counts)
    expected = (
        n_events * 0.75
        + n_events * np.log(2.0)
        + float(Uniform(0.0, 2.0).log_prob(value))
    )
    chex.assert_trees_all_close(density, jnp.asarray(expected))


###############################################################################
# where_fns
###############################################################################


def test_where_fns_none_leaves_the_likelihood_untouched(
    model_factory, estimator, discrete_data
):
    data = discrete_data()
    params = {"log_scale": jnp.asarray(0.75)}
    plain, _ = log_density(
        _build(model_factory(), estimator()), _model_args(data), {}, params
    )
    guarded, _ = log_density(
        _build(model_factory(), estimator(), where_fns=[]),
        _model_args(data),
        {},
        params,
    )
    chex.assert_trees_all_close(plain, guarded)


def test_satisfied_where_fns_do_not_change_the_value(
    model_factory, estimator, discrete_data
):
    data = discrete_data()
    params = {"log_scale": jnp.asarray(0.75)}
    plain, _ = log_density(
        _build(model_factory(), estimator()), _model_args(data), {}, params
    )
    guarded, _ = log_density(
        _build(
            model_factory(),
            estimator(),
            where_fns=[lambda **kwargs: kwargs["log_scale"] < 2.0],
        ),
        _model_args(data),
        {},
        params,
    )
    chex.assert_trees_all_close(plain, guarded)


def test_failing_where_fn_gives_minus_inf(model_factory, estimator, discrete_data):
    model = _build(
        model_factory(),
        estimator(),
        where_fns=[lambda **kwargs: kwargs["log_scale"] < 0.5],
    )
    density, _ = log_density(
        model, _model_args(discrete_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert bool(jnp.isneginf(density))


def test_where_fns_are_combined_with_logical_and(
    model_factory, estimator, discrete_data
):
    data = discrete_data()
    params = {"log_scale": jnp.asarray(0.75)}
    passing = [lambda **kwargs: kwargs["log_scale"] > 0.0]
    failing = [lambda **kwargs: kwargs["log_scale"] > 1.0]
    for where_fns in (passing + failing, failing + passing):
        model = _build(model_factory(), estimator(), where_fns=where_fns)
        density, _ = log_density(model, _model_args(data), {}, params)
        assert bool(jnp.isneginf(density))


def test_where_fns_receive_constants(model_factory, estimator, discrete_data):
    seen = {}

    def where_fn(**kwargs):
        seen.update(kwargs)
        return jnp.asarray(True)

    model = _build(
        model_factory(),
        estimator(),
        where_fns=[where_fn],
        constants={"mmin": jnp.asarray(5.0)},
    )
    log_density(
        model, _model_args(discrete_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert set(seen) == {"mmin", "log_scale"}


def test_where_fns_sanitise_a_nan_likelihood(model_factory, estimator, discrete_data):
    """The NaN guard lives inside the ``where_fns`` branch of this wrapper."""
    model = _build(
        model_factory(),
        estimator(mean=jnp.nan),
        where_fns=[lambda **kwargs: jnp.asarray(True)],
    )
    density, _ = log_density(
        model, _model_args(discrete_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert bool(jnp.isneginf(density))


def test_nan_likelihood_survives_without_where_fns(
    model_factory, estimator, discrete_data
):
    """Without ``where_fns`` there is no sanitising step, so a NaN propagates.

    NumPyro's samplers reject NaN potential energies themselves, so this is the
    wrapper's actual contract rather than an oversight worth papering over in a test.
    """
    model = _build(model_factory(), estimator(mean=jnp.nan))
    density, _ = log_density(
        model, _model_args(discrete_data()), {}, {"log_scale": jnp.asarray(0.75)}
    )
    assert bool(jnp.isnan(density))


###############################################################################
# variance tapering
###############################################################################


def test_variance_cut_threshold_is_applied(model_factory, estimator, discrete_data):
    data = discrete_data(n_events=3, n_samples=5)
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


###############################################################################
# lazy priors
###############################################################################


def _lazy_priors():
    """``b ~ Normal(loc=a, scale=1)`` with ``a ~ Uniform(0, 1)``."""
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
    model_factory, estimator, discrete_data
):
    """A lazy prior's hyper-parameters are themselves sampled sites.

    Here ``b``'s location is the sampled value of ``a``, so the total log-density must
    contain ``Normal(a, 1).log_prob(b)`` -- not the log-density of any fixed Normal.
    """
    model = _build(
        model_factory(),
        estimator(),
        priors=_lazy_priors(),
        variables=_lazy_variables(),
        variables_index={"log_scale": 1},
    )
    data = discrete_data(n_events=3, n_samples=5)
    params = {"a": jnp.asarray(0.3), "b": jnp.asarray(0.7)}
    density, trace = log_density(model, _model_args(data), {}, params)
    expected = (
        3 * 0.7
        + float(Uniform(0.0, 1.0).log_prob(jnp.asarray(0.3)))
        + float(Normal(0.3, 1.0).log_prob(jnp.asarray(0.7)))
    )
    chex.assert_trees_all_close(density, jnp.asarray(expected))
    assert list(trace) == ["a", "b", "log_likelihood"]


def test_lazy_prior_dependency_actually_moves_with_its_parent(
    model_factory, estimator, discrete_data
):
    """Changing the parent must change the child's prior density.

    A wrapper that ignored ``dependencies`` and instantiated the Partial with its
    default arguments would return the same number for both parents.
    """
    model = _build(
        model_factory(),
        estimator(),
        priors=_lazy_priors(),
        variables=_lazy_variables(),
        variables_index={"log_scale": 1},
    )
    args = _model_args(discrete_data(n_events=3, n_samples=5))
    densities = [
        log_density(model, args, {}, {"a": jnp.asarray(a), "b": jnp.asarray(0.7)})[0]
        for a in (0.1, 0.9)
    ]
    deltas = [
        density - float(Uniform(0.0, 1.0).log_prob(jnp.asarray(a)))
        for density, a in zip(densities, (0.1, 0.9))
    ]
    expected = [
        3 * 0.7 + float(Normal(a, 1.0).log_prob(jnp.asarray(0.7))) for a in (0.1, 0.9)
    ]
    chex.assert_trees_all_close(jnp.asarray(deltas), jnp.asarray(expected))


def test_lazy_prior_respects_where_fns(model_factory, estimator, discrete_data):
    model = _build(
        model_factory(),
        estimator(),
        priors=_lazy_priors(),
        variables=_lazy_variables(),
        variables_index={"log_scale": 1},
        where_fns=[lambda **kwargs: kwargs["log_scale"] < 0.0],
    )
    density, _ = log_density(
        model,
        _model_args(discrete_data()),
        {},
        {"a": jnp.asarray(0.3), "b": jnp.asarray(0.7)},
    )
    assert bool(jnp.isneginf(density))
