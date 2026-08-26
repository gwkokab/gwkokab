# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flowMC discrete log-posterior wrapper.

flowMC hands the log-posterior a flat parameter vector plus a data dict, and expects a
finite scalar or ``-inf`` back — never a NaN, which would poison every walker. The
tests below pin down the three jobs the wrapper does on top of
:func:`~gwkokab.inference.poissonlikelihood_utils.discrete_poisson_likelihood_fn`:
mapping the vector onto model keyword arguments, adding the prior, and sanitising the
result.
"""

import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpyro.distributions import Normal, Uniform

from gwkokab.inference import flowMC_discrete_poisson_likelihood
from gwkokab.models.utils import JointDistribution


def _build(
    model_factory,
    estimator,
    *,
    priors=None,
    variables_index=None,
    where_fns=None,
    constants=None,
    variance_cut_threshold=None,
    dist_fn=None,
    est=None,
):
    priors = JointDistribution(Uniform(0.0, 2.0)) if priors is None else priors
    dist_fn = model_factory() if dist_fn is None else dist_fn
    est = estimator() if est is None else est
    log_posterior = flowMC_discrete_poisson_likelihood(
        dist_fn=dist_fn,
        priors=priors,
        variables=None,
        variables_index={"log_scale": 0}
        if variables_index is None
        else variables_index,
        poisson_mean_estimator=est,
        where_fns=where_fns,
        constants={} if constants is None else constants,
        variance_cut_threshold=variance_cut_threshold,
    )
    return log_posterior, dist_fn, est, priors


@pytest.mark.parametrize("log_scale", [-1.0, 0.0, 0.75, 1.9])
def test_log_posterior_is_prior_plus_likelihood(
    model_factory, estimator, discrete_data, log_scale
):
    """The wrapper adds ``priors.log_prob(x)`` to the closed-form likelihood."""
    log_posterior, _, _, priors = _build(model_factory, estimator)
    data = discrete_data(n_events=3, n_samples=5)
    x = jnp.asarray([log_scale])
    expected = 3 * log_scale + priors.log_prob(x)
    chex.assert_trees_all_close(log_posterior(x, data), expected)


def test_variables_index_selects_the_right_component(
    model_factory, estimator, discrete_data
):
    """``variables_index`` is the only thing tying a vector slot to a name.

    The mapping is checked through the returned value rather than through the recorded
    arguments, because the wrapper is JIT-compiled and the recorded arguments are
    tracers. With a uniform model the log-likelihood is exactly ``n_events *
    log_scale``, so the slot that was read is directly readable off the result.
    """
    dist_fn = model_factory()
    priors = JointDistribution(Uniform(0.0, 2.0), Uniform(0.0, 2.0))
    data = discrete_data(n_events=3, n_samples=5)
    x = jnp.asarray([0.25, 1.5])
    for variables_index, log_scale in (
        ({"unused": 0, "log_scale": 1}, 1.5),
        ({"log_scale": 0, "unused": 1}, 0.25),
    ):
        log_posterior = flowMC_discrete_poisson_likelihood(
            dist_fn=dist_fn,
            priors=priors,
            variables=None,
            variables_index=variables_index,
            poisson_mean_estimator=estimator(),
            where_fns=None,
            constants={},
            variance_cut_threshold=None,
        )
        chex.assert_trees_all_close(
            log_posterior(x, data), 3 * log_scale + priors.log_prob(x)
        )
    assert set(dist_fn.last_call) == {"unused", "log_scale"}


def test_constants_are_forwarded_to_the_model(model_factory, estimator, discrete_data):
    """Frozen parameters reach ``dist_fn`` alongside the sampled ones."""
    dist_fn = model_factory()
    constants = {"mmin": jnp.asarray(5.0), "mmax": jnp.asarray(50.0)}
    log_posterior, _, _, _ = _build(
        model_factory, estimator, dist_fn=dist_fn, constants=constants
    )
    log_posterior(jnp.asarray([0.75]), discrete_data())
    recorded = dist_fn.last_call
    assert recorded["mmin"] == 5.0
    assert recorded["mmax"] == 50.0


def test_variables_argument_is_unused(model_factory, estimator, discrete_data):
    """The wrapper deletes ``variables`` outright; passing junk must not matter."""
    data = discrete_data()
    results = [
        flowMC_discrete_poisson_likelihood(
            dist_fn=model_factory(),
            priors=JointDistribution(Uniform(0.0, 2.0)),
            variables=variables,
            variables_index={"log_scale": 0},
            poisson_mean_estimator=estimator(),
            where_fns=None,
            constants={},
            variance_cut_threshold=None,
        )(jnp.asarray([0.75]), data)
        for variables in (None, {}, {"log_scale": Normal(0.0, 1.0)}, "junk")
    ]
    chex.assert_trees_all_close(results[0], *results[1:])


def test_estimator_receives_pmean_kwargs_from_the_data_dict(
    model_factory, estimator, discrete_data
):
    """``pmean_kwargs`` travels in the data payload, not in the closure."""
    est = estimator()
    log_posterior, _, _, _ = _build(model_factory, estimator, est=est)
    log_posterior(jnp.asarray([0.75]), discrete_data(T_obs=3.0))
    assert len(est.calls) == 1
    assert set(est.calls[0][1]) == {"T_obs"}


###############################################################################
# sanitisation
###############################################################################


def test_nan_log_posterior_becomes_minus_inf(model_factory, estimator, discrete_data):
    """A NaN would silently corrupt every flowMC chain, so it is mapped to ``-inf``."""
    log_posterior, _, _, _ = _build(
        model_factory, estimator, est=estimator(mean=jnp.nan)
    )
    out = log_posterior(jnp.asarray([0.75]), discrete_data())
    assert bool(jnp.isneginf(out))


def test_positive_infinity_becomes_minus_inf(model_factory, estimator, discrete_data):
    """``+inf`` is rejected too: an infinitely good point would trap the sampler."""
    log_posterior, _, _, _ = _build(
        model_factory, estimator, est=estimator(mean=-jnp.inf)
    )
    out = log_posterior(jnp.asarray([0.75]), discrete_data())
    assert bool(jnp.isneginf(out))


def test_negative_infinity_is_preserved(model_factory, estimator, discrete_data):
    log_posterior, _, _, _ = _build(
        model_factory, estimator, est=estimator(mean=jnp.inf)
    )
    out = log_posterior(jnp.asarray([0.75]), discrete_data())
    assert bool(jnp.isneginf(out))


###############################################################################
# where_fns
###############################################################################


def test_where_fns_none_skips_the_support_check(
    model_factory, estimator, discrete_data
):
    """Without ``where_fns`` there is no guard at all — not even on the prior support.

    NumPyro's ``Uniform.log_prob`` returns a finite constant outside ``[low, high]``, so
    an out-of-support proposal survives when no ``where_fns`` are supplied. This is
    exactly why the analyses pass ``priors.support.check`` through ``where_fns``.
    """
    log_posterior, _, _, _ = _build(model_factory, estimator, where_fns=None)
    out = log_posterior(jnp.asarray([5.0]), discrete_data())
    assert bool(jnp.isfinite(out))


def test_where_fns_reject_out_of_prior_support(model_factory, estimator, discrete_data):
    """With ``where_fns`` supplied, the prior support *is* enforced."""
    log_posterior, _, _, _ = _build(
        model_factory, estimator, where_fns=[lambda **kwargs: jnp.asarray(True)]
    )
    assert bool(jnp.isneginf(log_posterior(jnp.asarray([5.0]), discrete_data())))
    assert bool(jnp.isfinite(log_posterior(jnp.asarray([0.75]), discrete_data())))


def test_satisfied_where_fns_do_not_change_the_value(
    model_factory, estimator, discrete_data
):
    data = discrete_data()
    x = jnp.asarray([0.75])
    plain, _, _, _ = _build(model_factory, estimator, where_fns=None)
    guarded, _, _, _ = _build(
        model_factory,
        estimator,
        where_fns=[lambda **kwargs: kwargs["log_scale"] < 2.0],
    )
    chex.assert_trees_all_close(guarded(x, data), plain(x, data))


def test_failing_where_fn_gives_minus_inf(model_factory, estimator, discrete_data):
    log_posterior, _, _, _ = _build(
        model_factory,
        estimator,
        where_fns=[lambda **kwargs: kwargs["log_scale"] < 0.5],
    )
    assert bool(jnp.isneginf(log_posterior(jnp.asarray([0.75]), discrete_data())))


def test_where_fns_are_combined_with_logical_and(
    model_factory, estimator, discrete_data
):
    """Every predicate must hold; one failure is enough to reject the point."""
    data = discrete_data()
    x = jnp.asarray([0.75])
    passing = [lambda **kwargs: kwargs["log_scale"] > 0.0]
    failing = [lambda **kwargs: kwargs["log_scale"] > 1.0]
    both, _, _, _ = _build(model_factory, estimator, where_fns=passing + failing)
    reversed_order, _, _, _ = _build(
        model_factory, estimator, where_fns=failing + passing
    )
    assert bool(jnp.isneginf(both(x, data)))
    assert bool(jnp.isneginf(reversed_order(x, data)))


def test_where_fns_receive_constants(model_factory, estimator, discrete_data):
    """Predicates are called with constants *and* mapped parameters."""
    seen = {}

    def where_fn(**kwargs):
        seen.update(kwargs)
        return jnp.asarray(True)

    log_posterior, _, _, _ = _build(
        model_factory,
        estimator,
        where_fns=[where_fn],
        constants={"mmin": jnp.asarray(5.0)},
    )
    log_posterior(jnp.asarray([0.75]), discrete_data())
    assert set(seen) == {"mmin", "log_scale"}


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_non_finite_input_is_rejected(
    model_factory, estimator, discrete_data, bad_value
):
    """FlowMC can propose non-finite positions; those must never be accepted."""
    log_posterior, _, _, _ = _build(
        model_factory, estimator, where_fns=[lambda **kwargs: jnp.asarray(True)]
    )
    out = log_posterior(jnp.asarray([bad_value]), discrete_data())
    assert bool(jnp.isneginf(out))


###############################################################################
# transformations
###############################################################################


def test_variance_cut_threshold_is_applied(model_factory, estimator, discrete_data):
    """The threshold is baked into the closure, not read from the data dict."""
    data = discrete_data(n_events=3, n_samples=5)
    x = jnp.asarray([0.75])
    est_variance, threshold = 4.0, 1.0
    plain, _, _, priors = _build(
        model_factory, estimator, est=estimator(variance=est_variance)
    )
    tapered, _, _, _ = _build(
        model_factory,
        estimator,
        est=estimator(variance=est_variance),
        variance_cut_threshold=threshold,
    )
    chex.assert_trees_all_close(
        plain(x, data) - tapered(x, data),
        jnp.asarray(100.0 * (est_variance - threshold) ** 2),
    )


def test_is_vmappable_over_walkers(model_factory, estimator, discrete_data):
    """FlowMC evaluates a whole ensemble at once."""
    log_posterior, _, _, priors = _build(model_factory, estimator)
    data = discrete_data(n_events=3, n_samples=5)
    positions = jnp.asarray([[0.25], [0.75], [1.5]])
    out = jax.vmap(log_posterior, in_axes=(0, None))(positions, data)
    expected = jnp.asarray([
        3 * float(p) + float(priors.log_prob(jnp.asarray([p]))) for p in positions[:, 0]
    ])
    chex.assert_trees_all_close(out, expected)


def test_is_differentiable(model_factory, estimator, discrete_data):
    r"""FlowMC's MALA kernel needs :math:`\nabla_x \log p`.

    For the uniform model the likelihood gradient is ``N`` and the uniform prior
    contributes nothing.
    """
    log_posterior, _, _, _ = _build(model_factory, estimator)
    data = discrete_data(n_events=3, n_samples=5)
    grad = jax.grad(log_posterior)(jnp.asarray([0.75]), data)
    chex.assert_trees_all_close(grad, jnp.asarray([3.0]))


def test_result_is_a_scalar(model_factory, estimator, discrete_data):
    log_posterior, _, _, _ = _build(model_factory, estimator)
    out = log_posterior(jnp.asarray([0.75]), discrete_data())
    chex.assert_shape(out, ())


def test_multiple_buckets(model_factory, estimator):
    """Ragged posterior-sample counts are grouped into buckets by the data loader."""
    log_posterior, _, _, priors = _build(model_factory, estimator)
    counts = [(2, 4), (3, 7)]
    data = {
        "data_group": tuple(jnp.full((e, k, 2), 0.5) for e, k in counts),
        "log_ref_priors_group": tuple(jnp.zeros((e, k)) for e, k in counts),
        "masks_group": tuple(jnp.ones((e, k), dtype=bool) for e, k in counts),
        "pmean_kwargs": {"T_obs": 2.0},
        "N_pes": tuple(jnp.full((e,), k) for e, k in counts),
    }
    x = jnp.asarray([0.75])
    n_events = sum(e for e, _ in counts)
    expected = n_events * 0.75 + n_events * np.log(2.0) + priors.log_prob(x)
    chex.assert_trees_all_close(log_posterior(x, data), expected)
