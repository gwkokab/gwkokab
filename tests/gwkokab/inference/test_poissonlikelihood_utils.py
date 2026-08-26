# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure-math core of the inference module.

Two kinds of assertion appear here. Where the model is uniform on the unit hyper-cube
every Monte-Carlo sum collapses and the log-likelihood is known exactly, so those tests
compare against a hand-derived formula. Where the model is Gaussian the sums no longer
collapse, so the tests compare against a deliberately naive NumPy transcription of the
equations in the module docstring — a second implementation that shares no code with the
one under test.
"""

import chex
import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpyro.distributions import LogNormal, Uniform
from scipy.special import logsumexp

from gwkokab.inference.poissonlikelihood_utils import (
    analytical_gwalk_poisson_likelihood_fn,
    discrete_poisson_likelihood_fn,
    variance_tapering_fn,
)
from gwkokab.models.utils import JointDistribution, ScaledMixture


def _reference_discrete(
    model, data_group, log_ref_priors_group, masks_group, N_pes, T_obs, expected_rate
):
    r"""Naive NumPy transcription of the discrete Poisson log-likelihood.

    Implements :math:`-\mu + \sum_n \log \sum_k p(\omega_k|\Lambda)/\pi_k - \sum_n \log
    M_n + N\log T_{\mathrm{obs}}` together with the per-event effective sample-size
    variance, without any of the masking gymnastics of the real implementation.
    """
    total = -sum(np.log(np.asarray(N_pe, dtype=float)).sum() for N_pe in N_pes)
    variance = 0.0
    n_events = 0
    for data, log_ref, mask, N_pe in zip(
        data_group, log_ref_priors_group, masks_group, N_pes
    ):
        n_events += data.shape[0]
        log_prob = np.asarray(model.log_prob(data)) - np.asarray(log_ref)
        log_prob = np.where(np.asarray(mask), log_prob, -np.inf)
        log_prob_sum = logsumexp(log_prob, axis=-1)
        log_prob_sum_2 = logsumexp(2.0 * log_prob, axis=-1)
        total += log_prob_sum.sum()
        variance += (
            np.exp(log_prob_sum_2 - 2.0 * log_prob_sum)
            - 1.0 / np.asarray(N_pe, dtype=float)
        ).sum()
    return total - expected_rate + n_events * np.log(T_obs), variance


def _reference_analytical(model, samples_stack, ln_offsets, T_obs, expected_rate):
    """Naive NumPy transcription of the analytical-GWalk Poisson log-likelihood."""
    mask = np.asarray(model.support.check(samples_stack))
    log_prob = np.asarray(model.log_prob(samples_stack)) + np.asarray(ln_offsets)
    log_prob = np.where(mask, log_prob, -np.inf)
    total = logsumexp(log_prob, axis=1).sum()
    n_events = samples_stack.shape[0]
    return total + n_events * np.log(T_obs) - expected_rate


###############################################################################
# variance_tapering_fn
###############################################################################


@pytest.mark.parametrize(
    ("variance", "threshold", "expected"),
    [
        (0.0, 1.0, 0.0),  # well below
        (0.999, 1.0, 0.0),  # just below
        (1.0, 1.0, 0.0),  # exactly at the threshold: no penalty
        (2.0, 1.0, 100.0),  # one unit above
        (4.0, 1.0, 900.0),
        (1.5, 0.5, 100.0),
        (10.0, 0.0, 10_000.0),
    ],
)
def test_variance_tapering_values(variance, threshold, expected):
    """The taper is zero below the threshold and quadratic above it."""
    chex.assert_trees_all_close(
        variance_tapering_fn(jnp.asarray(variance), jnp.asarray(threshold)),
        jnp.asarray(expected),
    )


def test_variance_tapering_is_elementwise():
    """The taper must act elementwise rather than on some reduction."""
    variance = jnp.asarray([0.0, 0.5, 1.0, 2.0, 3.0])
    threshold = jnp.asarray(1.0)
    chex.assert_trees_all_close(
        variance_tapering_fn(variance, threshold),
        jnp.asarray([0.0, 0.0, 0.0, 100.0, 400.0]),
    )


def test_variance_tapering_broadcasts_threshold():
    """A per-element threshold broadcasts against a per-element variance."""
    variance = jnp.asarray([2.0, 2.0, 2.0])
    threshold = jnp.asarray([1.0, 2.0, 3.0])
    chex.assert_trees_all_close(
        variance_tapering_fn(variance, threshold), jnp.asarray([100.0, 0.0, 0.0])
    )


def test_variance_tapering_is_non_negative():
    """A negative taper would *reward* a high-variance estimate."""
    variance = jnp.linspace(-5.0, 20.0, 101)
    penalty = variance_tapering_fn(variance, jnp.asarray(3.0))
    assert bool(jnp.all(penalty >= 0.0))


def test_variance_tapering_is_monotone_above_threshold():
    variance = jnp.linspace(3.0, 20.0, 51)
    penalty = variance_tapering_fn(variance, jnp.asarray(3.0))
    assert bool(jnp.all(jnp.diff(penalty) >= 0.0))


@pytest.mark.parametrize(
    ("variance", "expected_grad"),
    [(0.5, 0.0), (2.0, 200.0), (3.0, 400.0)],
)
def test_variance_tapering_gradient(variance, expected_grad):
    """The penalty is differentiable and flat below the threshold."""
    grad = jax.grad(variance_tapering_fn)(jnp.asarray(variance), jnp.asarray(1.0))
    chex.assert_trees_all_close(grad, jnp.asarray(expected_grad))


def test_variance_tapering_infinite_variance():
    """An infinite variance yields an infinite penalty, which drives ``logL`` to
    ``-inf`` in the callers.
    """
    assert jnp.isinf(variance_tapering_fn(jnp.asarray(jnp.inf), jnp.asarray(1.0)))


def test_variance_tapering_jit():
    jitted = jax.jit(variance_tapering_fn)
    chex.assert_trees_all_close(
        jitted(jnp.asarray(4.0), jnp.asarray(1.0)), jnp.asarray(900.0)
    )


###############################################################################
# discrete_poisson_likelihood_fn -- closed form
###############################################################################


@pytest.mark.parametrize("log_scale", [-2.0, 0.0, 0.75, 3.5])
@pytest.mark.parametrize(("n_events", "n_samples"), [(1, 1), (3, 5), (7, 4)])
def test_discrete_uniform_closed_form(
    mixture, estimator, discrete_data, log_scale, n_events, n_samples
):
    r"""For a uniform model with unit reference priors and full masks the whole
    expression collapses to :math:`N \log s`.

    Each of the ``K`` samples contributes the same ``log_scale``, so ``logsumexp`` gives
    ``log_scale + log K`` per event, and the ``-sum(log N_pe)`` bookkeeping term cancels
    the ``log K`` exactly.
    """
    model = mixture([log_scale])
    data = discrete_data(n_events=n_events, n_samples=n_samples)
    out = discrete_poisson_likelihood_fn(
        model,
        estimator(),
        data["data_group"],
        data["log_ref_priors_group"],
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
        None,
    )
    chex.assert_trees_all_close(out, jnp.asarray(n_events * log_scale))


@pytest.mark.parametrize("expected_rate", [0.0, 2.5, 17.0])
def test_discrete_subtracts_expected_rate(
    mixture, estimator, discrete_data, expected_rate
):
    """The Poisson mean enters as a plain subtraction."""
    data = discrete_data(n_events=3, n_samples=5)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(mean=expected_rate),
        data["data_group"],
        data["log_ref_priors_group"],
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
        None,
    )
    chex.assert_trees_all_close(out, jnp.asarray(3 * 0.75 - expected_rate))


@pytest.mark.parametrize("T_obs", [0.5, 1.0, 4.0])
def test_discrete_observing_time_scaling(mixture, estimator, discrete_data, T_obs):
    r"""The observing time enters as :math:`N\log T_{\mathrm{obs}}`."""
    data = discrete_data(n_events=3, n_samples=5, T_obs=T_obs)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(),
        data["data_group"],
        data["log_ref_priors_group"],
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
        None,
    )
    chex.assert_trees_all_close(out, jnp.asarray(3 * 0.75 + 3 * np.log(T_obs)))


def test_discrete_event_count_spans_all_buckets(mixture, estimator):
    r"""``N`` in :math:`N\log T_{\mathrm{obs}}` counts events across *every* bucket.

    Buckets exist only to group events by posterior-sample count; they must not change
    the answer beyond the events they carry.
    """
    log_scale, T_obs = 0.5, 3.0
    counts = [(2, 4), (3, 6)]  # (n_events, n_samples) per bucket
    data_group = tuple(jnp.full((e, k, 2), 0.5) for e, k in counts)
    log_ref = tuple(jnp.zeros((e, k)) for e, k in counts)
    masks = tuple(jnp.ones((e, k), dtype=bool) for e, k in counts)
    N_pes = tuple(jnp.full((e,), k) for e, k in counts)
    out = discrete_poisson_likelihood_fn(
        mixture([log_scale]),
        estimator(),
        data_group,
        log_ref,
        masks,
        {"T_obs": T_obs},
        N_pes,
        None,
    )
    n_events = sum(e for e, _ in counts)
    chex.assert_trees_all_close(
        out, jnp.asarray(n_events * log_scale + n_events * np.log(T_obs))
    )


@pytest.mark.parametrize("log_ref_prior", [-1.5, 0.0, 2.0])
def test_discrete_divides_by_reference_prior(
    mixture, estimator, discrete_data, log_ref_prior
):
    """A constant reference prior shifts the log-likelihood by ``-N * log_pi``."""
    data = discrete_data(n_events=3, n_samples=5)
    log_ref = (jnp.full_like(data["log_ref_priors_group"][0], log_ref_prior),)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(),
        data["data_group"],
        log_ref,
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
        None,
    )
    chex.assert_trees_all_close(out, jnp.asarray(3 * (0.75 - log_ref_prior)))


def test_discrete_masked_samples_are_ignored(mixture, estimator):
    """Padding slots must not contribute, even when they hold non-finite garbage.

    ``pad_and_stack`` leaves whatever it likes in the padded tail; the likelihood
    relies on ``support.feasible_like`` to keep those entries out of the arithmetic.
    NaN padding is the sharpest test: any leak produces a NaN result.
    """
    n_events, n_samples, n_valid = 3, 6, 4
    padded = jnp.full((n_events, n_samples, 2), 0.5).at[:, n_valid:, :].set(jnp.nan)
    mask = jnp.ones((n_events, n_samples), dtype=bool).at[:, n_valid:].set(False)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(),
        (padded,),
        (jnp.zeros((n_events, n_samples)),),
        (mask,),
        {"T_obs": 1.0},
        (jnp.full((n_events,), n_valid),),
        None,
    )
    chex.assert_trees_all_close(out, jnp.asarray(n_events * 0.75))


def test_discrete_masking_matches_a_trimmed_batch(mixture, estimator):
    """Masking the tail is equivalent to never padding in the first place."""
    n_events, n_valid, n_pad = 4, 5, 3
    core = jnp.linspace(0.1, 0.9, n_events * n_valid * 2).reshape(n_events, n_valid, 2)
    padded = jnp.concatenate([core, jnp.zeros((n_events, n_pad, 2))], axis=1)
    mask = jnp.concatenate(
        [
            jnp.ones((n_events, n_valid), dtype=bool),
            jnp.zeros((n_events, n_pad), dtype=bool),
        ],
        axis=1,
    )
    model = mixture([0.75])
    kwargs = dict(pmean_kwargs={"T_obs": 1.0}, variance_cut_threshold=None)
    masked = discrete_poisson_likelihood_fn(
        model,
        estimator(),
        (padded,),
        (jnp.zeros((n_events, n_valid + n_pad)),),
        (mask,),
        kwargs["pmean_kwargs"],
        (jnp.full((n_events,), n_valid),),
        kwargs["variance_cut_threshold"],
    )
    trimmed = discrete_poisson_likelihood_fn(
        model,
        estimator(),
        (core,),
        (jnp.zeros((n_events, n_valid)),),
        (jnp.ones((n_events, n_valid), dtype=bool),),
        kwargs["pmean_kwargs"],
        (jnp.full((n_events,), n_valid),),
        kwargs["variance_cut_threshold"],
    )
    chex.assert_trees_all_close(masked, trimmed)


@pytest.mark.parametrize("n_pe", [1, 5, 100])
def test_discrete_normalises_by_sample_count(mixture, estimator, n_pe):
    r"""``N_pes`` enters the log-likelihood only as :math:`-\sum_n \log M_n`."""
    n_events, n_samples = 3, 5
    data = (jnp.full((n_events, n_samples, 2), 0.5),)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(),
        data,
        (jnp.zeros((n_events, n_samples)),),
        (jnp.ones((n_events, n_samples), dtype=bool),),
        {"T_obs": 1.0},
        (jnp.full((n_events,), n_pe),),
        None,
    )
    expected = n_events * (0.75 + np.log(n_samples) - np.log(n_pe))
    chex.assert_trees_all_close(out, jnp.asarray(expected))


def test_discrete_does_not_apply_a_support_mask(mixture, estimator):
    """The discrete form trusts ``masks_group``; it never calls ``support.check``.

    This is a real asymmetry with :func:`analytical_gwalk_poisson_likelihood_fn`, which
    *does* mask on the support. NumPyro's ``Uniform.log_prob`` returns its (finite)
    constant density everywhere, so an unmasked out-of-support sample is silently kept
    here — the caller is responsible for the mask.
    """
    n_events, n_samples = 2, 4
    in_support = jnp.full((n_events, n_samples, 2), 0.5)
    out_of_support = in_support.at[0].set(5.0)
    model = mixture([0.75])
    common = (
        estimator(),
        (jnp.zeros((n_events, n_samples)),),
        (jnp.ones((n_events, n_samples), dtype=bool),),
        {"T_obs": 1.0},
        (jnp.full((n_events,), n_samples),),
        None,
    )
    out = discrete_poisson_likelihood_fn(
        model, common[0], (out_of_support,), *common[1:]
    )
    reference = discrete_poisson_likelihood_fn(
        model, common[0], (in_support,), *common[1:]
    )
    chex.assert_trees_all_close(out, reference)
    chex.assert_trees_all_close(out, jnp.asarray(n_events * 0.75))


def test_discrete_event_with_zero_weight_gives_minus_inf(mixture, estimator):
    """An event whose every sample carries zero weight kills the whole run.

    An infinite reference prior is the cleanest way to drive one event's importance
    weights to zero without touching the model.
    """
    n_events, n_samples = 2, 4
    log_ref = jnp.zeros((n_events, n_samples)).at[0].set(jnp.inf)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(),
        (jnp.full((n_events, n_samples, 2), 0.5),),
        (log_ref,),
        (jnp.ones((n_events, n_samples), dtype=bool),),
        {"T_obs": 1.0},
        (jnp.full((n_events,), n_samples),),
        None,
    )
    assert bool(jnp.isneginf(out))


def test_discrete_passes_model_and_kwargs_to_estimator(mixture, estimator):
    """The estimator receives the *instantiated* model plus every pmean kwarg."""
    model = mixture([0.75])
    est = estimator()
    pmean_kwargs = {"T_obs": 2.0, "extra_knob": jnp.asarray(7.0)}
    discrete_poisson_likelihood_fn(
        model,
        est,
        (jnp.full((2, 3, 2), 0.5),),
        (jnp.zeros((2, 3)),),
        (jnp.ones((2, 3), dtype=bool),),
        pmean_kwargs,
        (jnp.full((2,), 3),),
        None,
    )
    assert len(est.calls) == 1
    recorded_model, recorded_kwargs = est.calls[0]
    assert recorded_model is model
    assert set(recorded_kwargs) == {"T_obs", "extra_knob"}
    assert recorded_kwargs["T_obs"] == 2.0


def test_discrete_requires_t_obs(mixture, estimator, discrete_data):
    """``T_obs`` is part of the pmean-kwargs contract, not an optional extra."""
    data = discrete_data()
    with pytest.raises(KeyError, match="T_obs"):
        discrete_poisson_likelihood_fn(
            mixture([0.75]),
            estimator(),
            data["data_group"],
            data["log_ref_priors_group"],
            data["masks_group"],
            {},
            data["N_pes"],
            None,
        )


###############################################################################
# discrete_poisson_likelihood_fn -- variance tapering
###############################################################################


def test_discrete_identical_samples_have_zero_pe_variance(
    mixture, estimator, discrete_data
):
    """With identical samples the effective sample size is exactly ``N_pe``, so the per-
    event variance term vanishes and the taper sees only the estimator variance.
    """
    data = discrete_data(n_events=3, n_samples=5)
    threshold = 1.0
    est_variance = 4.0
    tapered = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(variance=est_variance),
        data["data_group"],
        data["log_ref_priors_group"],
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
        threshold,
    )
    expected = 3 * 0.75 - 100.0 * (est_variance - threshold) ** 2
    chex.assert_trees_all_close(tapered, jnp.asarray(expected))


def test_discrete_no_taper_below_threshold(mixture, estimator, discrete_data):
    data = discrete_data(n_events=3, n_samples=5)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(variance=0.25),
        data["data_group"],
        data["log_ref_priors_group"],
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
        1.0,
    )
    chex.assert_trees_all_close(out, jnp.asarray(3 * 0.75))


def test_discrete_taper_disabled_when_threshold_is_none(
    mixture, estimator, discrete_data
):
    """A huge estimator variance is ignored entirely when no threshold is set."""
    data = discrete_data(n_events=3, n_samples=5)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(variance=1e6),
        data["data_group"],
        data["log_ref_priors_group"],
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
        None,
    )
    chex.assert_trees_all_close(out, jnp.asarray(3 * 0.75))


def test_discrete_non_finite_variance_is_tapered_to_minus_inf(
    mixture, estimator, discrete_data
):
    """A NaN estimator variance must not leak into the log-likelihood as a NaN.

    The implementation maps every non-finite variance to ``+inf`` before tapering, which
    turns the penalty into ``+inf`` and the log-likelihood into ``-inf`` — a proposal
    the sampler simply rejects.
    """
    data = discrete_data(n_events=3, n_samples=5)
    out = discrete_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(variance=jnp.nan),
        data["data_group"],
        data["log_ref_priors_group"],
        data["masks_group"],
        data["pmean_kwargs"],
        data["N_pes"],
        1.0,
    )
    assert bool(jnp.isneginf(out))


###############################################################################
# discrete_poisson_likelihood_fn -- against an independent implementation
###############################################################################


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("variance_cut_threshold", [None, 0.05, 1e3])
def test_discrete_matches_reference_implementation(
    normal_mixture, estimator, seed, variance_cut_threshold
):
    """Random Gaussian model, ragged buckets, ragged masks, non-trivial priors."""
    key = jax.random.key(seed)
    keys = jax.random.split(key, 6)
    model = normal_mixture(
        jax.random.uniform(keys[0], (2,), minval=-1.0, maxval=1.0),
        locs=jax.random.normal(keys[1], (2, 3)),
        scales=jax.random.uniform(keys[2], (2, 3), minval=0.5, maxval=2.0),
    )

    shapes = [(4, 6), (2, 9)]
    data_group, log_ref_group, masks_group, N_pes = [], [], [], []
    for i, (n_events, n_samples) in enumerate(shapes):
        sub = jax.random.split(keys[3 + i], 3)
        data_group.append(jax.random.normal(sub[0], (n_events, n_samples, 3)))
        log_ref_group.append(
            jax.random.uniform(sub[1], (n_events, n_samples), minval=-1.0, maxval=1.0)
        )
        mask = jax.random.bernoulli(sub[2], 0.75, (n_events, n_samples))
        # every event must retain at least one live sample
        mask = mask.at[:, 0].set(True)
        masks_group.append(mask)
        N_pes.append(jnp.count_nonzero(mask, axis=-1))

    data_group = tuple(data_group)
    log_ref_group = tuple(log_ref_group)
    masks_group = tuple(masks_group)
    N_pes = tuple(N_pes)

    expected_rate, expected_variance = 3.25, 0.5
    T_obs = 2.0
    out = discrete_poisson_likelihood_fn(
        model,
        estimator(mean=expected_rate, variance=expected_variance),
        data_group,
        log_ref_group,
        masks_group,
        {"T_obs": T_obs},
        N_pes,
        variance_cut_threshold,
    )

    reference, pe_variance = _reference_discrete(
        model, data_group, log_ref_group, masks_group, N_pes, T_obs, expected_rate
    )
    if variance_cut_threshold is not None:
        total_variance = pe_variance + expected_variance
        reference -= 100.0 * max(total_variance - variance_cut_threshold, 0.0) ** 2

    chex.assert_trees_all_close(out, jnp.asarray(reference), rtol=1e-6)


def test_discrete_reference_case_exercises_the_taper(normal_mixture, estimator):
    """Guards the test above: at a small threshold the taper must actually bite."""
    model = normal_mixture(
        [0.0, 0.5], locs=[[0.0] * 3, [1.0] * 3], scales=[[1.0] * 3] * 2
    )
    data = (jax.random.normal(jax.random.key(0), (4, 6, 3)),)
    log_ref = (jnp.zeros((4, 6)),)
    masks = (jnp.ones((4, 6), dtype=bool),)
    N_pes = (jnp.full((4,), 6),)
    common = (model, estimator(), data, log_ref, masks, {"T_obs": 1.0}, N_pes)
    assert discrete_poisson_likelihood_fn(
        *common, 1e-6
    ) < discrete_poisson_likelihood_fn(*common, None)


###############################################################################
# discrete_poisson_likelihood_fn -- transformations
###############################################################################


def test_discrete_gradient_wrt_log_scales(estimator, discrete_data):
    r"""For the uniform model, :math:`\partial \log L/\partial s = N`."""
    n_events = 3
    data = discrete_data(n_events=n_events, n_samples=5)
    est = estimator()

    def log_likelihood(log_scales):
        model = ScaledMixture(
            log_scales, [JointDistribution(Uniform(0.0, 1.0), Uniform(0.0, 1.0))]
        )
        return discrete_poisson_likelihood_fn(
            model,
            est,
            data["data_group"],
            data["log_ref_priors_group"],
            data["masks_group"],
            data["pmean_kwargs"],
            data["N_pes"],
            None,
        )

    grad = jax.grad(log_likelihood)(jnp.asarray([0.75]))
    chex.assert_trees_all_close(grad, jnp.asarray([float(n_events)]))


def test_discrete_is_jittable(mixture, estimator, discrete_data):
    data = discrete_data(n_events=3, n_samples=5)
    est = estimator()

    @jax.jit
    def log_likelihood(log_scales):
        model = ScaledMixture(
            log_scales, [JointDistribution(Uniform(0.0, 1.0), Uniform(0.0, 1.0))]
        )
        return discrete_poisson_likelihood_fn(
            model,
            est,
            data["data_group"],
            data["log_ref_priors_group"],
            data["masks_group"],
            data["pmean_kwargs"],
            data["N_pes"],
            None,
        )

    chex.assert_trees_all_close(
        log_likelihood(jnp.asarray([0.75])), jnp.asarray(3 * 0.75)
    )


###############################################################################
# analytical_gwalk_poisson_likelihood_fn
###############################################################################


@pytest.mark.parametrize("log_scale", [-2.0, 0.0, 0.75])
@pytest.mark.parametrize(("n_events", "n_samples"), [(1, 1), (3, 5), (6, 8)])
def test_analytical_uniform_closed_form(
    mixture, estimator, analytical_data, log_scale, n_events, n_samples
):
    r"""Unlike the discrete form there is no :math:`-\log M_n` term, so the ``log K``
    from the resampling sum survives into the answer.
    """
    data = analytical_data(n_events=n_events, n_samples=n_samples)
    out = analytical_gwalk_poisson_likelihood_fn(
        mixture([log_scale]),
        estimator(),
        data["samples_stack"],
        data["ln_offsets"],
        data["pmean_kwargs"],
        None,
    )
    expected = n_events * (log_scale + np.log(n_samples))
    chex.assert_trees_all_close(out, jnp.asarray(expected))


@pytest.mark.parametrize("offset", [-2.0, 0.0, 1.5])
def test_analytical_ln_offsets_shift_the_result(
    mixture, estimator, analytical_data, offset
):
    """``ln_offsets`` carries the Jacobian of the sample transform; a constant offset
    shifts the log-likelihood by ``N * offset``.
    """
    data = analytical_data(n_events=3, n_samples=5)
    out = analytical_gwalk_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(),
        data["samples_stack"],
        jnp.full_like(data["ln_offsets"], offset),
        data["pmean_kwargs"],
        None,
    )
    expected = 3 * (0.75 + np.log(5) + offset)
    chex.assert_trees_all_close(out, jnp.asarray(expected))


@pytest.mark.parametrize(("expected_rate", "T_obs"), [(0.0, 1.0), (4.0, 3.0)])
def test_analytical_rate_and_observing_time(
    mixture, estimator, analytical_data, expected_rate, T_obs
):
    data = analytical_data(n_events=3, n_samples=5, T_obs=T_obs)
    out = analytical_gwalk_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(mean=expected_rate),
        data["samples_stack"],
        data["ln_offsets"],
        data["pmean_kwargs"],
        None,
    )
    expected = 3 * (0.75 + np.log(5)) + 3 * np.log(T_obs) - expected_rate
    chex.assert_trees_all_close(out, jnp.asarray(expected))


def test_analytical_out_of_support_samples_are_dropped(mixture, estimator):
    """Resampled points can land outside the model support; those are masked out.

    Here the surviving samples are the first two of each event, so the result must equal
    that of a stack that only ever contained those two.
    """
    n_events, n_keep, n_drop = 3, 2, 4
    kept = jnp.full((n_events, n_keep, 2), 0.5)
    stack = jnp.concatenate([kept, jnp.full((n_events, n_drop, 2), 5.0)], axis=1)
    model = mixture([0.75], validate_args=False)
    est = estimator()
    out = analytical_gwalk_poisson_likelihood_fn(
        model, est, stack, jnp.zeros((n_events, n_keep + n_drop)), {"T_obs": 1.0}, None
    )
    reference = analytical_gwalk_poisson_likelihood_fn(
        model, est, kept, jnp.zeros((n_events, n_keep)), {"T_obs": 1.0}, None
    )
    chex.assert_trees_all_close(out, reference)
    chex.assert_trees_all_close(out, jnp.asarray(n_events * (0.75 + np.log(n_keep))))


def test_analytical_masks_out_nan_log_probs(estimator):
    """The ``where=`` mask must exclude samples whose log-density is NaN, not just those
    whose log-density is ``-inf``.

    A log-normal component returns NaN for a negative argument; without the support mask
    that NaN would propagate through ``logsumexp`` and poison the whole run.
    """
    n_events = 3
    model = ScaledMixture(
        jnp.asarray([0.75]),
        [JointDistribution(LogNormal(0.0, 1.0), LogNormal(0.0, 1.0))],
    )
    stack = jnp.concatenate(
        [jnp.ones((n_events, 2, 2)), jnp.full((n_events, 3, 2), -1.0)], axis=1
    )
    assert bool(jnp.any(jnp.isnan(model.log_prob(stack))))
    out = analytical_gwalk_poisson_likelihood_fn(
        model, estimator(), stack, jnp.zeros((n_events, 5)), {"T_obs": 1.0}, None
    )
    assert bool(jnp.isfinite(out))
    expected = n_events * (
        0.75 + np.log(2) + 2.0 * float(LogNormal(0.0, 1.0).log_prob(jnp.asarray(1.0)))
    )
    chex.assert_trees_all_close(out, jnp.asarray(expected))


def test_analytical_event_with_no_valid_samples_gives_minus_inf(mixture, estimator):
    n_events, n_samples = 2, 4
    stack = jnp.full((n_events, n_samples, 2), 0.5).at[0].set(5.0)
    out = analytical_gwalk_poisson_likelihood_fn(
        mixture([0.75], validate_args=False),
        estimator(),
        stack,
        jnp.zeros((n_events, n_samples)),
        {"T_obs": 1.0},
        None,
    )
    assert bool(jnp.isneginf(out))


def test_analytical_taper_uses_only_the_estimator_variance(
    mixture, estimator, analytical_data
):
    """Unlike the discrete form, no per-event variance is accumulated here, so the taper
    depends solely on the estimator's variance.
    """
    threshold, est_variance = 1.0, 4.0
    data = analytical_data(n_events=3, n_samples=5)
    out = analytical_gwalk_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(variance=est_variance),
        data["samples_stack"],
        data["ln_offsets"],
        data["pmean_kwargs"],
        threshold,
    )
    expected = 3 * (0.75 + np.log(5)) - 100.0 * (est_variance - threshold) ** 2
    chex.assert_trees_all_close(out, jnp.asarray(expected))


def test_analytical_no_taper_below_threshold(mixture, estimator, analytical_data):
    data = analytical_data(n_events=3, n_samples=5)
    out = analytical_gwalk_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(variance=0.25),
        data["samples_stack"],
        data["ln_offsets"],
        data["pmean_kwargs"],
        1.0,
    )
    chex.assert_trees_all_close(out, jnp.asarray(3 * (0.75 + np.log(5))))


def test_analytical_non_finite_variance_is_tapered_to_minus_inf(
    mixture, estimator, analytical_data
):
    data = analytical_data(n_events=3, n_samples=5)
    out = analytical_gwalk_poisson_likelihood_fn(
        mixture([0.75]),
        estimator(variance=jnp.inf),
        data["samples_stack"],
        data["ln_offsets"],
        data["pmean_kwargs"],
        1.0,
    )
    assert bool(jnp.isneginf(out))


def test_analytical_passes_model_and_kwargs_to_estimator(
    mixture, estimator, analytical_data
):
    model = mixture([0.75])
    est = estimator()
    data = analytical_data(n_events=2, n_samples=3, T_obs=2.0)
    analytical_gwalk_poisson_likelihood_fn(
        model,
        est,
        data["samples_stack"],
        data["ln_offsets"],
        data["pmean_kwargs"],
        None,
    )
    assert len(est.calls) == 1
    recorded_model, recorded_kwargs = est.calls[0]
    assert recorded_model is model
    assert recorded_kwargs == {"T_obs": 2.0}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("variance_cut_threshold", [None, 0.25, 1e3])
def test_analytical_matches_reference_implementation(
    normal_mixture, estimator, seed, variance_cut_threshold
):
    keys = jax.random.split(jax.random.key(seed), 5)
    model = normal_mixture(
        jax.random.uniform(keys[0], (2,), minval=-1.0, maxval=1.0),
        locs=jax.random.normal(keys[1], (2, 3)),
        scales=jax.random.uniform(keys[2], (2, 3), minval=0.5, maxval=2.0),
    )
    samples_stack = jax.random.normal(keys[3], (5, 7, 3))
    ln_offsets = jax.random.uniform(keys[4], (5, 7), minval=-1.0, maxval=1.0)

    expected_rate, expected_variance, T_obs = 3.25, 0.5, 2.0
    out = analytical_gwalk_poisson_likelihood_fn(
        model,
        estimator(mean=expected_rate, variance=expected_variance),
        samples_stack,
        ln_offsets,
        {"T_obs": T_obs},
        variance_cut_threshold,
    )
    reference = _reference_analytical(
        model, samples_stack, ln_offsets, T_obs, expected_rate
    )
    if variance_cut_threshold is not None:
        reference -= 100.0 * max(expected_variance - variance_cut_threshold, 0.0) ** 2
    chex.assert_trees_all_close(out, jnp.asarray(reference), rtol=1e-6)


def test_analytical_gradient_wrt_log_scales(estimator, analytical_data):
    n_events = 3
    data = analytical_data(n_events=n_events, n_samples=5)
    est = estimator()

    def log_likelihood(log_scales):
        model = ScaledMixture(
            log_scales, [JointDistribution(Uniform(0.0, 1.0), Uniform(0.0, 1.0))]
        )
        return analytical_gwalk_poisson_likelihood_fn(
            model,
            est,
            data["samples_stack"],
            data["ln_offsets"],
            data["pmean_kwargs"],
            None,
        )

    grad = jax.grad(log_likelihood)(jnp.asarray([0.75]))
    chex.assert_trees_all_close(grad, jnp.asarray([float(n_events)]))


def test_analytical_is_jittable(estimator, analytical_data):
    data = analytical_data(n_events=3, n_samples=5)
    est = estimator()

    @jax.jit
    def log_likelihood(log_scales):
        model = ScaledMixture(
            log_scales, [JointDistribution(Uniform(0.0, 1.0), Uniform(0.0, 1.0))]
        )
        return analytical_gwalk_poisson_likelihood_fn(
            model,
            est,
            data["samples_stack"],
            data["ln_offsets"],
            data["pmean_kwargs"],
            None,
        )

    chex.assert_trees_all_close(
        log_likelihood(jnp.asarray([0.75])), jnp.asarray(3 * (0.75 + np.log(5)))
    )
