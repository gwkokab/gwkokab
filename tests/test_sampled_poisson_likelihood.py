# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Test-problem sequence for the sample-mode (likelihood-mode) Poisson likelihood.

The sampled path evaluates per-event *likelihoods* at samples drawn from the
population *model*, the dual of the analytical path (model density at event
samples).  This suite is a graded sequence on a problem where everything is
available in closed form — a Gaussian population intensity with Gaussian event
likelihoods — so each acceptance gate of
``demo/PLAN_gwkokab_sampled_likelihood.md`` can be checked against an exact
reference:

* **TP0** — estimator identity: ``Σ_j w_j L_i(x_j)`` recovers the closed-form
  evidence ``I_i = ∫ L_i n dx`` within MC error.
* **TP1** — mode equivalence (**gate i**): on the same catalogue the sampled
  log-likelihood matches both the exact log-likelihood and the existing
  ``analytical_poisson_likelihood_fn`` within MC error, across a grid of θ.
* **TP2** — AD through the sampler (**gate ii**): ``jax.grad`` of the sampled
  log-likelihood w.r.t. a population parameter is finite and matches a
  many-key finite-difference **mean** (not a single-key FD).
* **TP3** — ESS diagnostic (**gate iii**): an event in the tail of ``n(·|θ)``
  collapses its effective sample size and is flagged by ``low_ess_events``.
* **TP4** — recovery: gradient ascent through the sampler recovers the true
  population mean; and the numpyro wrapper assembles a finite, differentiable
  log-density (full NUTS recovery lives in the demo for the workstation).

The core function (TP0–TP3) needs only jax + numpyro.  TP1's density-mode
cross-check and TP4's wrapper test import the real gwkokab inference layer and
``importorskip`` when it (or its RIFT import) is unavailable — same pattern as
the bridge's G6/G7 gates.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gwkokab.inference.poissonlikelihood_utils import (
    low_ess_events,
    sampled_poisson_likelihood_fn,
)
from gwkokab.inference.event_likelihoods import (
    GaussianEventLikelihood,
    gaussian_event_log_likelihoods,
)


jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# closed-form reference
# --------------------------------------------------------------------------- #
def _exact_log_evidence(d, sigma, log_rate, m, s):
    r"""log I_i = log R + log N(d_i; m, sqrt(sigma_i^2 + s^2)) (1-D)."""
    std = jnp.sqrt(sigma**2 + s**2)
    return log_rate + jax.scipy.stats.norm.logpdf(d, loc=m, scale=std)


def _exact_log_likelihood(d, sigma, log_rate, m, s, T_obs):
    r"""Σ_i log I_i + n_events log T_obs − μ,   μ = T_obs · R (flat selection)."""
    log_I = _exact_log_evidence(d, sigma, log_rate, m, s)
    n_events = d.shape[0]
    mu = T_obs * jnp.exp(log_rate)
    return jnp.sum(log_I) + n_events * jnp.log(T_obs) - mu


def _draw_model_samples(key, m, s, n_samples, log_rate):
    """x_j ~ Normal(m, s), uniform log-weights summing (in exp) to the rate R."""
    z = jax.random.normal(key, (n_samples, 1))
    samples = m + s * z
    log_w = jnp.full((n_samples,), log_rate - math.log(n_samples))
    return samples, log_w


# small fixed catalogue (1-D parameter x)
_M_TRUE, _S_TRUE, _LOGR_TRUE, _T_OBS = 1.0, 0.8, math.log(120.0), 2.0
_D = jnp.asarray([-1.0, -0.3, 0.0, 0.5, 1.0, 1.4, 2.0, -0.8])[:, None]  # (n_ev,1)
_SIGMA = jnp.asarray([0.3, 0.25, 0.4, 0.2, 0.35, 0.3, 0.45, 0.28])


def _event_likelihoods(d=_D, sigma=_SIGMA):
    covs = (sigma**2)[:, None, None]  # (n_ev,1,1)
    return gaussian_event_log_likelihoods(d, covs)


def _draw_catalog(key, m_true, s, n_events, sigma_obs=0.3):
    """Catalogue actually drawn from the population, so the MLE of the mean ~ m_true.

    True source param x_i ~ Normal(m_true, s); observed event centre
    d_i = x_i + sigma_obs · noise.  The marginal of d_i is Normal(m_true,
    sqrt(s^2 + sigma_obs^2)), so the maximum-likelihood population mean tends to
    m_true as n_events grows.
    """
    k1, k2 = jax.random.split(key)
    x_true = m_true + s * jax.random.normal(k1, (n_events,))
    d = (x_true + sigma_obs * jax.random.normal(k2, (n_events,)))[:, None]  # (n_ev,1)
    sigma = jnp.full((n_events,), sigma_obs)
    return d, sigma


# --------------------------------------------------------------------------- #
# TP0 — estimator identity
# --------------------------------------------------------------------------- #
def test_TP0_estimator_recovers_closed_form_evidence():
    key = jax.random.PRNGKey(0)
    n_samples = 200_000
    samples, log_w = _draw_model_samples(key, _M_TRUE, _S_TRUE, n_samples, _LOGR_TRUE)

    d, sigma = jnp.asarray([[0.5]]), jnp.asarray([0.3])
    L = GaussianEventLikelihood(d[0], jnp.asarray([[sigma[0] ** 2]]))
    log_I_hat = jax.nn.logsumexp(log_w + L(samples))
    log_I_exact = _exact_log_evidence(d[0, 0], sigma[0], _LOGR_TRUE, _M_TRUE, _S_TRUE)

    # MC relative error on I ~ 1/sqrt(ESS); 200k draws, central event -> tight.
    assert jnp.abs(jnp.exp(log_I_hat - log_I_exact) - 1.0) < 5e-3


# --------------------------------------------------------------------------- #
# TP1 — mode equivalence (gate i)
# --------------------------------------------------------------------------- #
def test_TP1_sampled_matches_exact_loglikelihood():
    key = jax.random.PRNGKey(1)
    n_samples = 200_000
    Ls = _event_likelihoods()

    for offset in (-0.4, 0.0, 0.3):  # a small grid of population means θ=m
        m = _M_TRUE + offset
        samples, log_w = _draw_model_samples(key, m, _S_TRUE, n_samples, _LOGR_TRUE)
        ll, diag = sampled_poisson_likelihood_fn(
            Ls, samples, log_w, None, _T_OBS, None
        )
        ll_exact = _exact_log_likelihood(
            _D[:, 0], _SIGMA, _LOGR_TRUE, m, _S_TRUE, _T_OBS
        )
        # Only Σ log I_i carries MC error (~O(0.05)); μ and the T_obs term are
        # exact, so this absolute tolerance on the full log L is a sub-percent
        # check on the per-event evidences.
        assert jnp.isfinite(ll)
        assert jnp.abs(ll - ll_exact) < 0.15
        # μ is exact for flat selection (Σ w_j = R)
        assert jnp.abs(diag["expected_rate"] - _T_OBS * math.exp(_LOGR_TRUE)) < 1e-6


def test_TP1_sampled_matches_analytical_density_mode():
    """Cross-check against gwkokab's own analytical (density-mode) estimator."""
    pytest.importorskip("gwkokab.models.utils")
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    from gwkokab.inference.poissonlikelihood_utils import (
        analytical_poisson_likelihood_fn,
    )

    class ScaledMVN(dist.MultivariateNormal):
        """log_prob = log R + log N(x; loc, cov) — a ScaledMixture-style intensity."""

        def __init__(self, log_rate, loc, cov):
            self._log_rate = log_rate
            super().__init__(loc, covariance_matrix=cov)

        def log_prob(self, value):
            return self._log_rate + super().log_prob(value)

    m, s = _M_TRUE - 0.2, _S_TRUE
    model = ScaledMVN(_LOGR_TRUE, jnp.asarray([m]), jnp.asarray([[s**2]]))

    def pmean(model_instance, T_obs):
        return T_obs * jnp.exp(model_instance._log_rate), jnp.zeros(())

    # density mode: event PE samples drawn FROM each event likelihood (proposal
    # = the normalised event normal), ln_offset = -log M  ->  estimates the same
    # I_i = ∫ L_i n dx as the sample mode.
    key = jax.random.PRNGKey(2)
    M = 20_000
    n_ev = _D.shape[0]
    keys = jax.random.split(key, n_ev)
    pe = jnp.stack(
        [_D[i] + _SIGMA[i] * jax.random.normal(keys[i], (M, 1)) for i in range(n_ev)]
    )  # (n_ev, M, 1)
    ln_offsets = jnp.full((n_ev, M), -math.log(M))
    ll_density = analytical_poisson_likelihood_fn(
        model, pmean, pe, ln_offsets, {"T_obs": _T_OBS}, None
    )

    # sample mode on the same population
    samples, log_w = _draw_model_samples(
        jax.random.PRNGKey(3), m, s, 400_000, _LOGR_TRUE
    )
    ll_sampled, _ = sampled_poisson_likelihood_fn(
        _event_likelihoods(), samples, log_w, None, _T_OBS, None
    )
    # both are MC estimators of the same log L; agreement at the combined
    # MC-error level confirms the two modes are the same identity.
    assert jnp.abs(ll_sampled - ll_density) < 0.15


# --------------------------------------------------------------------------- #
# TP2 — AD through the sampler (gate ii)
# --------------------------------------------------------------------------- #
def _loglik_of_mean(m, z, log_w, Ls, T_obs):
    """Reparameterised: samples = m + s*z, so grad flows through the draws."""
    samples = m + _S_TRUE * z
    ll, _ = sampled_poisson_likelihood_fn(Ls, samples, log_w, None, T_obs, None)
    return ll


def test_TP2_grad_through_sampler_matches_manykey_fd_mean():
    Ls = _event_likelihoods()
    n_samples = 30_000
    log_w = jnp.full((n_samples,), _LOGR_TRUE - math.log(n_samples))
    m0 = jnp.asarray(_M_TRUE)
    h = 1e-3
    n_keys = 12

    grad_fn = jax.grad(lambda m, z: _loglik_of_mean(m, z, log_w, Ls, _T_OBS))

    ad_grads, fd_grads = [], []
    for k in range(n_keys):
        z = jax.random.normal(jax.random.PRNGKey(100 + k), (n_samples, 1))
        ad_grads.append(float(grad_fn(m0, z)))
        # finite difference with the SAME draws (z fixed) — the reparameterised
        # analogue of fixed-selection-replay FD; averaged over keys per gate (ii).
        fp = _loglik_of_mean(m0 + h, z, log_w, Ls, _T_OBS)
        fm = _loglik_of_mean(m0 - h, z, log_w, Ls, _T_OBS)
        fd_grads.append(float((fp - fm) / (2 * h)))

    ad_mean, fd_mean = np.mean(ad_grads), np.mean(fd_grads)
    assert np.all(np.isfinite(ad_grads))
    assert abs(ad_mean - fd_mean) < 1e-2 * (abs(fd_mean) + 1.0)


# --------------------------------------------------------------------------- #
# TP3 — ESS diagnostic (gate iii)
# --------------------------------------------------------------------------- #
def test_TP3_tail_event_collapses_ess():
    key = jax.random.PRNGKey(4)
    n_samples = 50_000
    samples, log_w = _draw_model_samples(key, _M_TRUE, _S_TRUE, n_samples, _LOGR_TRUE)

    d = jnp.asarray([[_M_TRUE], [_M_TRUE + 9.0 * _S_TRUE]])  # central, far-tail
    sigma = jnp.asarray([0.3, 0.3])
    Ls = gaussian_event_log_likelihoods(d, (sigma**2)[:, None, None])

    _, diag = sampled_poisson_likelihood_fn(Ls, samples, log_w, None, _T_OBS, None)
    ess = diag["ess_per_event"]

    assert ess[0] > 0.05 * n_samples  # central event well resolved
    assert ess[1] < 0.01 * n_samples  # tail event collapsed
    flagged = low_ess_events(ess, n_samples, frac=0.05)
    assert 1 in flagged and 0 not in flagged


# --------------------------------------------------------------------------- #
# TP4 — recovery
# --------------------------------------------------------------------------- #
def test_TP4a_gradient_ascent_recovers_population_mean():
    """End-to-end: ascend log L(m) through the reparameterised sampler on a
    catalogue actually drawn from the true population, recovering m_true."""
    d, sigma = _draw_catalog(jax.random.PRNGKey(70), _M_TRUE, _S_TRUE, n_events=120)
    Ls = gaussian_event_log_likelihoods(d, (sigma**2)[:, None, None])

    n_samples = 40_000
    z = jax.random.normal(jax.random.PRNGKey(7), (n_samples, 1))
    log_w = jnp.full((n_samples,), _LOGR_TRUE - math.log(n_samples))

    val_grad = jax.jit(
        jax.value_and_grad(lambda m: _loglik_of_mean(m, z, log_w, Ls, _T_OBS))
    )
    m = jnp.asarray(_M_TRUE + 0.6)  # start displaced from truth
    lr = 3e-3
    for _ in range(300):
        _, g = val_grad(m)
        m = m + lr * g
    # MLE of the mean ~ truth, up to O(std/sqrt(n_events)) sampling scatter.
    assert jnp.abs(m - _M_TRUE) < 0.2


def test_TP4b_numpyro_wrapper_assembles_finite_differentiable_density():
    pytest.importorskip("gwkokab.models.utils")
    numpyro = pytest.importorskip("numpyro")
    from numpyro.infer.util import log_density
    import numpyro.distributions as dist
    from gwkokab.inference.numpyro_sampled_poisson_likelihood import (
        numpyro_sampled_poisson_likelihood,
    )

    n_samples = 20_000
    z = jax.random.normal(jax.random.PRNGKey(8), (n_samples, 1))

    def sampler_fn(loc_pop, log_rate, scale_pop):
        samples = loc_pop + scale_pop * z
        log_w = jnp.full((n_samples,), log_rate - math.log(n_samples))
        return samples, log_w

    variables = {"loc_pop": dist.Normal(0.0, 3.0)}
    variables_index = {"loc_pop": 0}
    constant_params = {"log_rate": _LOGR_TRUE, "scale_pop": _S_TRUE}

    # priors: a JointDistribution over sorted(variables) — single var here.
    from gwkokab.models.utils import JointDistribution

    priors = JointDistribution(dist.Normal(0.0, 3.0))

    # catalogue drawn from the true population so the likelihood peaks at m_true
    d, sigma = _draw_catalog(jax.random.PRNGKey(80), _M_TRUE, _S_TRUE, n_events=120)
    event_Ls = gaussian_event_log_likelihoods(d, (sigma**2)[:, None, None])

    model = numpyro_sampled_poisson_likelihood(
        sampler_fn,
        priors,
        variables,
        constant_params,
        variables_index,
        event_Ls,
        None,
        None,
    )

    def potential(loc):
        lp, _ = log_density(model, (_T_OBS,), {}, {"loc_pop": loc})
        return lp

    val = potential(jnp.asarray(_M_TRUE))
    grad = jax.grad(potential)(jnp.asarray(_M_TRUE))
    assert jnp.isfinite(val) and jnp.isfinite(grad)
    # from a start below the truth the posterior gradient points up toward it
    g_lo = jax.grad(potential)(jnp.asarray(_M_TRUE - 0.5))
    assert g_lo > 0
