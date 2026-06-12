# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Tests for the mixed-mode Poisson likelihood (sampled + discrete, one μ).

Same closed-form problem as ``test_sampled_poisson_likelihood.py`` (1-D Gaussian
population intensity × Gaussian event likelihoods), so every assertion is against
an exact reference:

* **MX0 — exact reduction.** With an empty discrete group, the mixed likelihood
  is byte-for-byte the sampled likelihood.
* **MX1 — discrete-only matches exact.** With an empty sampled group (model draws
  kept only for the shared μ), the discrete per-event sum + sample-based μ
  recovers the closed-form log L within MC error.
* **MX2 — split equals whole.** Splitting ONE catalogue into a broad subset
  (sampled term) and a narrow subset (discrete term, posterior samples vs the
  model density) reproduces both the all-sampled log L and the exact log L within
  MC error — the core equivalence justifying the mixed estimator.
* **MX3 — diagnostics.** Per-event ESS arrays for both groups are exposed.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from gwkokab.inference.poissonlikelihood_utils import (
    mixed_poisson_likelihood_fn,
    sampled_poisson_likelihood_fn,
)
from gwkokab.inference.event_likelihoods import gaussian_event_log_likelihoods

jax.config.update("jax_enable_x64", True)

_M, _S, _LOGR, _T = 1.0, 0.8, math.log(120.0), 2.0
_D = jnp.asarray([-1.0, -0.3, 0.0, 0.5, 1.0, 1.4, 2.0, -0.8])[:, None]
_SIGMA = jnp.asarray([0.3, 0.25, 0.4, 0.2, 0.35, 0.3, 0.45, 0.28])


def _draw_model_samples(key, m, s, n, logr):
    z = jax.random.normal(key, (n, 1))
    return m + s * z, jnp.full((n,), logr - math.log(n))


def _event_Ls(d=_D, sig=_SIGMA):
    return gaussian_event_log_likelihoods(d, (sig**2)[:, None, None])


def _exact_logL(d, sig, logr, m, s, T):
    std = jnp.sqrt(sig**2 + s**2)
    logI = logr + jax.scipy.stats.norm.logpdf(d, loc=m, scale=std)
    return jnp.sum(logI) + d.shape[0] * jnp.log(T) - T * jnp.exp(logr)


def _scaled_mvn(log_rate, m, s):
    import numpyro.distributions as dist

    class ScaledMVN(dist.MultivariateNormal):
        def __init__(self, lr, loc, cov):
            self._lr = lr
            super().__init__(loc, covariance_matrix=cov)

        def log_prob(self, v):
            return self._lr + super().log_prob(v)

    return ScaledMVN(log_rate, jnp.asarray([m]), jnp.asarray([[s**2]]))


def _discrete_group(d, sig, key, M=20_000):
    """Posterior samples y ~ N(d_i, sig_i) (=normalised event likelihood) with a
    FLAT ref prior (log π = 0), so logsumexp(log ρ(y) − log π) − log M estimates
    I_i = ∫ L_i n dx = E_{y~L_i}[n(y)] — the same I_i the sampled term forms."""
    n = d.shape[0]
    keys = jax.random.split(key, n)
    pe = jnp.stack([d[i] + sig[i] * jax.random.normal(keys[i], (M, 1)) for i in range(n)])
    data_group = (pe,)                                  # one batch (n, M, 1)
    log_ref_priors_group = (jnp.zeros((n, M)),)         # flat π
    masks_group = (jnp.ones((n, M), dtype=bool),)
    N_pes = (jnp.full((n,), float(M)),)
    return data_group, log_ref_priors_group, masks_group, N_pes


# --------------------------------------------------------------------------- #
def test_MX0_empty_discrete_reduces_to_sampled():
    samples, log_w = _draw_model_samples(jax.random.PRNGKey(0), _M, _S, 200_000, _LOGR)
    Ls = _event_Ls()
    ll_s, _ = sampled_poisson_likelihood_fn(Ls, samples, log_w, None, _T, None)
    ll_m, diag = mixed_poisson_likelihood_fn(
        Ls, samples, log_w, None,
        None, (), (), (), (),          # empty discrete group
        _T, None,
    )
    assert jnp.abs(ll_s - ll_m) < 1e-9          # exact reduction
    assert diag["ess_per_event_sampled"].shape == (len(Ls),)


def test_MX1_discrete_only_matches_exact():
    samples, log_w = _draw_model_samples(jax.random.PRNGKey(1), _M, _S, 200_000, _LOGR)
    model = _scaled_mvn(_LOGR, _M, _S)
    dg, lpg, mg, npes = _discrete_group(_D, _SIGMA, jax.random.PRNGKey(2))
    ll_m, _ = mixed_poisson_likelihood_fn(
        [], samples, log_w, None,           # empty sampled group (samples kept for μ)
        model, dg, lpg, mg, npes,
        _T, None,
    )
    ll_exact = _exact_logL(_D[:, 0], _SIGMA, _LOGR, _M, _S, _T)
    assert jnp.isfinite(ll_m)
    assert jnp.abs(ll_m - ll_exact) < 0.2


def test_MX2_split_equals_whole_and_exact():
    samples, log_w = _draw_model_samples(jax.random.PRNGKey(3), _M, _S, 300_000, _LOGR)
    # all-sampled reference on the full catalogue
    ll_all, _ = sampled_poisson_likelihood_fn(_event_Ls(), samples, log_w, None, _T, None)
    ll_exact = _exact_logL(_D[:, 0], _SIGMA, _LOGR, _M, _S, _T)

    # split: |d|>=0.5 -> broad/sampled ; |d|<0.5 -> narrow/discrete
    broad = jnp.abs(_D[:, 0]) >= 0.5
    import numpy as np
    bi = np.where(np.asarray(broad))[0]
    ni = np.where(~np.asarray(broad))[0]
    Ls_broad = gaussian_event_log_likelihoods(_D[bi], (_SIGMA[bi] ** 2)[:, None, None])
    model = _scaled_mvn(_LOGR, _M, _S)
    dg, lpg, mg, npes = _discrete_group(_D[ni], _SIGMA[ni], jax.random.PRNGKey(4))

    ll_mixed, diag = mixed_poisson_likelihood_fn(
        Ls_broad, samples, log_w, None,
        model, dg, lpg, mg, npes,
        _T, None,
    )
    # same event count, same μ, same per-event evidences (different estimators)
    assert jnp.abs(ll_mixed - ll_all) < 0.2
    assert jnp.abs(ll_mixed - ll_exact) < 0.2
    assert diag["ess_per_event_sampled"].shape == (len(bi),)
    assert diag["log_evidence_per_event_discrete"].shape == (len(ni),)


def test_MX4_numpyro_wrapper_finite_differentiable():
    pytest.importorskip("gwkokab.models.utils")
    from numpyro.infer.util import log_density
    import numpyro.distributions as dist
    from gwkokab.models.utils import JointDistribution
    from gwkokab.inference.numpyro_mixed_poisson_likelihood import (
        numpyro_mixed_poisson_likelihood,
    )

    n_samples = 40_000
    z = jax.random.normal(jax.random.PRNGKey(8), (n_samples, 1))

    def sampler_fn(loc_pop, log_rate, scale_pop):
        samples = loc_pop + scale_pop * z
        log_w = jnp.full((n_samples,), log_rate - math.log(n_samples))
        return samples, log_w

    def dist_fn(loc_pop, log_rate, scale_pop):
        return _scaled_mvn(log_rate, loc_pop, scale_pop)

    variables = {"loc_pop": dist.Normal(0.0, 3.0)}
    variables_index = {"loc_pop": 0}
    constant_params = {"log_rate": _LOGR, "scale_pop": _S}
    priors = JointDistribution(dist.Normal(0.0, 3.0))

    # split the fixed catalogue: broad -> sampled, narrow -> discrete
    broad = jnp.abs(_D[:, 0]) >= 0.5
    import numpy as np
    bi = np.where(np.asarray(broad))[0]
    ni = np.where(~np.asarray(broad))[0]
    Ls_broad = gaussian_event_log_likelihoods(_D[bi], (_SIGMA[bi] ** 2)[:, None, None])
    dg, lpg, mg, npes = _discrete_group(_D[ni], _SIGMA[ni], jax.random.PRNGKey(9))

    model = numpyro_mixed_poisson_likelihood(
        sampler_fn, dist_fn, priors, variables, constant_params, variables_index,
        event_log_likelihood_fns=Ls_broad,
        data_group=dg, log_ref_priors_group=lpg, masks_group=mg, N_pes=npes,
        pdet_fn=None, variance_cut_threshold=None,
    )

    def potential(loc):
        lp, _ = log_density(model, (_T,), {}, {"loc_pop": loc})
        return lp

    val = potential(jnp.asarray(_M))
    grad = jax.grad(potential)(jnp.asarray(_M))
    assert jnp.isfinite(val) and jnp.isfinite(grad)
    # the likelihood peaks near the catalogue mean (~0.35 for the fixed _D);
    # the gradient must bracket it: positive below, negative above.
    g = jax.grad(potential)
    assert g(jnp.asarray(-0.3)) > 0 and g(jnp.asarray(1.2)) < 0


def test_MX3_diagnostics_present():
    samples, log_w = _draw_model_samples(jax.random.PRNGKey(5), _M, _S, 50_000, _LOGR)
    model = _scaled_mvn(_LOGR, _M, _S)
    dg, lpg, mg, npes = _discrete_group(_D[:3], _SIGMA[:3], jax.random.PRNGKey(6), M=5_000)
    _, diag = mixed_poisson_likelihood_fn(
        _event_Ls(_D[3:], _SIGMA[3:]), samples, log_w, None,
        model, dg, lpg, mg, npes, _T, None,
    )
    for k in ("ess_per_event_sampled", "log_evidence_per_event_sampled",
              "log_evidence_per_event_discrete", "expected_rate"):
        assert k in diag
