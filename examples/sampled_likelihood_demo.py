# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Demo: sample-mode (likelihood-mode) population inference end to end.

Runs the test-problem sequence of ``demo/PLAN_gwkokab_sampled_likelihood.md``
on a closed-form Gaussian problem and finishes with a **real numpyro NUTS run**
that recovers the population mean from samples drawn from the model — the
opposite of gwkokab's density mode, and the path a particle Monte-Carlo
population engine takes natively (draws + weights, no grid, no KDE).

Usage
-----
    export PYTHONPATH=src:.            # + RIFT stub in a lightweight sandbox
    python examples/sampled_likelihood_demo.py --n-events 150 --num-samples 400

The numpyro NUTS section is the heavy one; reduce ``--num-samples`` /
``--n-events`` for a quick smoke test.  Everything else runs in a few seconds.
"""

from __future__ import annotations

import argparse
import math

import jax
import jax.numpy as jnp

from gwkokab.inference.event_likelihoods import gaussian_event_log_likelihoods
from gwkokab.inference.poissonlikelihood_utils import (
    low_ess_events,
    sampled_poisson_likelihood_fn,
)


jax.config.update("jax_enable_x64", True)

M_TRUE, S_TRUE, LOGR_TRUE, T_OBS = 1.0, 0.8, math.log(120.0), 2.0


def draw_catalog(key, n_events, sigma_obs=0.3):
    k1, k2 = jax.random.split(key)
    x_true = M_TRUE + S_TRUE * jax.random.normal(k1, (n_events,))
    d = (x_true + sigma_obs * jax.random.normal(k2, (n_events,)))[:, None]
    sigma = jnp.full((n_events,), sigma_obs)
    return d, sigma


def model_sampler(m, s, log_rate, z):
    samples = m + s * z
    log_w = jnp.full((z.shape[0],), log_rate - math.log(z.shape[0]))
    return samples, log_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=150)
    ap.add_argument("--n-model-samples", type=int, default=40_000)
    ap.add_argument("--num-warmup", type=int, default=300)
    ap.add_argument("--num-samples", type=int, default=400)
    ap.add_argument("--skip-nuts", action="store_true")
    args = ap.parse_args()

    d, sigma = draw_catalog(jax.random.PRNGKey(0), args.n_events)
    Ls = gaussian_event_log_likelihoods(d, (sigma**2)[:, None, None])
    z = jax.random.normal(jax.random.PRNGKey(1), (args.n_model_samples, 1))

    # --- diagnostics at the truth ---------------------------------------- #
    samples, log_w = model_sampler(M_TRUE, S_TRUE, LOGR_TRUE, z)
    ll, diag = sampled_poisson_likelihood_fn(Ls, samples, log_w, None, T_OBS, None)
    flagged = low_ess_events(diag["ess_per_event"], args.n_model_samples, frac=0.1)
    print(f"log L at truth         : {float(ll):+.3f}")
    print(f"expected detected count: {float(diag['expected_rate']):.2f}")
    print(f"median per-event ESS   : {float(jnp.median(diag['ess_per_event'])):.0f}"
          f" / {args.n_model_samples}")
    print(f"low-ESS (tail) events  : {len(flagged)}")

    # --- MAP recovery via gradient ascent through the sampler ------------- #
    def loglik(m):
        s_, log_w_ = model_sampler(m, S_TRUE, LOGR_TRUE, z)
        out, _ = sampled_poisson_likelihood_fn(Ls, s_, log_w_, None, T_OBS, None)
        return out

    vg = jax.jit(jax.value_and_grad(loglik))
    m = jnp.asarray(M_TRUE + 0.8)
    for _ in range(400):
        _, g = vg(m)
        m = m + 3e-3 * g
    print(f"MAP (grad-ascent) mean : {float(m):.3f}   (truth {M_TRUE})")

    if args.skip_nuts:
        return

    # --- full numpyro NUTS posterior ------------------------------------- #
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from gwkokab.inference.numpyro_sampled_poisson_likelihood import (
        numpyro_sampled_poisson_likelihood,
    )
    from gwkokab.models.utils import JointDistribution

    def sampler_fn(loc_pop, log_rate, scale_pop):
        return model_sampler(loc_pop, scale_pop, log_rate, z)

    variables = {"loc_pop": dist.Normal(0.0, 3.0)}
    priors = JointDistribution(dist.Normal(0.0, 3.0))
    model = numpyro_sampled_poisson_likelihood(
        sampler_fn,
        priors,
        variables,
        {"log_rate": LOGR_TRUE, "scale_pop": S_TRUE},
        {"loc_pop": 0},
        Ls,
        None,
        None,
    )

    mcmc = MCMC(
        NUTS(model),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(2), T_OBS)
    post = mcmc.get_samples()["loc_pop"]
    print(
        f"NUTS posterior loc_pop : {float(jnp.mean(post)):.3f} "
        f"+/- {float(jnp.std(post)):.3f}   (truth {M_TRUE})"
    )


if __name__ == "__main__":
    main()
