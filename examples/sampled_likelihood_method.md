# Sample-mode (likelihood-mode) population inference

GWKokab's two established likelihoods evaluate the population model as a
**pointwise density** `p(x | θ)` at samples drawn from each event's posterior
(`analytical_poisson_likelihood_fn`, `discrete_poisson_likelihood_fn`). That
requires the model to expose a density — fine for closed-form mixtures, but a
hard wall for population models that are only available as **weighted Monte-Carlo
draws** (e.g. a hierarchical-merger coagulation engine). Reconstructing a density
from those draws means KDE, which over-smooths narrow features.

`sampled_poisson_likelihood_fn` is the dual. It evaluates a per-event
**likelihood** `L_i(x)` at samples drawn from the **model**, using the other
direction of the same importance-sampling identity for the per-event term of the
inhomogeneous-Poisson likelihood:

```
I_i(θ) = ∫ L_i(x) n(x|θ) dx ≈ Σ_j w_j L_i(x_j),     x_j ~ n(·|θ)
```

where `n(x|θ)` is the merger **rate density (intensity)** and the weights carry
its normalisation (`Σ_j w_j ≈ ∫ n dx =` total rate). The assembled
log-likelihood mirrors the analytical form exactly,

```
log L(θ) = Σ_i log I_i + n_events·log T_obs − μ(θ),
μ(θ) = T_obs · Σ_j w_j p_det(x_j)         (sample-based selection)
```

so with one model sample, flat selection and `T_obs = 1` it reduces to the same
single-event Poisson term the analytical path produces — a framework-standard
generalisation, not a new estimator. Because the samples and weights come from a
reparameterised JAX sampler, `jax.grad` of `log L` w.r.t. the population
parameters θ is well defined: the per-event draw probabilities become continuous,
differentiable quantities, enabling gradient-based inference and model selection
directly through the sample traces.

## What's provided

| piece | where |
|---|---|
| core estimator + per-event ESS / variance diagnostics | `gwkokab.inference.sampled_poisson_likelihood_fn`, `low_ess_events` |
| numpyro / flowMC wrappers (`sampler_fn(**params) -> (samples, log_w)` replaces `dist_fn`) | `numpyro_sampled_poisson_likelihood`, `flowMC_sampled_poisson_likelihood`; `factory.get_likelihood_fn(..., "sampled")` |
| per-event likelihood adapters | `GaussianEventLikelihood` (CI reference), `RIFTMarginalLikelihood` (production interpolant interface) |
| graded test-problem sequence (TP0–TP4) | `tests/test_sampled_poisson_likelihood.py` |
| end-to-end NUTS recovery demo | `examples/sampled_likelihood_demo.py` |

The public API is **additive** — the two density modes are untouched; the only
swaps in the new wrappers are `dist_fn → sampler_fn` and "model density at event
samples" → "event likelihoods at model samples". The same
`priors / variables / variables_index` `sorted`-order plumbing applies.

## Acceptance gates (closed-form Gaussian problem)

A Gaussian population intensity with Gaussian per-event likelihoods has a
closed-form per-event evidence `I_i = R · N(d_i; m, √(σ_i²+s²))`, so each gate is
checked against an exact reference:

* **TP0** — `Σ_j w_j L_i(x_j)` recovers the closed-form `I_i` within MC error.
* **TP1 (gate i)** — the sampled log-likelihood matches both the exact value and
  the existing `analytical_poisson_likelihood_fn` within MC error, across a grid
  of θ. The two modes are the same identity evaluated from opposite directions.
* **TP2 (gate ii)** — `jax.grad` through the sampler is finite and matches a
  **many-key finite-difference mean**, not a single-key FD (single-key FD is not
  a valid reference once the sampler carries stochastic selection — see the
  engine's `feedback_single_key_fd_not_ad_reference`).
* **TP3 (gate iii)** — an event in the tail of `n(·|θ)` collapses its effective
  sample size; `low_ess_events` flags it. The variance term is **not optional**
  when events can sit in the tail; stratified / weighted model proposals target
  the under-populated regions.
* **TP4** — gradient ascent through the sampler recovers the true population
  mean, and a real numpyro NUTS run recovers its posterior.

## Thread-back to a population engine

The production adapter is `RIFTMarginalLikelihood`: wrap a per-event RIFT
marginal-likelihood interpolant as an `L_i`, supply the engine's particle-MC
draws + weights as `sampler_fn`, and the population is constrained directly from
model samples — no grid, no KDE. That engine-side wiring is the next increment
(see `demo/PLAN_gwkokab_sampled_likelihood.md`).
