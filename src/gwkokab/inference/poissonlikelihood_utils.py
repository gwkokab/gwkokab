# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import Any, Dict, Tuple

import jax
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions.distribution import Distribution


__all__ = [
    "discrete_poisson_likelihood_fn",
    "analytical_poisson_likelihood_fn",
    "sampled_poisson_likelihood_fn",
    "low_ess_events",
]


def variance_tapering_fn(variance: Array, threshold: Array) -> Array:
    r"""Tapering function that penalizes high variance values more aggressively as they
    exceed the threshold.

    This is a modified implementation of the original variance tapering function
    available in Equation (12) of
    `PixelPop: High Resolution Nonparameteric Inference of Gravitational-Wave Populations in Multiple Dimensions <https://arxiv.org/abs/2406.16813>`_

    .. math::

        \mathcal{T}[\sigma^2_{\ln \hat{\mathcal{L}}}]=100(\sigma^2_{\ln \hat{\mathcal{L}}}-\sigma^2_{\mathrm{threshold}})^2

    Parameters
    ----------
    variance : Array
        Variance of the Poisson likelihood estimator.
    threshold : Array
        Threshold above which the variance will be penalized.

    Returns
    -------
    Array
        Tapering factor that can be applied to the log-likelihood to penalize high variance values.
    """
    return jnp.where(
        variance < threshold,
        jnp.zeros_like(variance),
        100.0 * jnp.square(variance - threshold),
    )


def discrete_poisson_likelihood_fn(
    model_instance: Distribution,
    poisson_mean_estimator: Callable[..., Tuple[Array, Array]],
    data_group: Tuple[Array, ...],
    log_ref_priors_group: Tuple[Array, ...],
    masks_group: Tuple[Array, ...],
    pmean_kwargs: Dict[str, Any],
    N_pes: Tuple[Array, ...],
    variance_cut_threshold: float | None,
) -> Array:

    n_events = sum([masks_group.shape[0] for masks_group in data_group])

    total_log_likelihood = -jnp.sum(
        jnp.asarray([jnp.log(N_pe).sum() for N_pe in N_pes])
    )  # - Σ log(M_i)
    pe_variance = jnp.zeros(())

    # Σ log Σ exp (log p(ω|data_n) - log π_n)
    for batched_data, batched_log_ref_priors, batched_mask, N_pe in zip(
        data_group, log_ref_priors_group, masks_group, N_pes
    ):
        feasible_point = model_instance.support.feasible_like(batched_data[0])

        safe_data = jnp.where(
            batched_mask[..., jnp.newaxis],
            batched_data,
            feasible_point,
        )

        # log p(ω|data_n)
        batch_model_log_prob: Array = model_instance.log_prob(safe_data)

        # log p(ω|data_n) - log π_n
        log_prob = batch_model_log_prob - batched_log_ref_priors
        log_prob = jnp.where(batched_mask, log_prob, -jnp.inf)

        # log Σ exp (log p(ω|data_n) - log π_n)
        log_prob_sum = jax.nn.logsumexp(log_prob, axis=-1)
        log_prob_sum_2 = jax.nn.logsumexp(2.0 * log_prob, axis=-1)

        total_log_likelihood += log_prob_sum.sum(axis=0, initial=0.0)

        pe_variance += (jnp.exp(log_prob_sum_2 - 2.0 * log_prob_sum) - 1.0 / N_pe).sum()

    # μ = E_{Ω|Λ}[VT(ω)]
    expected_rate, expected_rate_variance = poisson_mean_estimator(
        model_instance, **pmean_kwargs
    )
    # log L(ω) = -μ + Σ log Σ exp (log p(ω|data_n) - log π_n) - Σ log(M_i)
    log_likelihood = (
        total_log_likelihood - expected_rate + n_events * jnp.log(pmean_kwargs["T_obs"])
    )
    if variance_cut_threshold is not None:
        total_variance = jnp.nan_to_num(
            pe_variance + expected_rate_variance,
            nan=jnp.inf,
            posinf=jnp.inf,
            neginf=jnp.inf,
        )

        variance_tapering_factor = variance_tapering_fn(
            total_variance, variance_cut_threshold
        )
        log_likelihood -= variance_tapering_factor

    return log_likelihood


def analytical_poisson_likelihood_fn(
    model_instance: Distribution,
    poisson_mean_estimator: Callable[..., tuple[Array, Array]],
    samples_stack: Array,
    ln_offsets: Array,
    pmean_kwargs: Dict[str, Any],
    variance_cut_threshold: float | None,
) -> Array:
    mask = model_instance.support.check(samples_stack)

    def compute_event_log_prob(samples):
        return model_instance.log_prob(samples)

    log_prob_model = jax.vmap(compute_event_log_prob)(samples_stack)

    total_ln_l = jnp.sum(
        jax.nn.logsumexp(log_prob_model + ln_offsets, axis=1, where=mask)
    )

    expected_rates, expected_rate_variance = poisson_mean_estimator(
        model_instance, **pmean_kwargs
    )

    n_events, _, _ = samples_stack.shape
    T_obs = pmean_kwargs["T_obs"]

    log_likelihood = total_ln_l + n_events * jnp.log(T_obs) - expected_rates

    if variance_cut_threshold is not None:
        total_variance = jnp.nan_to_num(
            expected_rate_variance,
            nan=jnp.inf,
            posinf=jnp.inf,
            neginf=jnp.inf,
        )

        variance_tapering_factor = variance_tapering_fn(
            total_variance, variance_cut_threshold
        )
        log_likelihood -= variance_tapering_factor

    return log_likelihood


def sampled_poisson_likelihood_fn(
    event_log_likelihood_fns: Sequence[Callable[[Array], Array]],
    model_samples: Array,
    model_log_weights: Array,
    pdet_fn: Callable[[Array], Array] | None,
    T_obs: Array,
    variance_cut_threshold: float | None,
) -> Tuple[Array, Dict[str, Array]]:
    r"""Sample-mode (likelihood-mode) inhomogeneous-Poisson population likelihood.

    This is the **dual** of :func:`analytical_poisson_likelihood_fn`.  Instead of
    evaluating a pointwise model *density* ``p(x|θ)`` at samples drawn from each
    event's posterior, it evaluates a per-event *likelihood* ``L_i(x)`` at samples
    drawn from the **model itself** — exactly what a particle Monte-Carlo
    population engine produces natively (draws + weights), with no grid and no
    KDE.  It is the right path when the model is only available as weighted
    samples (so no pointwise density exists) and a per-event likelihood evaluator
    *is* available (e.g. a RIFT marginal-likelihood interpolant).

    The per-event term of the inhomogeneous-Poisson likelihood is the importance
    identity

    .. math::

        \mathcal{I}_i(\theta)=\int \mathcal{L}_i(x)\,n(x\mid\theta)\,dx
        \;\approx\;\sum_j w_j\,\mathcal{L}_i(x_j),
        \qquad x_j\sim n(\cdot\mid\theta),

    where ``n(x|θ)`` is the merger **rate density (intensity)** and the weights
    carry its normalisation, :math:`\sum_j w_j\approx\int n\,dx=` total rate.
    The assembled log-likelihood mirrors the analytical form exactly

    .. math::

        \log\mathcal{L} = \sum_i \log\mathcal{I}_i
        + n_\mathrm{events}\log T_\mathrm{obs} - \mu(\theta),
        \qquad
        \mu(\theta)=T_\mathrm{obs}\sum_j w_j\,p_\mathrm{det}(x_j),

    so with one model sample, flat selection and ``T_obs = 1`` it reduces to the
    same single-event Poisson term the analytical path produces — a faithful,
    framework-standard generalisation, not a new estimator.

    AD: the weights and samples flow from a reparameterised JAX sampler, so
    ``jax.grad`` of this term w.r.t. the population parameters ``θ`` is well
    defined.  Validate gradients against a many-key FD **mean**, never a
    single-key FD (selection indices are held fixed under AD but move under a
    single-key perturbation — see ``feedback_single_key_fd_not_ad_reference``).

    Parameters
    ----------
    event_log_likelihood_fns : Sequence[Callable]
        One duck-typed evaluator per event, ``L_i: (n_samples, n_dim) ->
        (n_samples,)`` returning **log**-likelihood values at the model samples.
    model_samples : Array
        ``(n_samples, n_dim)`` draws from the model intensity ``n(·|θ)``.
    model_log_weights : Array
        ``(n_samples,)`` log-weights ``log w_j``; ``logsumexp`` of these is the
        log total rate (per unit observing time).
    pdet_fn : Callable or None
        ``(n_samples, n_dim) -> (n_samples,)`` detection probability in ``(0, 1]``.
        ``None`` ⇒ flat selection ``VT ≡ 1`` (``μ = T_obs · Σ w_j``), the
        detected-rate convention with efficiency folded into the rate scale.
    T_obs : Array
        Observing time.
    variance_cut_threshold : float or None
        If set, penalise high total importance-sampling variance with
        :func:`variance_tapering_fn` (same hygiene as the discrete mode).

    Returns
    -------
    log_likelihood : Array
        Scalar population log-likelihood.
    diagnostics : Dict[str, Array]
        Per-event arrays — ``log_evidence_per_event`` (``log I_i``),
        ``ess_per_event`` (effective sample size of the IS estimate of
        ``I_i``; collapses when the model puts few samples under an event),
        ``rel_variance_per_event``, and the scalar ``expected_rate`` ``μ``.
        Feed ``ess_per_event`` to :func:`low_ess_events` for a host-side
        low-ESS warning (gate iii); the variance term is **not** optional when
        events can sit in the tail of ``n(·|θ)``.
    """
    log_w = model_log_weights
    n_samples = log_w.shape[0]
    n_events = len(event_log_likelihood_fns)

    # ---- per-event evidence  I_i = ∫ L_i n dx ≈ Σ_j w_j L_i(x_j) ----------
    # Python loop over the (statically many) events, mirroring the discrete
    # mode's loop over PE batches.  Each L_i may be a distinct callable, so we
    # do not vmap across events.
    log_I_list = []
    log_sq_list = []
    for L_i in event_log_likelihood_fns:
        log_L = L_i(model_samples)  # (n_samples,)  log-likelihood values
        a = log_w + log_L  # (n_samples,)  log(w_j · L_i(x_j))
        log_I_list.append(jax.nn.logsumexp(a))  # log Σ_j w_j L_i(x_j)
        log_sq_list.append(jax.nn.logsumexp(2.0 * a))  # log Σ_j (w_j L_i)^2

    log_I = jnp.stack(log_I_list)  # (n_events,)
    log_sq = jnp.stack(log_sq_list)  # (n_events,)

    # Effective sample size of each importance estimate: (Σ a)^2 / Σ a^2.
    log_ess = 2.0 * log_I - log_sq
    ess = jnp.exp(log_ess)
    # Relative variance of the (non-self-normalised) IS estimator of I_i, with
    # the finite-sample correction, exactly as the discrete mode forms it.
    rel_variance = jnp.exp(log_sq - 2.0 * log_I) - 1.0 / n_samples

    total_ln_l = jnp.sum(log_I)

    # ---- Poisson mean  μ = T_obs Σ_j w_j pdet(x_j)  (sample-based) ---------
    if pdet_fn is None:
        log_mu = jnp.log(T_obs) + jax.nn.logsumexp(log_w)
    else:
        log_pdet = jnp.log(pdet_fn(model_samples))  # (n_samples,)
        log_mu = jnp.log(T_obs) + jax.nn.logsumexp(log_w + log_pdet)
    expected_rate = jnp.exp(log_mu)

    # log L = Σ_i log I_i + n_events·log T_obs − μ   (same shape as analytical)
    log_likelihood = total_ln_l + n_events * jnp.log(T_obs) - expected_rate

    if variance_cut_threshold is not None:
        total_variance = jnp.nan_to_num(
            jnp.sum(rel_variance),
            nan=jnp.inf,
            posinf=jnp.inf,
            neginf=jnp.inf,
        )
        log_likelihood -= variance_tapering_fn(total_variance, variance_cut_threshold)

    diagnostics: Dict[str, Array] = {
        "log_evidence_per_event": log_I,
        "ess_per_event": ess,
        "rel_variance_per_event": rel_variance,
        "expected_rate": expected_rate,
    }
    return log_likelihood, diagnostics


def low_ess_events(
    ess_per_event: Array,
    n_samples: int,
    frac: float = 0.1,
) -> Array:
    r"""Host-side helper: indices of events whose IS estimate is under-resolved.

    An event whose true parameters sit in the tail of ``n(·|θ)`` receives very
    few effective model samples, so its evidence estimate ``I_i`` is noisy.
    Returns the integer indices where ``ess_per_event[i] < frac · n_samples``.
    Intended for a non-jitted diagnostic pass (logging / warnings), not inside
    the likelihood itself.
    """
    import numpy as _np

    ess = _np.asarray(ess_per_event)
    return _np.nonzero(ess < frac * n_samples)[0]
