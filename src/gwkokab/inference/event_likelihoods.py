# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Per-event log-likelihood evaluators for the sample-mode population likelihood.

:func:`gwkokab.inference.poissonlikelihood_utils.sampled_poisson_likelihood_fn`
consumes one duck-typed callable per event,

    ``L_i: (n_samples, n_dim) -> (n_samples,)``  (log-likelihood values),

evaluated at samples drawn from the population model.  This module provides

* :class:`GaussianEventLikelihood` / :func:`gaussian_event_log_likelihoods` — an
  analytic multivariate-normal event likelihood used as the closed-form CI
  reference (the analytical-Poisson path and the sample path then agree exactly
  in the large-sample limit; see the TP1 acceptance gate).
* :class:`RIFTMarginalLikelihood` — the production adapter interface: it wraps a
  per-event RIFT marginal-likelihood **interpolant** (a callable produced by the
  RIFT pipeline) as an ``L_i``.  This is the thread-back hook; the interpolant
  itself is loaded engine-side and is intentionally not built here.
"""

from collections.abc import Sequence
from typing import Callable

import jax
from jax import numpy as jnp
from jaxtyping import Array


__all__ = [
    "GaussianEventLikelihood",
    "gaussian_event_log_likelihoods",
    "RIFTMarginalLikelihood",
]


class GaussianEventLikelihood:
    r"""Analytic multivariate-normal per-event log-likelihood.

    ``L_i(x) = log N(x; mean, cov)``.  With a Gaussian population intensity the
    per-event evidence ``I_i = ∫ L_i(x) n(x|θ) dx`` is available in closed form,
    which is what makes this the exact CI reference for the two likelihood modes.

    Parameters
    ----------
    mean : Array
        ``(n_dim,)`` event central value.
    cov : Array
        ``(n_dim, n_dim)`` covariance, or ``(n_dim,)`` diagonal, or scalar
        (isotropic) measurement covariance.
    """

    def __init__(self, mean: Array, cov: Array):
        self.mean = jnp.atleast_1d(jnp.asarray(mean, dtype=float))
        n_dim = self.mean.shape[0]
        cov = jnp.asarray(cov, dtype=float)
        if cov.ndim == 0:
            cov = cov * jnp.eye(n_dim)
        elif cov.ndim == 1:
            cov = jnp.diag(cov)
        self.cov = cov

    def __call__(self, samples: Array) -> Array:
        samples = jnp.atleast_2d(samples)
        return jax.scipy.stats.multivariate_normal.logpdf(
            samples, self.mean, self.cov
        )


def gaussian_event_log_likelihoods(
    means: Array,
    covs: Array,
) -> tuple[GaussianEventLikelihood, ...]:
    r"""Build a tuple of :class:`GaussianEventLikelihood` for a mock catalogue.

    Parameters
    ----------
    means : Array
        ``(n_events, n_dim)`` per-event central values.
    covs : Array
        ``(n_events, n_dim, n_dim)`` covariances, or ``(n_events, n_dim)``
        diagonals, or ``(n_events,)`` isotropic scalars.

    Returns
    -------
    tuple of GaussianEventLikelihood
        One evaluator per event, ready to pass as ``event_log_likelihood_fns``.
    """
    means = jnp.atleast_2d(means)
    n_events = means.shape[0]
    return tuple(
        GaussianEventLikelihood(means[i], covs[i]) for i in range(n_events)
    )


class RIFTMarginalLikelihood:
    r"""Adapter wrapping a per-event RIFT marginal-likelihood interpolant.

    The production path of ``demo/PLAN_gwkokab_sampled_likelihood.md``: RIFT
    produces, per event, an interpolant / fit of the marginal likelihood over
    the binary parameters ``x``.  Wrapping that interpolant as an ``L_i`` lets
    the population be constrained directly from model **samples**, with no
    model density and no KDE.

    This class is the **interface only** — pass an already-built interpolant
    callable (engine-side loading lives in the bridge, per the thread-back
    increment).  ``interpolant`` must accept ``(n_samples, n_dim)`` and return
    ``(n_samples,)`` *log* marginal-likelihood values; if it returns linear
    likelihood, set ``log_input=False`` and it is logged here.

    Parameters
    ----------
    interpolant : Callable
        The per-event RIFT interpolant, ``(n_samples, n_dim) -> (n_samples,)``.
    log_input : bool, default True
        Whether ``interpolant`` already returns log-likelihood.
    """

    def __init__(self, interpolant: Callable[[Array], Array], log_input: bool = True):
        if not callable(interpolant):
            raise TypeError(
                "RIFTMarginalLikelihood requires a callable per-event interpolant "
                "(n_samples, n_dim) -> (n_samples,). Build it from the RIFT fit "
                "engine-side and pass it here."
            )
        self.interpolant = interpolant
        self.log_input = log_input

    def __call__(self, samples: Array) -> Array:
        values = self.interpolant(jnp.atleast_2d(samples))
        return values if self.log_input else jnp.log(values)


def stack_event_log_likelihoods(
    event_log_likelihood_fns: Sequence[Callable[[Array], Array]],
) -> Callable[[Array], Array]:
    r"""Optional helper: fold a tuple of evaluators into one batched callable.

    Returns ``L: (n_samples, n_dim) -> (n_events, n_samples)`` by stacking the
    per-event evaluators.  Useful when every event shares the same functional
    form and a single ``vmap`` is cheaper than the Python loop; the core
    likelihood accepts either the tuple or — via a one-line wrapper around this
    — a batched form.
    """

    def batched(samples: Array) -> Array:
        return jnp.stack([L_i(samples) for L_i in event_log_likelihood_fns], axis=0)

    return batched
