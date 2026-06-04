# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import Any, Dict

import jax
import numpyro
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import (
    sampled_poisson_likelihood_fn,
)
from gwkokab.models.utils import JointDistribution, LazyJointDistribution


__all__ = ["numpyro_sampled_poisson_likelihood"]


def numpyro_sampled_poisson_likelihood(
    sampler_fn: Callable[..., tuple[Array, Array]],
    priors: JointDistribution,
    variables: Dict[str, Distribution],
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    event_log_likelihood_fns: Sequence[Callable[[Array], Array]],
    pdet_fn: Callable[[Array], Array] | None,
    variance_cut_threshold: float | None,
) -> Callable[[Array], None]:
    r"""Sample-mode (likelihood-mode) numpyro population likelihood.

    The dual of :func:`numpyro_analytical_poisson_likelihood`.  Where the
    analytical wrapper binds a ``dist_fn(**params) -> Distribution`` (a pointwise
    model **density**) and is handed each event's PE samples, this wrapper binds
    a ``sampler_fn(**params) -> (samples, log_weights)`` (the model as weighted
    **draws**) and is handed a per-event **likelihood evaluator** for each event.
    This is what a particle Monte-Carlo population engine supplies natively — no
    grid, no KDE — see ``demo/PLAN_gwkokab_sampled_likelihood.md``.

    Plumbing is identical to the analytical path: the same ``priors`` /
    ``variables`` / ``variables_index`` ``sorted``-order contract maps numpyro
    prior draws onto the model's keyword arguments.  The only swaps are
    ``dist_fn → sampler_fn`` and (model density at event samples) → (event
    likelihoods at model samples).  Public API is additive; the two density
    modes are untouched.

    Parameters
    ----------
    sampler_fn : Callable
        ``(**constant_params, **mapped_params) -> (samples, log_weights)`` with
        ``samples`` shape ``(n_samples, n_dim)`` and ``log_weights`` shape
        ``(n_samples,)``.  Must be reparameterised (AD-traceable) for gradients.
    priors, variables, constant_params, variables_index
        As in :func:`numpyro_analytical_poisson_likelihood`.
    event_log_likelihood_fns : Sequence[Callable]
        One ``L_i: (n_samples, n_dim) -> (n_samples,)`` log-likelihood evaluator
        per event (the "data").  Bound at construction (they are static,
        per-analysis objects); the returned model takes only ``T_obs``.
    pdet_fn : Callable or None
        Detection-probability evaluator for the sample-based Poisson mean; ``None``
        ⇒ flat selection ``VT ≡ 1``.
    variance_cut_threshold : float or None
        Optional importance-sampling variance tapering.

    Returns
    -------
    Callable
        ``log_likelihood_fn(T_obs)`` — a numpyro model registering
        ``numpyro.factor("log_likelihood", ...)``.
    """
    if is_lazy_prior := isinstance(priors, LazyJointDistribution):
        dependencies = priors.dependencies
        partial_order = priors.partial_order
    del priors

    sorted_variables = sorted(variables.items(), key=lambda x: x[0])

    def log_likelihood_fn(T_obs: Array):
        if is_lazy_prior:
            partial_variables_samples = [
                numpyro.sample(parameter_name, prior_dist)
                if isinstance(prior_dist, Distribution)
                else (parameter_name, prior_dist)
                for parameter_name, prior_dist in sorted_variables
            ]

            for i in partial_order:
                kwargs = {
                    k: partial_variables_samples[v] for k, v in dependencies[i].items()
                }
                parameter_name, prior_dist_fn = partial_variables_samples[i]
                if isinstance(prior_dist_fn, jax.tree_util.Partial):
                    prior_dist = prior_dist_fn.func(
                        *prior_dist_fn.args, **prior_dist_fn.keywords, **kwargs
                    )  # type: ignore[arg-type]
                else:
                    prior_dist = prior_dist_fn  # type: ignore[assignment]
                partial_variables_samples[i] = numpyro.sample(
                    parameter_name, prior_dist
                )

            variables_samples = partial_variables_samples  # type: ignore[assignment]
        else:
            variables_samples = [
                numpyro.sample(parameter_name, prior_dist)
                for parameter_name, prior_dist in sorted_variables
            ]

        mapped_params = {
            name: variables_samples[i] for name, i in variables_index.items()
        }

        # The model as weighted draws, instead of a pointwise density.
        model_samples, model_log_weights = sampler_fn(
            **constant_params, **mapped_params
        )

        log_likelihood, _diagnostics = sampled_poisson_likelihood_fn(
            event_log_likelihood_fns,
            model_samples,
            model_log_weights,
            pdet_fn,
            T_obs,
            variance_cut_threshold,
        )

        log_likelihood = jnp.nan_to_num(
            log_likelihood,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        numpyro.factor("log_likelihood", log_likelihood)

    return log_likelihood_fn  # type: ignore[return-value]
