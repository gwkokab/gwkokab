# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import Any, Dict, Tuple

import jax
import numpyro
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import (
    mixed_poisson_likelihood_fn,
)
from gwkokab.models.utils import JointDistribution, LazyJointDistribution


__all__ = ["numpyro_mixed_poisson_likelihood"]


def numpyro_mixed_poisson_likelihood(
    sampler_fn: Callable[..., tuple[Array, Array]],
    dist_fn: Callable[..., Distribution] | None,
    priors: JointDistribution,
    variables: Dict[str, Distribution],
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    *,
    event_log_likelihood_fns: Sequence[Callable[[Array], Array]],
    data_group: Tuple[Array, ...] = (),
    log_ref_priors_group: Tuple[Array, ...] = (),
    masks_group: Tuple[Array, ...] = (),
    N_pes: Tuple[Array, ...] = (),
    pdet_fn: Callable[[Array], Array] | None = None,
    variance_cut_threshold: float | None = None,
) -> Callable[[Array], None]:
    r"""Mixed-mode numpyro population likelihood (sampled + discrete, one μ).

    Binds BOTH a ``sampler_fn(**params) -> (samples, log_weights)`` (the model as
    weighted draws — used for the broad/high-mass **sampled** group and for the
    shared Poisson mean μ) AND a ``dist_fn(**params) -> Distribution`` (the model
    *density* — used for the narrow/low-mass **discrete** group's posterior-sample
    reweighting).  A coagulation engine supplies both from one solve at θ
    (``make_coagulation_sampler_fn`` / ``make_coagulation_dist_fn``).  Prior
    plumbing is identical to :func:`numpyro_sampled_poisson_likelihood`; the only
    additions are ``dist_fn`` and the discrete-group event arrays.  Public API is
    additive; the single-mode wrappers are untouched.

    ``dist_fn`` may be ``None`` when ``data_group`` is empty (all events broad),
    in which case this reduces exactly to the sampled wrapper.

    See :func:`gwkokab.inference.poissonlikelihood_utils.mixed_poisson_likelihood_fn`
    for the estimator and the per-event-ESS split criterion.
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
                partial_variables_samples[i] = numpyro.sample(parameter_name, prior_dist)
            variables_samples = partial_variables_samples  # type: ignore[assignment]
        else:
            variables_samples = [
                numpyro.sample(parameter_name, prior_dist)
                for parameter_name, prior_dist in sorted_variables
            ]

        mapped_params = {
            name: variables_samples[i] for name, i in variables_index.items()
        }

        # Same population θ in both representations.
        model_samples, model_log_weights = sampler_fn(**constant_params, **mapped_params)
        model_instance = (
            dist_fn(**constant_params, **mapped_params) if dist_fn is not None else None
        )

        log_likelihood, _diagnostics = mixed_poisson_likelihood_fn(
            event_log_likelihood_fns,
            model_samples,
            model_log_weights,
            pdet_fn,
            model_instance,
            data_group,
            log_ref_priors_group,
            masks_group,
            N_pes,
            T_obs,
            variance_cut_threshold,
        )

        log_likelihood = jnp.nan_to_num(
            log_likelihood, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
        )
        numpyro.factor("log_likelihood", log_likelihood)

    return log_likelihood_fn  # type: ignore[return-value]
