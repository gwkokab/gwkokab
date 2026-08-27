# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""NumPyro model over per-event Gaussian summaries.

The counterpart of
:mod:`~gwkokab.inference.numpyro_discrete_poisson_likelihood` for the analytical
GWalk data representation, where each event is a mean and covariance rather than a
cloud of posterior samples.

When ``priors`` is a
:class:`~gwkokab.models.utils.LazyJointDistribution`, the variables are sampled in
two passes: the independent ones first, then the lazy ones in topological order,
each built from the values it depends on. NumPyro needs this because every variable
must be an individually named ``sample`` site.
"""

from collections.abc import Callable
from typing import Any, Dict

import jax
import numpyro
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import (
    analytical_gwalk_poisson_likelihood_fn,
)
from gwkokab.models.utils import JointDistribution, LazyJointDistribution, ScaledMixture


__all__ = ["numpyro_analytical_gwalk_poisson_likelihood"]


def numpyro_analytical_gwalk_poisson_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables: Dict[str, Distribution],
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], tuple[Array, Array]],
    variance_cut_threshold: float | None,
) -> Callable[[Array, Array, Dict[str, Any]], None]:
    r"""Build the NumPyro model for the analytical GWalk data representation.

    The likelihood is that of the inhomogeneous Poisson process,

    .. math::
        \log\mathcal{L}(\Lambda) \propto -\mu(\Lambda)
        + \sum_{n=1}^N \log \int \ell_n(\lambda)\,\rho(\lambda\mid\Lambda)
        \,\mathrm{d}\lambda

    where :math:`\rho(\lambda\mid\Lambda) =
    \mathrm{d}N/\mathrm{d}V\,\mathrm{d}t\,\mathrm{d}\lambda` is the merger rate
    density of a population parameterised by :math:`\Lambda`, :math:`\mu(\Lambda)` is
    the expected number of detections for that population, and :math:`\ell_n(\lambda)`
    is the likelihood for the :math:`n`-th observed event's parameters.

    with the per-event integral evaluated over points resampled from each event's
    Gaussian summary.

    Parameters
    ----------
    dist_fn : Callable[..., Distribution]
        Builds the population model from the sampled variables and the constants.
    priors : JointDistribution
        Joint prior over the sampled variables, ordered as ``variables_index`` says.
    variables : Dict[str, Distribution]
        The sampled variables and their priors.
    constant_params : Dict[str, Any]
        Hyper-parameters frozen by the prior configuration rather than sampled.
    variables_index : Dict[str, int]
        Maps each variable name to its position in the flat parameter vector. This is
        the ordering recorded as ``variables_index`` in the output HDF5.
    poisson_mean_estimator : Callable
        Estimator of :math:`\mu(\Lambda)`, returning ``(mean, variance)``. See
        :mod:`gwkokab.poisson_mean`.
    variance_cut_threshold : float | None
        Threshold above which the Monte Carlo variance of the estimate is penalised by
        :func:`~gwkokab.inference.poissonlikelihood_utils.variance_tapering_fn`.
        :data:`None` disables the penalty.

    Returns
    -------
    Callable
        A NumPyro model taking the event data and registering the likelihood as a
        ``factor`` site.
    """
    if is_lazy_prior := isinstance(priors, LazyJointDistribution):
        dependencies = priors.dependencies
        partial_order = priors.partial_order
    del priors

    sorted_variables = sorted(variables.items(), key=lambda x: x[0])

    def log_likelihood_fn(
        samples_stack: Array,
        ln_offsets: Array,
        pmean_kwargs: Dict[str, Any],
    ):
        """The NumPyro model: sample the hyper-parameters and score the data.

        When ``priors`` is a
        :class:`~gwkokab.models.utils.LazyJointDistribution`, the variables are sampled in
        two passes: the independent ones first, then the lazy ones in topological order,
        each built from the values it depends on. NumPyro needs this because every variable
        must be an individually named ``sample`` site.

        Parameters
        ----------
        samples_stack : Array
            Resampled event coordinates, of shape ``(n_events, n_samples, n_parameters)``.
        ln_offsets : Array
            Log weight of each resampled point.
        pmean_kwargs : Dict[str, Any]
            Extra arguments for the Poisson mean estimator; must include ``T_obs``.

        Returns
        -------
        None
            The log-likelihood is registered with :func:`numpyro.factor` rather than returned.
        """
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

        model_instance = dist_fn(**constant_params, **mapped_params)

        log_likelihood = analytical_gwalk_poisson_likelihood_fn(
            model_instance,
            poisson_mean_estimator,
            samples_stack,
            ln_offsets,
            pmean_kwargs,
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
