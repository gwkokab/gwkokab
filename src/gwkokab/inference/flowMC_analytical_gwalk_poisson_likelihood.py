# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""FlowMC log-posterior over per-event Gaussian summaries.

The counterpart of
:mod:`~gwkokab.inference.flowMC_discrete_poisson_likelihood` for the analytical
GWalk data representation, where each event is a mean and covariance rather than a
cloud of posterior samples.
"""

from typing import Any, Callable, Dict

import jax
from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import (
    analytical_gwalk_poisson_likelihood_fn,
)
from gwkokab.models.utils import JointDistribution, ScaledMixture


__all__ = ["flowMC_analytical_gwalk_poisson_likelihood"]


def flowMC_analytical_gwalk_poisson_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables: Dict[str, Distribution],
    constant_params: Dict[str, Any],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], tuple[Array, Array]],
    variance_cut_threshold: float | None,
) -> Callable[[Array, Dict[str, Any]], Array]:
    r"""Build the flowMC log-posterior for the analytical GWalk data representation.

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
        Unused here; flowMC takes its priors from ``priors`` as a joint distribution
        rather than site by site.
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
    Callable[[Array, Dict[str, Any]], Array]
        A ``log_posterior(x, data)`` callable. Non-finite values are mapped to
        :math:`-\infty`, so the sampler rejects such points rather than propagating NaN.
    """
    del variables

    def _map_params(x: Array) -> Dict[str, Array]:
        """Split the flat parameter vector into named hyper-parameters.

        Parameters
        ----------
        x : Array
            The flat parameter vector.

        Returns
        -------
        Dict[str, Array]
            The hyper-parameters, keyed by name.
        """
        return {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

    def log_posterior_fn(x: Array, data: Dict[str, Any]) -> Array:
        r"""Evaluate the log posterior at a point.

        Parameters
        ----------
        x : Array
            The flat parameter vector.
        data : Dict[str, Any]
            The event data, keyed ``samples_stack``, ``ln_offsets`` and ``pmean_kwargs``.

        Returns
        -------
        Array
            The scalar log posterior, with non-finite values mapped to :math:`-\infty`.
        """
        ln_offsets = data["ln_offsets"]
        pmean_kwargs = data["pmean_kwargs"]
        samples_stack = data["samples_stack"]

        mapped_params = _map_params(x)
        model_instance = dist_fn(**constant_params, **mapped_params)

        log_likelihood = analytical_gwalk_poisson_likelihood_fn(
            model_instance,
            poisson_mean_estimator,
            samples_stack,
            ln_offsets,
            pmean_kwargs,
            variance_cut_threshold,
        )

        log_posterior = priors.log_prob(x) + log_likelihood

        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        return log_posterior

    return log_posterior_fn
