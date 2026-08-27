# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""FlowMC log-posterior over per-event posterior samples.

flowMC works with an explicit flat parameter vector and a plain
``log_posterior(x, data)`` callable, so this module maps the vector back to named
hyper-parameters through ``variables_index``, builds the population model, adds the
log prior to the log likelihood, and JIT-compiles the result.
"""

from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

import equinox as eqx
import jax
from jax import Array, numpy as jnp
from numpyro.distributions.distribution import Distribution

from gwkokab.inference.poissonlikelihood_utils import discrete_poisson_likelihood_fn

from ..models.utils import JointDistribution, ScaledMixture


__all__ = ["flowMC_discrete_poisson_likelihood"]


def flowMC_discrete_poisson_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables: Dict[str, Distribution],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    where_fns: Optional[List[Callable[..., Array]]],
    constants: Dict[str, Array],
    variance_cut_threshold: float | None,
) -> Callable[[Array, Dict[str, Any]], Array]:
    r"""Build the flowMC log-posterior for the discrete data representation.

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

    Using Bayes' theorem, the posterior :math:`p(\Lambda\mid\text{data})` follows by
    multiplying by a prior :math:`\pi(\Lambda)`:

    .. math::
        p(\Lambda\mid\text{data}) \propto \pi(\Lambda)\,\mathcal{L}(\Lambda)

    The integral inside the likelihood is evaluated by Monte Carlo as

    .. math::
        \int \ell_n(\lambda)\,\rho(\lambda\mid\Lambda)\,\mathrm{d}\lambda \propto
        \int \frac{p(\lambda \mid \mathrm{data}_n)}{\pi_n(\lambda)}
        \rho(\lambda\mid\Lambda)\,\mathrm{d}\lambda \approx
        \frac{1}{N_{\mathrm{samples}}}
        \sum_{i=1}^{N_{\mathrm{samples}}}
        \frac{\rho(\lambda_{n,i}\mid\Lambda)}{\pi_{n,i}}

    Parameters
    ----------
    dist_fn : Callable[..., Distribution]
        Builds the population model from the sampled variables and the constants.
    priors : JointDistribution
        Joint prior over the sampled variables, ordered as ``variables_index`` says.
    variables : Dict[str, Distribution]
        Unused here; flowMC takes its priors from ``priors`` as a joint distribution
        rather than site by site.
    variables_index : Dict[str, int]
        Maps each variable name to its position in the flat parameter vector. This is
        the ordering recorded as ``variables_index`` in the output HDF5.
    poisson_mean_estimator : Callable
        Estimator of :math:`\mu(\Lambda)`, returning ``(mean, variance)``. See
        :mod:`gwkokab.poisson_mean`.
    where_fns : Optional[List[Callable[..., Array]]]
        Extra validity predicates on the hyper-parameters, beyond the prior support.
        Points failing any of them get :math:`-\infty`. :data:`None` skips the check
        entirely.
    constants : Dict[str, Array]
        Hyper-parameters frozen by the prior configuration rather than sampled.
    variance_cut_threshold : float | None
        Threshold above which the Monte Carlo variance of the estimate is penalised by
        :func:`~gwkokab.inference.poissonlikelihood_utils.variance_tapering_fn`.
        :data:`None` disables the penalty.

    Returns
    -------
    Callable[[Array, Dict[str, Any]], Array]
        A JIT-compiled ``log_posterior(x, data)``. Non-finite values are mapped to
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

    def log_posterior_fn(
        x: Array, data: Dict[str, Tuple[Array, ...] | Dict[str, Any]]
    ) -> Array:
        r"""Evaluate the log posterior at a point.

        Parameters
        ----------
        x : Array
            The flat parameter vector.
        data : Dict[str, Tuple[Array, ...] | Dict[str, Any]]
            The bucketed event data, keyed ``data_group``, ``log_ref_priors_group``,
            ``masks_group``, ``pmean_kwargs`` and ``N_pes``.

        Returns
        -------
        Array
            The scalar log posterior, with non-finite values mapped to :math:`-\infty`.
        """
        data_group: Tuple[Array, ...] = data["data_group"]  # type: ignore
        log_ref_priors_group: Tuple[Array, ...] = data["log_ref_priors_group"]  # type: ignore
        masks_group: Tuple[Array, ...] = data["masks_group"]  # type: ignore
        pmean_kwargs: Dict[str, Any] = data["pmean_kwargs"]  # type: ignore
        N_pes = data["N_pes"]  # type: ignore

        mapped_params = _map_params(x)

        model_instance = dist_fn(**constants, **mapped_params)

        log_likelihood = discrete_poisson_likelihood_fn(
            model_instance,
            poisson_mean_estimator,
            data_group,
            log_ref_priors_group,
            masks_group,
            pmean_kwargs,
            N_pes,
            variance_cut_threshold,
        )

        # log π(ω)
        log_prior = priors.log_prob(x)

        # log p(ω|data) = log π(ω) + log L(ω)
        log_posterior = log_prior + log_likelihood

        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )

        return log_posterior

    if where_fns is None:
        return eqx.filter_jit(log_posterior_fn)

    def log_posterior_fn_with_checks(
        x: Array, data: Dict[str, Tuple[Array, ...]]
    ) -> Array:
        r"""Evaluate the log posterior, first screening the point for validity.

        The point must lie in the prior support, satisfy every ``where_fn``, and be finite;
        otherwise :math:`-\infty` is returned without evaluating the model at all.

        Parameters
        ----------
        x : Array
            The flat parameter vector.
        data : Dict[str, Tuple[Array, ...]]
            The bucketed event data.

        Returns
        -------
        Array
            The scalar log posterior, or :math:`-\infty` for an invalid point.
        """
        mapped_params = _map_params(x)
        predicate = priors.support.check(x)
        for where_fn in where_fns:  # type: ignore
            predicate = jnp.logical_and(
                predicate, where_fn(**constants, **mapped_params)
            )
        predicate = jnp.logical_and(jnp.all(jnp.isfinite(x)), predicate)
        return jnp.where(predicate, log_posterior_fn(x, data), -jnp.inf)

    return eqx.filter_jit(log_posterior_fn_with_checks)
