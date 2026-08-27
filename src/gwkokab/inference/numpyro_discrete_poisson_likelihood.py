# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""NumPyro model over per-event posterior samples.

Where the flowMC modules return a plain function of a flat vector, NumPyro needs a
*model*: each hyper-parameter is drawn at its own named ``sample`` site and the
likelihood enters as a ``factor``. The site names are the hyper-parameter names
sorted alphabetically, which is the same ordering ``variables_index`` records.

When ``priors`` is a
:class:`~gwkokab.models.utils.LazyJointDistribution`, the variables are sampled in
two passes: the independent ones first, then the lazy ones in topological order,
each built from the values it depends on. NumPyro needs this because every variable
must be an individually named ``sample`` site.
"""

from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

import jax
import numpyro
from jax import Array, numpy as jnp
from numpyro.distributions import Distribution

from gwkokab.inference.poissonlikelihood_utils import discrete_poisson_likelihood_fn

from ..models.utils import JointDistribution, LazyJointDistribution, ScaledMixture


__all__ = ["numpyro_discrete_poisson_likelihood"]


def numpyro_discrete_poisson_likelihood(
    dist_fn: Callable[..., Distribution],
    priors: JointDistribution,
    variables: Dict[str, Distribution],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[[ScaledMixture], Array],
    where_fns: Optional[List[Callable[..., Array]]],
    constants: Dict[str, Array],
    variance_cut_threshold: float | None,
) -> Callable[[Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...]], Array]:
    r"""Build the NumPyro model for the discrete data representation.

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

    with the per-event integral evaluated as a Monte Carlo average over that event's
    posterior samples, reweighted by the PE prior they were drawn under.

    Parameters
    ----------
    dist_fn : Callable[..., Distribution]
        Builds the population model from the sampled variables and the constants.
    priors : JointDistribution
        Joint prior over the sampled variables, ordered as ``variables_index`` says.
    variables : Dict[str, Distribution]
        The sampled variables and their priors.
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
    Callable
        A NumPyro model taking the bucketed event data and registering the likelihood as
        a ``factor`` site.
    """
    if is_lazy_prior := isinstance(priors, LazyJointDistribution):
        dependencies = priors.dependencies
        partial_order = priors.partial_order
    del priors

    sorted_variables = sorted(variables.items(), key=lambda x: x[0])

    def log_likelihood_fn(
        data_group: Tuple[Array, ...],
        log_ref_priors_group: Tuple[Array, ...],
        masks_group: Tuple[Array, ...],
        pmean_kwargs: Dict[str, Any],
        N_pes: Tuple[Array, ...],
    ):
        """The NumPyro model: sample the hyper-parameters and score the data.

        When ``priors`` is a
        :class:`~gwkokab.models.utils.LazyJointDistribution`, the variables are sampled in
        two passes: the independent ones first, then the lazy ones in topological order,
        each built from the values it depends on. NumPyro needs this because every variable
        must be an individually named ``sample`` site.

        Parameters
        ----------
        data_group : Tuple[Array, ...]
            One padded array of posterior samples per bucket of events.
        log_ref_priors_group : Tuple[Array, ...]
            Log PE prior of each sample, bucketed to match.
        masks_group : Tuple[Array, ...]
            Boolean masks marking the real samples in each bucket.
        pmean_kwargs : Dict[str, Any]
            Extra arguments for the Poisson mean estimator; must include ``T_obs``.
        N_pes : Tuple[Array, ...]
            Number of real posterior samples per event, bucketed to match.

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
        model_instance: Distribution = dist_fn(
            **constants, **mapped_params, validate_args=True
        )

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

        if where_fns is not None and len(where_fns) > 0:
            mapped_params = {
                name: variables_samples[i] for name, i in variables_index.items()
            }
            mask = where_fns[0](**constants, **mapped_params)
            for where_fn in where_fns[1:]:
                mask = jnp.logical_and(mask, where_fn(**constants, **mapped_params))

            log_likelihood = jnp.where(
                mask,
                log_likelihood,
                -jnp.inf,  # type: ignore[arg-type]
            )
            log_likelihood = jnp.nan_to_num(
                log_likelihood, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
            )

        numpyro.factor("log_likelihood", log_likelihood)

    return log_likelihood_fn  # type: ignore[return-value]
