# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Tuple

import equinox as eqx
import jax
from jax import Array, lax, nn as jnn, numpy as jnp, tree as jtr
from numpyro.distributions import Distribution

from ..models.utils import JointDistribution, ScaledMixture
from .bake import Bake


__all__ = ["PoissonLikelihood"]


class PoissonLikelihood(eqx.Module):
    r"""This class is used to provide a likelihood function for the inhomogeneous Poisson
    process. The likelihood is given by,

    .. math::
        \log\mathcal{L}(\Lambda) \propto -\mu(\Lambda)
        +\log\sum_{n=1}^N \int \ell_n(\lambda) \rho(\lambda\mid\Lambda)
        \mathrm{d}\lambda


    where, :math:`\displaystyle\rho(\lambda\mid\Lambda) =
    \frac{\mathrm{d}N}{\mathrm{d}V\mathrm{d}t \mathrm{d}\lambda}` is the merger
    rate density for a population parameterized by :math:`\Lambda`, :math:`\mu(\Lambda)` is
    the expected number of detected mergers for that population, and
    :math:`\ell_n(\lambda)` is the likelihood for the :math:`n`-th observed event's
    parameters. Using Bayes' theorem, we can obtain the posterior
    :math:`p(\Lambda\mid\text{data})` by multiplying the likelihood by a prior
    :math:`\pi(\Lambda)`.

    .. math::
        p(\Lambda\mid\text{data}) \propto \pi(\Lambda) \mathcal{L}(\Lambda)

    The integral inside the main likelihood expression is then evaluated via
    Monte Carlo as

    .. math::
        \int \ell_n(\lambda) \rho(\lambda\mid\Lambda) \mathrm{d}\lambda \propto
        \int \frac{p(\lambda | \mathrm{data}_n)}{\pi_n(\lambda)}
        \rho(\lambda\mid\Lambda) \mathrm{d}\lambda \approx
        \frac{1}{N_{\mathrm{samples}}}
        \sum_{i=1}^{N_{\mathrm{samples}}}
        \frac{\rho(\lambda_{n,i}\mid\Lambda)}{\pi_{n,i}}
    """

    data: Sequence[Array] = eqx.field(static=False)
    model: Callable[..., Distribution] = eqx.field(static=False)
    log_ref_priors: Sequence[Array] = eqx.field(static=False)
    priors: JointDistribution = eqx.field(static=False)
    variables_index: Mapping[str, int] = eqx.field(static=True)
    ERate_fn: Callable[[Distribution | ScaledMixture], Array] = eqx.field(static=False)

    def __init__(
        self,
        model: Bake,
        log_ref_priors: Sequence[Array],
        data: Sequence[Array],
        ERate_fn: Callable[[Distribution | ScaledMixture], Array],
    ) -> None:
        """
        Parameters
        ----------
        model : Bake
            model to be used for the likelihood calculation.
        log_ref_priors : Sequence[Array]
            Log reference priors to be used for the likelihood calculation.
        data : Sequence[Array]
            Data to be used for the likelihood calculation.
        ERate_fn : Callable[[Distribution | ScaledMixture], Array]
            Expected rate function to be used for the likelihood calculation.
        """
        self.data = data
        self.model = model
        self.log_ref_priors = log_ref_priors
        self.ERate_fn = ERate_fn

        dummy_model = model.get_dummy()
        if not isinstance(dummy_model, ScaledMixture):
            warnings.warn(
                "The model provided is not a ScaledMixture. This means rate estimation "
                "will not be possible."
            )

        variables, duplicates, self.model = model.get_dist()
        self.variables_index = {key: i for i, key in enumerate(variables.keys())}

        for key, value in duplicates.items():
            self.variables_index[key] = self.variables_index[value]

        self.priors = JointDistribution(*variables.values(), validate_args=True)

    def log_likelihood(self, x: Array) -> Array:
        """The log likelihood function for the inhomogeneous Poisson process.

        Parameters
        ----------
        x : Array
            Recovered parameters.

        Returns
        -------
        Array
            Log likelihood value for the given parameters.
        """
        mapped_params = {name: x[..., i] for name, i in self.variables_index.items()}

        model: Distribution = self.model(**mapped_params)

        def _nth_prob(y: Tuple[Array, Array]) -> Array:
            """Calculate the likelihood for the nth event.

            Parameters
            ----------
            y : Array
                The data for the nth event.

            Returns
            -------
            Array
                The likelihood for the nth event.
            """
            event_data, log_ref_prior_y = y

            _log_prob = (
                lax.map(model.log_prob, event_data, batch_size=10000) - log_ref_prior_y
            )

            return jnn.logsumexp(
                _log_prob,
                axis=-1,
                where=~jnp.isneginf(_log_prob),  # to avoid nans
            ) - jnp.log(event_data.shape[0])

        log_likelihood = jtr.reduce(
            lambda x, y: x + _nth_prob(y),
            list(zip(self.data, self.log_ref_priors)),
            jnp.zeros(()),
            is_leaf=lambda x: isinstance(x, tuple),
        )

        expected_rates = self.ERate_fn(model)

        return log_likelihood - expected_rates

    def log_posterior(self, x: Array, _: dict) -> Array:
        """The likelihood function for the inhomogeneous Poisson process.

        Parameters
        ----------
        x : Array
            Recovered parameters.
        _ : dict
            Dictionary of additional arguments. (Unused)

        Returns
        -------
        Array
            Log likelihood value for the given parameters.
        """
        log_prior = self.priors.log_prob(x)
        log_likelihood = self.log_likelihood(x)

        log_posterior = log_prior + log_likelihood
        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )
        return log_posterior


def poisson_likelihood(
    model: Bake,
    stacked_data: Array,
    stacked_log_ref_priors: Array,
    ERate_fn: Callable[[Distribution], Array],
    data_shapes: list[int],
) -> Tuple[dict[str, int], JointDistribution, Callable[[Array, Array], Array]]:
    dummy_model = model.get_dummy()
    if not isinstance(dummy_model, ScaledMixture):
        warnings.warn(
            "The model provided is not a ScaledMixture. This means rate estimation "
            "will not be possible."
        )

    variables, duplicates, model = model.get_dist()  # type: ignore
    variables_index = {key: i for i, key in enumerate(variables.keys())}
    for key, value in duplicates.items():
        variables_index[key] = variables_index[value]

    priors = JointDistribution(*variables.values(), validate_args=True)

    # Compute start/stop indices for each event from data_shapes
    data_shapes = list(data_shapes)
    start_indices = [0] + list(jnp.cumsum(jnp.array(data_shapes[:-1])).tolist())
    stop_indices = [start + size for start, size in zip(start_indices, data_shapes)]
    start_stops = list(zip(start_indices, stop_indices))

    @jax.jit
    def likelihood_fn(x: Array, _: Array) -> Array:
        mapped_params = {
            name: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
            for name, i in variables_index.items()
        }

        model_instance: Distribution = model(**mapped_params)

        log_probs = jax.lax.map(
            model_instance.log_prob, stacked_data, batch_size=10_000
        )
        log_probs -= stacked_log_ref_priors

        total_log_likelihood = jnp.zeros(())
        for start, stop in start_stops:
            log_probs_slice = jax.lax.dynamic_slice_in_dim(
                log_probs, start, stop - start, axis=-1
            )
            total_log_likelihood += jnn.logsumexp(
                log_probs_slice, axis=-1, where=~jnp.isneginf(log_probs_slice)
            ) - jnp.log(stop - start)

        expected_rates = ERate_fn(model_instance)
        log_prior = priors.log_prob(x)
        log_likelihood = total_log_likelihood - expected_rates
        log_posterior = log_prior + log_likelihood

        log_posterior = jnp.nan_to_num(
            log_posterior, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
        )

        return jax.block_until_ready(log_posterior)

    return variables_index, priors, likelihood_fn
