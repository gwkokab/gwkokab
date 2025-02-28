# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Tuple

import equinox as eqx
import jax
from jax import Array, nn as jnn, numpy as jnp, tree as jtr
from numpyro.distributions import Distribution

from ..logger import logger
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
                jax.lax.map(model.log_prob, event_data, batch_size=1000)
                - log_ref_prior_y
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

        logger.debug(
            "PoissionLikelihood: \n"
            "\tmapped_params = {mp}\n"
            "\tmodel_log_likelihood = {mll}\n"
            "\texpected_rate = {expr}",
            mp=mapped_params,
            mll=log_likelihood,
            expr=expected_rates,
        )

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
        logger.debug(
            "PoissionLikelihood:\n\tlog_prior + log_likelihood = {lp} + {ll}",
            lp=log_prior,
            ll=log_likelihood,
        )
        log_posterior = log_prior + log_likelihood
        log_posterior = jnp.nan_to_num(
            log_posterior,
            nan=-jnp.inf,
            posinf=-jnp.inf,
            neginf=-jnp.inf,
        )
        return log_posterior
