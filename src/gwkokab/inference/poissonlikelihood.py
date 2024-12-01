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


from collections.abc import Callable, Mapping, Sequence

import equinox as eqx
from jax import nn as jnn, numpy as jnp, tree as jtr
from jaxtyping import Array
from numpyro.distributions import Distribution

from ..debug import debug_flush
from ..models.utils import JointDistribution, ScaledMixture
from ..parameters import Parameter
from .bake import Bake


__all__ = ["PoissonLikelihood"]


class PoissonLikelihood(eqx.Module):
    r"""This class is used to provide a likelihood function for the inhomogeneous
    Poisson process. The likelihood is given by,

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

    :param custom_vt: Custom VT function to use.
    :param time: Time interval for the Poisson process.
    """

    parameters: Sequence[Parameter]
    data: Sequence[Array]
    model: Callable[..., Distribution]
    ref_priors: JointDistribution
    priors: JointDistribution
    variables_index: Mapping[str, int]
    log_weights: Array
    samples: Array

    def __init__(
        self,
        model: Bake,
        parameters: Sequence[Parameter],
        data: Sequence[Array],
        log_weights: Array,
        samples: Array,
        time: float = 1.0,
    ) -> None:
        self.data = data
        self.model = model
        self.parameters = parameters
        # weights and time are multiplied to each other in the end.
        # This is done one time to avoid a single multiplication for each iteration
        # in the expected rate calculation.
        self.log_weights = log_weights + jnp.log(time)
        self.samples = samples

        dummy_model = model.get_dummy()
        assert isinstance(
            dummy_model, ScaledMixture
        ), "Model must be a scaled mixture model."

        variables, duplicates, self.model = model.get_dist()
        self.variables_index = {key: i for i, key in enumerate(variables.keys())}

        for key, value in duplicates.items():
            self.variables_index[key] = self.variables_index[value]

        self.ref_priors = JointDistribution(
            *map(lambda x: x.prior, self.parameters), validate_args=True
        )
        self.priors = JointDistribution(*variables.values(), validate_args=True)

    def log_likelihood(self, x: Array) -> Array:
        """The log likelihood function for the inhomogeneous Poisson process.

        :param x: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        mapped_params = {name: x[..., i] for name, i in self.variables_index.items()}

        debug_flush("mapped params: {mp}", mp=mapped_params)

        model: ScaledMixture = self.model(**mapped_params)

        log_likelihood = jtr.reduce(
            lambda x, y: x
            + jnn.logsumexp(model.log_prob(y) - self.ref_priors.log_prob(y), axis=-1)
            - jnp.log(y.shape[0]),
            self.data,
            jnp.zeros(()),
            is_leaf=lambda x: isinstance(x, Array),
        )

        debug_flush("model_log_likelihood: {mll}", mll=log_likelihood)

        expected_rates = jnp.mean(
            jnp.exp(self.log_weights + model.log_prob(self.samples))
        )

        debug_flush("expected_rate={expr}", expr=expected_rates)

        return log_likelihood - expected_rates

    def log_posterior(self, x: Array, _: dict) -> Array:
        r"""The likelihood function for the inhomogeneous Poisson process.

        .. math::
            \log p(\Lambda\mid\text{data}) \propto \log\pi(\Lambda) + \log\mathcal{L}(\Lambda)

        :param x: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        log_prior = self.priors.log_prob(x)
        log_likelihood = self.log_likelihood(x)
        debug_flush(
            "log_prior: {lp}\nlog_likelihood: {ll}", lp=log_prior, ll=log_likelihood
        )
        return log_prior + log_likelihood
