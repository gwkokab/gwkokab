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


from typing_extensions import Optional, Sequence

from jax import nn as jnn, numpy as jnp, tree as jtr
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array

from ..debug import debug_flush
from ..models.utils import JointDistribution, ScaledMixture
from ..parameters import Parameter
from .bake import Bake


__all__ = ["poisson_likelihood"]


@register_pytree_node_class
class PoissonLikelihood:
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

    time: float = 1.0

    def tree_flatten(self) -> tuple:
        children = ()
        aux_data_keys = [
            "model",
            "ref_priors",
            "time",
            "variables_index",
            "variables",
            "priors",
        ]
        aux_data = {k: getattr(self, k) for k in aux_data_keys}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple) -> "PoissonLikelihood":
        del children
        obj = cls.__new__(cls)
        for k, v in aux_data.items():
            if v is not None:
                setattr(obj, k, v)
        PoissonLikelihood.__init__(obj)
        return obj

    def set_model(
        self,
        /,
        params: Optional[Sequence[Parameter]] = None,
        *,
        model: Optional[Bake] = None,
    ) -> None:
        assert model is not None, "Model must be provided."
        assert params is not None, "Params must be provided."
        dummy_model = model.get_dummy()
        assert isinstance(
            dummy_model, ScaledMixture
        ), "Model must be a scaled mixture model."

        self.variables, duplicates, self.model = model.get_dist()
        self.variables_index = {key: i for i, key in enumerate(self.variables.keys())}

        for key, value in duplicates.items():
            self.variables_index[key] = self.variables_index[value]

        self.ref_priors = JointDistribution(*map(lambda x: x.prior, params))
        self.priors = JointDistribution(*self.variables.values())

    def log_likelihood(self, x: Array, data: dict) -> Array:
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
            data["data"],
            jnp.zeros(()),
        )

        debug_flush("model_log_likelihood: {mll}", mll=log_likelihood)

        vt_samples = data["vt_samples"]
        log_prob_vt = model.log_prob(vt_samples)
        expected_rates = self.time * jnp.mean(jnp.exp(log_prob_vt))

        debug_flush("expected_rate={expr}", expr=expected_rates)

        return log_likelihood - expected_rates

    def log_posterior(self, x: Array, data: dict) -> Array:
        r"""The likelihood function for the inhomogeneous Poisson process.

        .. math::
            \log p(\Lambda\mid\text{data}) \propto \log\pi(\Lambda) + \log\mathcal{L}(\Lambda)

        :param x: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        log_prior = self.priors.log_prob(x)
        debug_flush("log_prior: {lp}", lp=log_prior)
        log_likelihood = self.log_likelihood(x, data)
        debug_flush("log_likelihood: {lp}", lp=log_prior)
        return log_prior + log_likelihood


poisson_likelihood = PoissonLikelihood()
