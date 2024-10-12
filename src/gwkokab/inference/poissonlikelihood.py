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


from typing_extensions import Callable, List, Optional, Sequence, Union

import numpy as np
from jax import numpy as jnp, random as jrd, tree as jtr
from jax.nn import logsumexp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int, PRNGKeyArray
from numpyro.distributions import Distribution, Uniform

from ..debug import debug_flush
from ..models.utils import JointDistribution, ScaledMixture
from ..parameters import DEFAULT_PRIORS, Parameter
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
    :param is_multi_rate_model: Flag to indicate if the model is multi-rate.
    :param logVT: Log of the VT function.
    :param time: Time interval for the Poisson process.
    :param vt_method: Method to use for the VT function. Options are `uniform`, `model`, and `custom`.
    :param vt_params: Parameters for the VT function.
    """

    _vt_method: Optional[str] = None
    _vt_params: Optional[Union[Parameter, Sequence[str], Sequence[Parameter]]] = None
    custom_vt: Optional[Callable[[Int, PRNGKeyArray, Distribution], Array]] = None
    is_multi_rate_model: bool = False
    logVT: Optional[Callable[[], Array]] = None
    scale_factor: Float = 1.0
    time: Float = 1.0

    def tree_flatten(self) -> tuple:
        children = ()
        aux_data_keys = [
            "_vt_method",
            "_vt_params",
            "custom_vt",
            "is_multi_rate_model",
            "logVT",
            "model",
            "ref_priors",
            "time",
            "scale_factor",
            "total_pop",
            "variables_index",
            "variables",
            "vt_params_index",
            "vt_params_unif_rvs",
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

    @property
    def vt_params(
        self,
    ) -> Optional[Union[Parameter, Sequence[str], Sequence[Parameter]]]:
        return self._vt_params

    @vt_params.setter
    def vt_params(
        self, params: Union[Parameter, Sequence[str], Sequence[Parameter]]
    ) -> None:
        """Pre-process the parameters before setting the model."""
        if isinstance(params, Parameter):
            params = (params.name,)
        elif all(isinstance(param, Parameter) for param in params):
            params = tuple(map(lambda x: x.name, params))
        self._vt_params = params

    @property
    def vt_method(self) -> Optional[str]:
        return self._vt_method

    @vt_method.setter
    def vt_method(self, method: str) -> None:
        assert method in ["uniform", "model", "custom"], "Invalid VT method."
        self._vt_method = method

    def set_model(
        self,
        /,
        params: Optional[Sequence[Parameter]] = None,
        # log_rates_prior: Optional[Distribution | Sequence[Distribution]] = None,
        *,
        model: Optional[Bake] = None,
    ) -> None:
        assert model is not None, "Model must be provided."
        # assert log_rates_prior is not None, "Rate prior must be provided."
        assert params is not None, "Params must be provided."
        dummy_model = model.get_dummy()
        assert isinstance(
            dummy_model, ScaledMixture
        ), "Model must be a scaled mixture model."

        args_name = list(map(lambda x: x.name, params))
        if self._vt_params is None:
            raise ValueError("VT parameters must be provided.")
        assert all(
            param in args_name for param in self._vt_params
        ), f"Missing parameters: {self._vt_params} in {args_name}"
        self.vt_params_index: List[Int] = list(
            map(lambda x: args_name.index(x), self._vt_params)
        )
        self.vt_params_unif_rvs = JointDistribution(
            *map(lambda x: DEFAULT_PRIORS[x], args_name)
        )

        self.variables, duplicates, self.model = model.get_dist()
        self.variables_index = {key: i for i, key in enumerate(self.variables.keys())}

        for key, value in duplicates.items():
            self.variables_index[key] = self.variables_index[value]

        self.ref_priors = JointDistribution(*map(lambda x: x.prior, params))
        self.priors = JointDistribution(*self.variables.values())

    def exp_rate_integral_uniform_samples(
        self, N: int, key: PRNGKeyArray, model: Distribution
    ) -> Array:
        r"""This method approximates the Monte-Carlo integral by sampling from the
        uniform distribution.

        :param N: Number of samples.
        :param key: PRNG key.
        :param model: :math:`\rho(\lambda\mid\Lambda)`.
        :return: Integral.
        """
        samples = self.vt_params_unif_rvs.sample(key, (N,))
        volume = jtr.reduce(
            lambda x, y: x * (y.high - y.low),
            self.vt_params_unif_rvs.marginal_distributions,
            1.0,
            is_leaf=lambda x: isinstance(x, Uniform),
        )
        logpdf = model.log_prob(samples) + self.logVT(
            samples[..., self.vt_params_index]
        )
        return volume * jnp.mean(jnp.exp(logpdf))

    def exp_rate_integral_model_samples(
        self, N: int, key: PRNGKeyArray, model: Distribution
    ) -> Array:
        r"""This method approximates the Monte-Carlo integral by sampling from the
        model.

        :param N: Number of samples.
        :param key: PRNG key.
        :param model: :math:`\rho(\lambda\mid\Lambda)`.
        :return: Integral.
        """
        samples = model.sample(key, (N,))[..., self.vt_params_index]
        return jnp.mean(jnp.exp(self.logVT(samples)))

    def exp_rate_integral(self, model: Distribution) -> Array:
        r"""This function calculates the integral inside the term
        :math:`\exp(\Lambda)` in the likelihood function. The integral is given by,

        .. math::
            \mu(\Lambda) =
            \int \mathrm{VT}(\lambda)\rho(\lambda\mid\Lambda) \mathrm{d}\lambda

        :param model: Distribution.
        :return: Integral.
        """
        N = 1 << 13
        key = jrd.PRNGKey(np.random.randint(1, 2**32 - 1))
        if self.vt_method == "uniform":
            vt_value = self.exp_rate_integral_uniform_samples(N, key, model)
        elif self.vt_method == "model":
            vt_value = self.exp_rate_integral_model_samples(N, key, model)
        elif self.vt_method == "custom":
            vt_value = self.custom_vt(N, key, model)
        else:
            raise ValueError("Invalid VT method.")
        vt_value *= self.time * self.scale_factor
        debug_flush("exp_rate_integral: {vt_value}", vt_value=vt_value)
        return vt_value

    def log_likelihood(self, x: Array, data: dict) -> Array:
        """The log likelihood function for the inhomogeneous Poisson process.

        :param x: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        model: ScaledMixture = self.model(
            **{
                name: (
                    x[..., i]
                    if not name.startswith("log_rate")
                    else x[..., i] / jnp.log10(jnp.e)
                )
                for name, i in self.variables_index.items()
            }
        )

        log_likelihood = jtr.reduce(
            lambda x, y: x
            + logsumexp(model.log_prob(y) - self.ref_priors.log_prob(y), axis=-1)
            - jnp.log(y.shape[0]),
            data["data"],
            0.0,
        )

        debug_flush("model_log_likelihood: {mll}", mll=log_likelihood)

        expected_rates = jnp.dot(
            model._log_scales,
            jnp.asarray(
                [
                    self.exp_rate_integral(model_i)
                    for model_i in model._component_distributions
                ]
            ),
        )

        debug_flush("expected_rate={expr}", expr=expected_rates)

        return log_likelihood - expected_rates

    def log_posterior(self, x: Array, data: Optional[dict] = None) -> Array:
        r"""The likelihood function for the inhomogeneous Poisson process.

        .. math::
            \log p(\Lambda\mid\text{data}) \propto \log\pi(\Lambda) + \log\mathcal{L}(\Lambda)

        :param x: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        log_prior = self.priors.log_prob(x)
        debug_flush("log_prior: {lp}", lp=log_prior)
        return log_prior + self.log_likelihood(x, data)


poisson_likelihood = PoissonLikelihood()
