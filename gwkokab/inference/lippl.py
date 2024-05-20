#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from functools import partial
from typing_extensions import Optional, Self

import jax
from jax import jit, lax, numpy as jnp
from jaxtyping import Array
from numpyro import distributions as dist

from ..models.utils.jointdistribution import JointDistribution
from ..vts.neuralvt import load_model


class LogInhomogeneousPoissonProcessLikelihood:
    r"""This class is used to provide a likelihood function for the inhomogeneous
    Poisson process. The likelihood is given by,

    $$
        \log\mathcal{L}(\Lambda) \propto -\mu(\Lambda)
        +\log\sum_{n=1}^N \int \ell_n(\lambda) \rho(\lambda\mid\Lambda)
        \mathrm{d}\lambda
    $$

    where, $\displaystyle\rho(\lambda\mid\Lambda) = \frac{\mathrm{d}N}{\mathrm{d}V
    \mathrm{d}t \mathrm{d}\lambda}$ is the merger rate density for a population
    parameterized by $\Lambda$, $\mu(\Lambda)$ is the expected number of detected
    mergers for that population, and $\ell_n(\lambda)$ is the likelihood for the
    $n$th observed event's parameters. Using Bayes' theorem, we can obtain the
    posterior $p(\Lambda\mid\text{data})$ by multiplying the likelihood by a prior
    $\pi(\Lambda)$.

    $$
        p(\Lambda\mid\text{data}) \propto \pi(\Lambda) \mathcal{L}(\Lambda)
    $$

    The integral inside the main likelihood expression is then evaluated via Monte Carlo as

    $$
        \int \ell_n(\lambda) \rho(\lambda\mid\Lambda) \mathrm{d}\lambda \propto
        \int \frac{p(\lambda | \mathrm{data}_n)}{\pi_n(\lambda)} \rho(\lambda\mid\Lambda) \mathrm{d}\lambda \approx
        \frac{1}{N_{\mathrm{samples}}}
        \sum_{i=1}^{N_{\mathrm{samples}}} \frac{\rho(\lambda_{n,i}\mid\Lambda)}{\pi_{n,i}}
    $$
    """

    def __init__(
        self: Self,
        *model_config: dict,
        frparams: Optional[dict] = None,
        neural_vt_path: Optional[str] = None,
    ) -> None:
        self.frparams = frparams
        self.model_configs = model_config
        self.subroutine()
        if neural_vt_path is not None:
            _, self.logVT = load_model(neural_vt_path)

    def subroutine(self: Self):
        k = 0
        self.rparams = []
        self.fparams = []
        self.input_keys = []
        self.models = []
        self.labels = {}
        self.vt_params_available = {}
        for i, model in enumerate(self.model_configs):
            model["id"] = i
            if model.get("for_rate", False) is True:
                for vt_param in model["vt_params"]:
                    self.vt_params_available[vt_param] = model["id"]
            if model.get("fparams") is None:
                model["fparams"] = {}
            for rparam in model["rparams"]:
                model["rparams"][rparam]["id"] = k
                k += 1
            self.rparams.append({rparam: model["rparams"][rparam]["id"] for rparam in model["rparams"].keys()})
            self.labels.update(
                {
                    model["rparams"][rparam]["id"]: model["rparams"][rparam]["label"]
                    for rparam in model["rparams"].keys()
                }
            )
            self.fparams.append(model["fparams"])
            self.input_keys.extend(model["input_keys"])
            self.models.append(model["name"])
        if self.frparams is not None:
            for rparam in self.frparams:
                self.frparams[rparam]["id"] = k
                k += 1
            self.labels.update(
                {self.frparams[rparam]["id"]: self.frparams[rparam]["label"] for rparam in self.frparams.keys()}
            )
        self.n_dim = k
        self.priors: list[dist.Distribution] = [None] * self.n_dim
        for i, model in enumerate(self.model_configs):
            for rparam in model["rparams"]:
                self.priors[model["rparams"][rparam]["id"]] = model["rparams"][rparam]["prior"]
        if self.frparams is not None:
            for rparam in self.frparams:
                self.priors[self.frparams[rparam]["id"]] = self.frparams[rparam]["prior"]

    def get_model(self: Self, model_id: int, rparams: Array) -> dist.Distribution:
        r"""Get the model for the given model_id and rparams.

        :param model_id: Model ID.
        :param rparams: Recovered parameters for the model.
        :return: Model.
        """
        model = self.models[model_id]
        fparams = self.fparams[model_id]
        rparam = jax.tree.map(lambda index: rparams[index], self.rparams[model_id])
        return model(**fparams, **rparam)

    @partial(jit, static_argnums=(0,))
    def sum_log_prior(self: Self, value: Array) -> Array:
        r"""Sum of log prior probabilities.

        :param value: Value for which the prior probabilities are to be calculated.
        :return: Sum of log prior probabilities.
        """
        return jnp.sum(jnp.asarray([prior.log_prob(value[i]) for i, prior in enumerate(self.priors)]))

    @partial(jit, static_argnums=(0,))
    def exp_rate(self: Self, rparams: Array) -> Array:
        r"""This function calculates the integral inside the term $\exp(\Lambda)$ in the
        likelihood function. The integral is given by,

        $$
            \mu(\Lambda) = \int \mathrm{VT}(\lambda)\rho(\lambda\mid\Lambda) \mathrm{d}\lambda
        $$

        :param rparams: Parameters for the model.
        :return: Integral.
        """
        N = 1 << 13
        model_ids = list(set(self.vt_params_available.values()))
        model = JointDistribution(*[self.get_model(model_id, rparams) for model_id in model_ids])
        samples = model.sample(10000, (N,))
        return jnp.mean(jnp.exp(self.logVT(samples)))

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self: Self, rparams: Array, data: Optional[dict] = None):
        """The log likelihood function for the inhomogeneous Poisson process.

        :param rparams: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        mapped_rparams = jax.tree.map(lambda index: rparams[index], self.rparams)
        mapped_models = jax.tree.map(
            lambda model, fparams, rparam: model(**fparams, **rparam), self.models, self.fparams, mapped_rparams
        )

        joint_model = JointDistribution(*mapped_models)

        integral_individual = jax.tree.map(
            lambda y: jax.scipy.special.logsumexp(joint_model.log_prob(y)) - jnp.log(y.shape[0]),
            data["data"],
        )

        log_likelihood = jnp.sum(jnp.asarray(jax.tree.leaves(integral_individual)))
        log_rate = jnp.log(rparams[self.frparams["rate"]["id"]])
        log_likelihood += data["N"] * log_rate

        log_likelihood -= lax.cond(
            jnp.isfinite(log_likelihood),
            lambda r: self.exp_rate(r),  # true function
            lambda r: 0.0,  # false function
            rparams,
        )

        return log_likelihood

    @partial(jit, static_argnums=(0,))
    def log_posterior(self: Self, rparams: Array, data: Optional[dict] = None):
        r"""The likelihood function for the inhomogeneous Poisson process.

        $$p(\Lambda\mid\text{data})$$

        :param rparams: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        log_prior = self.sum_log_prior(rparams)
        return log_prior + lax.cond(
            jnp.isfinite(log_prior),
            lambda r, d: self.log_likelihood(r, d),  # true function
            lambda r, d: 0.0,  # false function
            rparams,
            data,
        )
