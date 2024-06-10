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


from typing_extensions import Optional, Self

import jax
import numpy as np
from jax import numpy as jnp, random as jrd, tree as jtr
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Int
from numpyro import distributions as dist

from ..models.utils.jointdistribution import JointDistribution
from ..vts.neuralvt import load_model


@register_pytree_node_class
class LogInhomogeneousPoissonProcessLikelihood:
    r"""This class is used to provide a likelihood function for the
    inhomogeneous Poisson process. The likelihood is given by,

    $$
        \log\mathcal{L}(\Lambda) \propto -\mu(\Lambda)
        +\log\sum_{n=1}^N \int \ell_n(\lambda) \rho(\lambda\mid\Lambda)
        \mathrm{d}\lambda
    $$

    where, $\displaystyle\rho(\lambda\mid\Lambda) =
    \frac{\mathrm{d}N}{\mathrm{d}V\mathrm{d}t \mathrm{d}\lambda}$ is the merger
    rate density for a population parameterized by $\Lambda$, $\mu(\Lambda)$ is
    the expected number of detected mergers for that population, and
    $\ell_n(\lambda)$ is the likelihood for the $n$th observed event's
    parameters. Using Bayes' theorem, we can obtain the posterior
    $p(\Lambda\mid\text{data})$ by multiplying the likelihood by a prior
    $\pi(\Lambda)$.

    $$
        p(\Lambda\mid\text{data}) \propto \pi(\Lambda) \mathcal{L}(\Lambda)
    $$

    The integral inside the main likelihood expression is then evaluated via
    Monte Carlo as

    $$
        \int \ell_n(\lambda) \rho(\lambda\mid\Lambda) \mathrm{d}\lambda \propto
        \int \frac{p(\lambda | \mathrm{data}_n)}{\pi_n(\lambda)}
        \rho(\lambda\mid\Lambda) \mathrm{d}\lambda \approx
        \frac{1}{N_{\mathrm{samples}}}
        \sum_{i=1}^{N_{\mathrm{samples}}}
        \frac{\rho(\lambda_{n,i}\mid\Lambda)}{\pi_{n,i}}
    $$
    """

    def __init__(
        self: Self,
        *model_config: dict,
        independent_recovering_params: Optional[dict] = None,
        neural_vt_path: Optional[str] = None,
    ) -> None:
        self.free_recovering_params = independent_recovering_params

        self.subroutine(model_config)
        if neural_vt_path is not None:
            _, self.logVT = load_model(neural_vt_path)
            self.logVT = jax.vmap(self.logVT)

    def subroutine(self: Self, model_configs: dict):
        k = 0
        self.recovering_params = []
        self.fixed_params = []
        self.input_keys = []
        self.models = []
        self.labels = {}
        self.vt_params_available = {}
        for i, model in enumerate(model_configs):
            model["id"] = i
            if model.get("for_rate", False) is True:
                for vt_param in model["vt_params"]:
                    self.vt_params_available[vt_param] = model["id"]
            if model.get("fparams") is None:
                model["fparams"] = {}
            for rparam in model["rparams"]:
                model["rparams"][rparam]["id"] = k
                k += 1
            self.recovering_params.append(
                {
                    rparam: model["rparams"][rparam]["id"]
                    for rparam in model["rparams"].keys()
                }
            )
            self.labels.update(
                {
                    model["rparams"][rparam]["id"]: model["rparams"][rparam][
                        "label"
                    ]
                    for rparam in model["rparams"].keys()
                }
            )
            self.fixed_params.append(model["fparams"])
            self.input_keys.extend(model["input_keys"])
            self.models.append(model["name"])
        if self.free_recovering_params is not None:
            for rparam in self.free_recovering_params:
                self.free_recovering_params[rparam]["id"] = k
                k += 1
            self.labels.update(
                {
                    self.free_recovering_params[rparam][
                        "id"
                    ]: self.free_recovering_params[rparam]["label"]
                    for rparam in self.free_recovering_params.keys()
                }
            )
        self.n_dim = k
        self.priors: list[dist.Distribution] = [None] * self.n_dim
        for i, model in enumerate(model_configs):
            for rparam in model["rparams"]:
                self.priors[model["rparams"][rparam]["id"]] = model["rparams"][
                    rparam
                ]["prior"]
        if self.free_recovering_params is not None:
            for rparam in self.free_recovering_params:
                self.priors[self.free_recovering_params[rparam]["id"]] = (
                    self.free_recovering_params[rparam]["prior"]
                )

    def tree_flatten(self):
        children = ()
        aux_data = {
            "models": self.models,
            "n_dim": self.n_dim,
            "priors": self.priors,
            "input_keys": self.input_keys,
            "labels": self.labels,
            "frparams": self.free_recovering_params,
            "rparams": self.recovering_params,
            "fparams": self.fixed_params,
            "vt_params_available": self.vt_params_available,
            "logVT": self.logVT,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data.values())

    def get_model(
        self: Self, model_id: Int, rparams: Array
    ) -> dist.Distribution:
        r"""Get the model for the given model_id and rparams.

        :param model_id: Model ID.
        :param rparams: Recovered parameters for the model.
        :return: Model.
        """
        model = self.models[model_id]
        fparams = self.fixed_params[model_id]
        rparam = jtr.map(
            lambda index: rparams[index], self.recovering_params[model_id]
        )
        return model(**fparams, **rparam)

    def sum_log_prior(self: Self, value: Array) -> Array:
        r"""Sum of log prior probabilities.

        :param value: Value for which the prior probabilities are to be
            calculated.
        :return: Sum of log prior probabilities.
        """
        log_prior = 0
        for i, prior in enumerate(self.priors):
            log_prior_i = prior.log_prob(value[i])
            if jnp.isfinite(log_prior_i) is False:
                return -jnp.inf
            log_prior += log_prior_i
        return log_prior

    def exp_rate(self: Self, recovered_params: Array) -> Array:
        r"""This function calculates the integral inside the term
        $\exp(\Lambda)$ in the likelihood function. The integral is given by,

        $$
            \mu(\Lambda) =
            \int \mathrm{VT}(\lambda)\rho(\lambda\mid\Lambda) \mathrm{d}\lambda
        $$

        :param rparams: Parameters for the model.
        :return: Integral.
        """
        N = 1 << 13
        model_ids = list(set(self.vt_params_available.values()))
        model = JointDistribution(
            *[
                self.get_model(model_id, recovered_params)
                for model_id in model_ids
            ]
        )
        samples = model.sample(
            jrd.PRNGKey(np.random.randint(1, 2**32 - 1)), (N,)
        )
        return jnp.mean(jnp.exp(self.logVT(samples)))

    def log_likelihood(self: Self, rparams: Array, data: Optional[dict] = None):
        """The log likelihood function for the inhomogeneous Poisson process.

        :param rparams: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        mapped_rparams = jtr.map(
            lambda index: rparams[index], self.recovering_params
        )
        mapped_models = jtr.map(
            lambda model, fparams, rparam: model(**fparams, **rparam),
            self.models,
            self.fixed_params,
            mapped_rparams,
        )

        joint_model = JointDistribution(*mapped_models)

        integral_individual = jtr.map(
            lambda y: jax.nn.logsumexp(joint_model.log_prob(y))
            - jnp.log(y.shape[0]),
            data["data"],
        )

        log_likelihood = jtr.reduce(lambda x, y: x + y, integral_individual)

        if jnp.isfinite(log_likelihood) is False:
            return -jnp.inf

        rate = rparams[self.free_recovering_params["rate"]["id"]]
        log_rate = jnp.log(rate)

        log_likelihood += data["N"] * log_rate

        log_likelihood -= rate * self.exp_rate(rparams)

        return log_likelihood

    def __call__(self: Self, rparams: Array, data: Optional[dict] = None):
        r"""The likelihood function for the inhomogeneous Poisson process.

        $$p(\Lambda\mid\text{data})$$

        :param rparams: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        log_prior = self.sum_log_prior(rparams)

        if jnp.isfinite(log_prior) is False:
            return -jnp.inf

        return log_prior + self.log_likelihood(rparams, data)
