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


from functools import reduce
from typing_extensions import Callable, Optional, Self

import jax
import numpy as np
from jax import lax, numpy as jnp, random as jrd, tree as jtr
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float

from ..models.utils.jointdistribution import JointDistribution
from .utils import ModelPack


@register_pytree_node_class
class BayesianHierarchicalModel:
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
        *models: ModelPack,
        vt_params: list[str],
        logVT: Callable,
        time: Float,
    ) -> None:
        self.logVT = logVT
        self.time = time
        parameters_to_recover = reduce(
            lambda x, y: x + y.parameters_to_recover, models, []
        )
        self.parameters_to_recover_name = list(
            map(lambda x: x.name, parameters_to_recover)
        )
        self.population_priors = JointDistribution(
            *map(lambda x: x.prior, parameters_to_recover)
        )
        self.reference_prior = JointDistribution(
            *map(
                lambda x: x.prior,
                reduce(
                    lambda x, y: x + y.output,
                    filter(lambda x: x.name is not None, models),
                    [],
                ),
            )
        )
        self.arguments = list(
            map(
                lambda x: x.arguments,
                filter(lambda x: x.name is not None, models),
            )
        )
        self.names = list(
            filter(lambda x: x is not None, map(lambda x: x.name, models))
        )
        k = 0
        indexes = []
        outputs = []
        for model in models:
            if model.name is None:
                continue
            outputs.append(list(map(lambda x: x.name, model.output)))
            k_ = len(model.parameters_to_recover)
            indexes.append(list(range(k, k + k_)))
            k += k_

        vt_model_index = []
        vt_joint_dist_params = []
        for j, out in enumerate(outputs):
            for vt_param in vt_params:
                if vt_param in out and vt_param not in vt_joint_dist_params:
                    vt_joint_dist_params.extend(out)
                    vt_model_index.append(j)
                    break

        self.vt_mask = [
            vt_joint_dist_params.index(_vt_param) for _vt_param in vt_params
        ]
        self.vt_model_index = vt_model_index
        self.indexes = indexes

    def tree_flatten(self):
        children = ()
        aux_data = {
            "logVT": self.logVT,
            "time": self.time,
            "names": self.names,
            "parameters_to_recover_name": self.parameters_to_recover_name,
            "arguments": self.arguments,
            "population_priors": self.population_priors,
            "reference_prior": self.reference_prior,
            "indexes": self.indexes,
            "vt_model_index": self.vt_model_index,
            "vt_mask": self.vt_mask,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        obj = cls.__new__(cls)
        obj.logVT = aux_data["logVT"]
        obj.time = aux_data["time"]
        obj.names = aux_data["names"]
        obj.parameters_to_recover_name = aux_data["parameters_to_recover_name"]
        obj.arguments = aux_data["arguments"]
        obj.population_priors = aux_data["population_priors"]
        obj.reference_prior = aux_data["reference_prior"]
        obj.indexes = aux_data["indexes"]
        obj.vt_model_index = aux_data["vt_model_index"]
        obj.vt_mask = aux_data["vt_mask"]
        return obj

    def exp_rate_integral(self, x: Array) -> Array:
        r"""This function calculates the integral inside the term
        $\exp(\Lambda)$ in the likelihood function. The integral is given by,

        $$
            \mu(\Lambda) =
            \int \mathrm{VT}(\lambda)\rho(\lambda\mid\Lambda) \mathrm{d}\lambda
        $$

        :param rparams: Parameters for the model.
        :return: Integral.
        """
        vt_models = JointDistribution(
            *jtr.map(
                lambda j: self.names[j](
                    **self.arguments[j],
                    **{
                        self.parameters_to_recover_name[i]: x[..., i]
                        for i in self.indexes[j]
                    },
                ),
                self.vt_model_index,
            )
        )
        N = 1 << 13
        samples = vt_models.sample(jrd.PRNGKey(np.random.randint(1, 2**32 - 1)), (N,))[
            ..., self.vt_mask
        ]
        return jnp.mean(jnp.exp(self.logVT(samples)))

    def log_likelihood(self, x: Array, data: Optional[dict] = None) -> Array:
        """The log likelihood function for the inhomogeneous Poisson process.

        :param x: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        joint_model = JointDistribution(
            *jtr.map(
                lambda name, index, argument: name(
                    **argument,
                    **{self.parameters_to_recover_name[i]: x[..., i] for i in index},
                ),
                self.names,
                self.indexes,
                self.arguments,
            )
        )

        log_likelihood = jtr.reduce(
            lambda x, y: x
            + jax.nn.logsumexp(
                joint_model.log_prob(y) - self.reference_prior.log_prob(y)
            )
            - jnp.log(y.shape[0]),
            data["data"],
            0.0,
        )

        log_rate = x[..., -1]

        log_likelihood += data["N"] * log_rate

        rate = jnp.exp(log_rate)

        return log_likelihood - rate * self.time * lax.cond(
            jnp.isinf(log_likelihood),
            lambda x_: 0.0,
            lambda x_: self.exp_rate_integral(x_),
            x,
        )

    def log_posterior(self, x: Array, data: Optional[dict] = None) -> Array:
        r"""The likelihood function for the inhomogeneous Poisson process.

        $$p(\Lambda\mid\text{data})$$

        :param x: Recovered parameters.
        :param data: Data provided by the user/sampler.
        :return: Log likelihood value for the given parameters.
        """
        log_prior = self.population_priors.log_prob(x)
        return log_prior + lax.cond(
            jnp.isinf(log_prior),
            lambda x_, d_: 0.0,
            lambda x_, d_: self.log_likelihood(x_, d_),
            x,
            data,
        )
