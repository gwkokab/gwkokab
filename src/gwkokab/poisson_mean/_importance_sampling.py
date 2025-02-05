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


from collections.abc import Callable, Sequence
from typing import Union

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from numpyro import distributions as dist

from ..models.utils import JointDistribution, ScaledMixture
from ..parameters import Parameter
from ._abc import PoissonMeanABC


class ImportanceSamplingPoissonMean(PoissonMeanABC):
    r"""It is very cheap to generate random samples from the proposal distribution and
    evaluate the importance weights and reusing them for rest of the calculations. We
    assume a proposal distribution :math:`\rho_{\phi}` and calculate the importance
    weights as,

    .. math::
        w_i = \frac{\operatorname{VT(\omega_i)}}{\rho_{\phi}(\omega_i)},
        \qquad \forall 0<i\leq N, \omega_i \sim \rho_{\phi}.

    We save these weights and samples for the later use and estimate
    :math:`\mu_{\Omega\mid\Lambda}` as,

    .. math::
        \hat{\mu}_{\Omega\mid\Lambda} \approx \frac{1}{N} \sum_{i=1}^{N}
        w_i \cdot \rho_{\Omega\mid\Lambda}(\omega_i\mid\lambda).

    Importance sampling is a variance reduction technique and it is very useful when the
    proposal distribution is close to the target distribution. Choice of a good proposal
    distribution is a tedious task and it is not always possible to find a good proposal
    distribution. To overcome this problem, we have used a mixture of proposal
    distributions to sample from. Usually, we use a uniform distribution and a
    log-uniform distribution over an appropriate range of the parameters. Some times we
    also add a peak at the maxima and at the expectation of the proposal distribution to
    improve the performance of the importance sampling.
    """

    log_weights: Array = eqx.field(init=False)
    samples: Array = eqx.field(init=False)

    def __init__(
        self,
        logVT_fn: Callable[[Array], Array],
        parameters: Sequence[Parameter],
        key: PRNGKeyArray,
        num_samples: int,
        scale: Union[int, float, Array] = 1.0,
    ) -> None:
        r"""
        Parameters
        ----------
        logVT_fn : Callable[[Array], Array]
            Log of the Volume Time Sensitivity function.
        parameters : Sequence[Parameter]
            List of parameters.
        key : PRNGKeyArray
            PRNG key.
        num_samples : int
            Number of samples
        scale : Union[int, float, Array]
            scale factor, defaults to 1.0
        """
        self.scale = scale
        hyper_uniform = JointDistribution(
            *[param.prior for param in parameters], validate_args=True
        )

        hyper_log_uniform = JointDistribution(
            *[
                dist.LogUniform(
                    low=jnp.maximum(param.prior.low, 1e-6),
                    high=param.prior.high,
                    validate_args=True,
                )
                for param in parameters
            ],
            validate_args=True,
        )

        component_distributions = [hyper_uniform, hyper_log_uniform]

        proposal_dist = dist.MixtureGeneral(
            dist.Categorical(probs=jnp.array([1.0 - 0.05, 0.05]), validate_args=True),
            component_distributions,
            support=hyper_uniform.support,
            validate_args=True,
        )

        proposal_samples = proposal_dist.sample(key, (num_samples,))

        logVT = logVT_fn(proposal_samples)
        proposal_log_prob = proposal_dist.log_prob(proposal_samples)

        self.log_weights = logVT - proposal_log_prob + jnp.log(self.scale)
        self.samples = proposal_samples

    def __call__(self, model: ScaledMixture) -> Array:
        log_prob = model.log_prob(self.samples).reshape(self.log_weights.shape)

        probs = jnp.where(
            jnp.isneginf(log_prob), 0.0, jnp.exp(self.log_weights + log_prob)
        )

        return jnp.mean(probs, axis=-1)
