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
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro import distributions as dist

from ..models.utils import JointDistribution, ScaledMixture
from ..parameters import Parameter
from ._abc import PoissonMeanABC


class ImportanceSamplingPoissonMean(PoissonMeanABC):
    log_weights: Array = eqx.field(init=False)
    samples: Array = eqx.field(init=False)

    def __init__(
        self,
        logVT_fn: Callable[[ScaledMixture], Array],
        parameters: Sequence[Parameter],
        key: PRNGKeyArray,
        num_samples: int,
        scale: Union[int, float, Array] = 1.0,
        add_peak: bool = False,
    ) -> None:
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

        uniform_key, proposal_key = jrd.split(key)
        component_distributions = [hyper_uniform, hyper_log_uniform]
        if add_peak:
            uniform_samples = hyper_uniform.sample(uniform_key, (num_samples,))

            logVT_val = logVT_fn(uniform_samples)

            VT_max_at = jnp.argmax(logVT_val)
            loc_vector_at_highest_density = uniform_samples[VT_max_at]

            loc_vector_by_expectation = jnp.average(
                uniform_samples, axis=0, weights=jnp.exp(logVT_val)
            )
            covariance_matrix = jnp.cov(uniform_samples.T)
            component_distributions.append(
                JointDistribution(
                    *[
                        dist.TruncatedNormal(
                            loc_vector_by_expectation[i],
                            jnp.sqrt(covariance_matrix[i, i]),
                            low=param.prior.low,
                            high=param.prior.high,
                            validate_args=True,
                        )
                        for i, param in enumerate(parameters)
                    ],
                    validate_args=True,
                )
            )
            component_distributions.append(
                JointDistribution(
                    *[
                        dist.TruncatedNormal(
                            loc_vector_at_highest_density[i],
                            jnp.sqrt(covariance_matrix[i, i]),
                            low=param.prior.low,
                            high=param.prior.high,
                            validate_args=True,
                        )
                        for i, param in enumerate(parameters)
                    ],
                    validate_args=True,
                )
            )

        n = len(component_distributions)

        proposal_dist = dist.MixtureGeneral(
            dist.Categorical(probs=jnp.ones(n) / n, validate_args=True),
            component_distributions,
            support=hyper_uniform.support,
            validate_args=True,
        )

        proposal_samples = proposal_dist.sample(proposal_key, (num_samples,))

        mask = parameters[0].prior.support(proposal_samples[..., 0])
        for i in range(1, len(parameters)):
            mask &= parameters[i].prior.support(proposal_samples[..., i])

        proposal_samples = proposal_samples[mask]

        self.log_weights = (
            logVT_fn(proposal_samples)
            - proposal_dist.log_prob(proposal_samples)
            + jnp.log(self.scale)
        )
        self.samples = proposal_samples

    def __call__(self, model: ScaledMixture) -> Array:
        return jnp.mean(
            jnp.exp(self.log_weights + model.log_prob(self.samples)), axis=-1
        )
