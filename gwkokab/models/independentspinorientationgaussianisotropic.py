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


from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Float
from numpyro import distributions as dist

from .utils import JointDistribution


def IndependentSpinOrientationGaussianIsotropic(zeta: Float, sigma1: Float, sigma2: Float) -> dist.MixtureGeneral:
    r"""A mixture model of spin orientations with isotropic and normally
    distributed components. See Eq. (4) of [Determining the population
    properties of spinning black holes](https://arxiv.org/abs/1704.08370).

    $$
        p(z_1,z_2\mid\zeta,\sigma_1,\sigma_2) = \frac{1-\zeta}{4} +
        \zeta\mathbb{I}_{[-1,1]}(z_1)\mathbb{I}_{[-1,1]}(z_2)
        \mathcal{N}(z_1\mid 1,\sigma_1)\mathcal{N}(z_2\mid 1,\sigma_2)
    $$

    where $\mathbb{I}(\cdot)$ is the indicator function.

    :param zeta: The mixing probability of the second component.
    :param sigma1: The standard deviation of the first component.
    :param sigma2: The standard deviation of the second component.
    :return: Mixture model of spin orientations.
    """
    mixing_probs = jnp.array([1 - zeta, zeta])
    component_0_dist = JointDistribution(
        dist.Uniform(low=-1, high=1, validate_args=True),
        dist.Uniform(low=-1, high=1, validate_args=True),
    )
    component_1_dist = JointDistribution(
        dist.TruncatedNormal(
            loc=1.0,
            scale=sigma1,
            low=-1,
            high=1,
            validate_args=True,
        ),
        dist.TruncatedNormal(
            loc=1.0,
            scale=sigma2,
            low=-1,
            high=1,
            validate_args=True,
        ),
    )

    return dist.MixtureGeneral(
        mixing_distribution=dist.Categorical(probs=mixing_probs),
        component_distributions=[component_0_dist, component_1_dist],
        support=dist.constraints.real,
        validate_args=True,
    )
