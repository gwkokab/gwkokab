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


from typing_extensions import Tuple

from jax import numpy as jnp
from jaxtyping import Array


def beta_dist_concentrations_to_mean_variance(
    alpha: Array, beta: Array
) -> Tuple[Array, Array]:
    r"""Let :math:`\alpha` and :math:`\beta` be the shape parameters of a beta
    distribution. This function returns the mean and variance of the distribution.
    Then concentrations are given by:

    .. math::
        \mu = \frac{\alpha}{\alpha + \beta}\qquad
        \sigma^2 = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}

    :param alpha: The shape parameter :math:`\alpha`.
    :param beta: The shape parameter :math:`\beta`.
    :return: The mean :math:`\mu` and variance :math:`\sigma^2` of the beta distribution.
    """
    sum_of_concentrations = jnp.add(alpha, beta)
    product_of_concentrations = jnp.multiply(alpha, beta)

    mean = jnp.divide(alpha, sum_of_concentrations)

    variance = jnp.add(sum_of_concentrations, 1)
    variance = jnp.multiply(jnp.square(sum_of_concentrations), variance)
    variance = jnp.divide(product_of_concentrations, variance)

    return mean, variance


def beta_dist_mean_variance_to_concentrations(
    mean: Array, variance: Array
) -> Tuple[Array, Array]:
    r"""Let :math:`\mu` and :math:`\sigma^2` be the mean and variance of a beta
    distribution. This function returns the shape parameters :math:`\alpha` and
    :math:`\beta` of the distribution. Then concentrations are given by:

    .. math::
        \alpha = \mu \left(\frac{\mu(1 - \mu)}{\sigma^2} - 1\right)\qquad
        \beta = \alpha\left(\frac{1}{\mu}-1\right)

    :param mean: The mean :math:`\mu` of the beta distribution.
    :param variance: The variance :math:`\sigma^2` of the beta distribution.
    :return: The shape parameters :math:`\alpha` and :math:`\beta` of the beta distribution.
    """
    alpha = jnp.subtract(1.0, mean)
    alpha = jnp.multiply(mean, alpha)
    alpha = jnp.divide(alpha, variance)
    alpha = jnp.subtract(alpha, 1.0)
    alpha = jnp.multiply(mean, alpha)

    beta = jnp.reciprocal(mean)
    beta = jnp.subtract(beta, 1.0)
    beta = jnp.multiply(alpha, beta)

    return alpha, beta
