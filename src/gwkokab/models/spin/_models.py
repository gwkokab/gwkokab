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


from typing import Optional

from jax import numpy as jnp
from jax.typing import ArrayLike
from numpyro.distributions import (
    Beta,
    CategoricalProbs,
    constraints,
    MixtureGeneral,
    MultivariateNormal,
    TransformedDistribution,
    TruncatedNormal,
    Uniform,
)
from numpyro.distributions.transforms import AffineTransform

from gwkokab.utils.math import beta_dist_mean_variance_to_concentrations


def GaussianSpinModel(
    mu_eff: ArrayLike,
    sigma_eff: ArrayLike,
    mu_p: ArrayLike,
    sigma_p: ArrayLike,
    rho: ArrayLike,
    *,
    validate_args: Optional[bool] = None,
) -> MultivariateNormal:
    r"""Bivariate normal distribution for the effective and precessing spins.
    See Eq. (D3) and (D4) in `Population Properties of Compact Objects from
    the Second LIGO-Virgo Gravitational-Wave Transient
    Catalog <https://arxiv.org/abs/2010.14533>`_.

    .. math::
        \left(\chi_{\text{eff}}, \chi_{p}\right) \sim \mathcal{N}\left(
            \begin{bmatrix}
                \mu_{\text{eff}} \\ \mu_{p}
            \end{bmatrix},
            \begin{bmatrix}
                \sigma_{\text{eff}}^2 & \rho \sigma_{\text{eff}} \sigma_{p} \\
                \rho \sigma_{\text{eff}} \sigma_{p} & \sigma_{p}^2
            \end{bmatrix}
        \right)

    where :math:`\chi_{\text{eff}}` is the effective spin and
    :math:`\chi_{\text{eff}}\in[-1,1]` and :math:`\chi_{p}` is the precessing spin and
    :math:`\chi_{p}\in[0,1]`.

    Parameters
    ----------

    mu_eff : ArrayLike
        mean of the effective spin
    sigma_eff : ArrayLike
        standard deviation of the effective spin
    mu_p : ArrayLike
        mean of the precessing spin
    sigma_p : ArrayLike
        standard deviation of the precessing spin
    rho : ArrayLike
        correlation coefficient between the effective and precessing
        spins

    Returns
    -------
    MultivariateNormal
        Multivariate normal distribution for the effective and precessing spins
    """
    return MultivariateNormal(
        loc=jnp.array([mu_eff, mu_p]),
        covariance_matrix=jnp.array(
            [
                [jnp.square(sigma_eff), rho * sigma_eff * sigma_p],
                [rho * sigma_eff * sigma_p, jnp.square(sigma_p)],
            ]
        ),
        validate_args=validate_args,
    )


def IndependentSpinOrientationGaussianIsotropic(
    zeta: ArrayLike,
    sigma1: ArrayLike,
    sigma2: ArrayLike,
    *,
    validate_args: Optional[bool] = None,
) -> MixtureGeneral:
    r"""A mixture model of spin orientations with isotropic and normally
    distributed components. See Eq. (4) of `Determining the population
    properties of spinning black holes <https://arxiv.org/abs/1704.08370>`_.

    .. math::
        p(z_1,z_2\mid\zeta,\sigma_1,\sigma_2) = \frac{1-\zeta}{4} +
        \zeta\mathbb{I}_{[-1,1]}(z_1)\mathbb{I}_{[-1,1]}(z_2)
        \mathcal{N}(z_1\mid 1,\sigma_1)\mathcal{N}(z_2\mid 1,\sigma_2)

    where :math:`\mathbb{I}(\cdot)` is the indicator function.

    Parameters
    ----------

    zeta : ArrayLike
        The mixing probability of the second component.
    sigma1 : ArrayLike
        The standard deviation of the first component.
    sigma2 : ArrayLike
        The standard deviation of the second component.

    Returns
    -------
    MixtureGeneral
        Mixture model of spin orientations.
    """
    mixing_probs = jnp.array([1 - zeta, zeta])
    component_0_dist = Uniform(low=-1, high=1, validate_args=validate_args)
    component_1_dist = TruncatedNormal(
        loc=1.0,
        scale=jnp.array([sigma1, sigma2]),
        low=-1,
        high=1,
        validate_args=validate_args,
    )
    return MixtureGeneral(
        mixing_distribution=CategoricalProbs(
            probs=mixing_probs, validate_args=validate_args
        ),
        component_distributions=[component_0_dist, component_1_dist],
        support=constraints.real,
        validate_args=validate_args,
    )


def BetaFromMeanVar(
    mean: ArrayLike,
    variance: ArrayLike,
    loc: ArrayLike = 0.0,
    scale: ArrayLike = 1.0,
    *,
    validate_args: Optional[bool] = None,
) -> TransformedDistribution:
    r"""Beta distribution parameterized by the expected value and variance.

    Parameters
    ----------

    mean : ArrayLike
        Expected value of the beta distribution.
    variance : ArrayLike
        Variance of the beta distribution.
    loc : ArrayLike
        lower bound of the beta distribution, defaults to 0.0
    scale : ArrayLike
        width of the beta distribution, defaults to 1.0

    Returns
    -------
    TransformedDistribution
        Transformed distribution of the beta distribution.
    """
    alpha, beta = beta_dist_mean_variance_to_concentrations(mean, variance, loc, scale)
    return TransformedDistribution(
        Beta(alpha, beta, validate_args=validate_args),
        transforms=AffineTransform(
            loc=loc, scale=scale, domain=constraints.interval(loc, loc + scale)
        ),
        validate_args=validate_args,
    )
