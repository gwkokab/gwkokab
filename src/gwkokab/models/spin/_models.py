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


from jax import numpy as jnp
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
    mu_eff, sigma_eff, mu_p, sigma_p, rho, *, validate_args=None
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

    :param mu_eff: mean of the effective spin
    :param sigma_eff: standard deviation of the effective spin
    :param mu_p: mean of the precessing spin
    :param sigma_p: standard deviation of the precessing spin
    :param rho: correlation coefficient between the effective and precessing
        spins
    :return: Multivariate normal distribution for the effective and precessing
        spins
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
    zeta, sigma1, sigma2, *, validate_args=None
) -> MixtureGeneral:
    r"""A mixture model of spin orientations with isotropic and normally
    distributed components. See Eq. (4) of `Determining the population
    properties of spinning black holes <https://arxiv.org/abs/1704.08370>`_.

    .. math::
        p(z_1,z_2\mid\zeta,\sigma_1,\sigma_2) = \frac{1-\zeta}{4} +
        \zeta\mathbb{I}_{[-1,1]}(z_1)\mathbb{I}_{[-1,1]}(z_2)
        \mathcal{N}(z_1\mid 1,\sigma_1)\mathcal{N}(z_2\mid 1,\sigma_2)

    where :math:`\mathbb{I}(\cdot)` is the indicator function.

    :param zeta: The mixing probability of the second component.
    :param sigma1: The standard deviation of the first component.
    :param sigma2: The standard deviation of the second component.
    :return: Mixture model of spin orientations.
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
    mean, variance, loc=0.0, scale=1.0, *, validate_args=None
) -> TransformedDistribution:
    r"""Beta distribution parameterized by the expected value and variance.

    :param mean: Expected value of the beta distribution.
    :param variance: Variance of the beta distribution.
    :param loc: lower bound of the beta distribution, defaults to 0.0
    :param scale: width of the beta distribution, defaults to 1.0
    :param validate_args: Whether to enable validation of distribution, defaults to
        None
    :return: Transformed distribution of the beta distribution
    """
    alpha, beta = beta_dist_mean_variance_to_concentrations(mean, variance, loc, scale)
    return TransformedDistribution(
        Beta(alpha, beta, validate_args=validate_args),
        transforms=AffineTransform(
            loc=loc, scale=scale, domain=constraints.interval(loc, loc + scale)
        ),
        validate_args=validate_args,
    )
