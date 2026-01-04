# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

import jax
from jax import numpy as jnp
from jax.scipy.stats import truncnorm
from jaxtyping import ArrayLike
from numpyro.distributions import (
    Beta,
    constraints,
    Distribution,
    MixtureGeneral,
    MultivariateNormal,
)
from numpyro.distributions.util import promote_shapes, validate_sample

from ..sundry import NDIsotropicAndTruncatedNormalMixture


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
    scale1: ArrayLike,
    scale2: ArrayLike,
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
    scale1 : ArrayLike
        The standard deviation of the first component.
    scale2 : ArrayLike
        The standard deviation of the second component.

    Returns
    -------
    MixtureGeneral
        Mixture model of spin orientations.
    """
    return NDIsotropicAndTruncatedNormalMixture(
        zeta=zeta,
        loc=1.0,
        scale=jnp.stack((scale1, scale2), axis=-1),
        isotropic_low=-1.0,
        isotropic_high=jnp.ones((2,)),
        gaussian_low=-1.0,
        gaussian_high=1.0,
        validate_args=validate_args,
    )


def BetaFromMeanVar(
    mean: ArrayLike,
    variance: ArrayLike,
    *,
    validate_args: Optional[bool] = None,
) -> Beta:
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
    Beta
        Beta distribution with the specified mean and variance.
    """
    alpha = (jnp.square(mean) * (1 - mean) - mean * variance) / variance
    beta = (mean * jnp.square(1 - mean) - (1 - mean) * variance) / variance
    return Beta(alpha, beta, validate_args=validate_args)


def MinimumTiltModel(
    zeta: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    minimum: ArrayLike = -1.0,
    *,
    validate_args: Optional[bool] = None,
) -> MixtureGeneral:
    r"""Minimum tilt model introduced in
    `GWTC-4.0: Population Properties of Merging Compact Binaries <https://arxiv.org/abs/2508.18083>`_.
    A mixture model of spin orientations with isotropic and normally
    distributed components, with a minimum tilt constraint.

    Parameters
    ----------
    zeta : ArrayLike
        Weight of the Gaussian component.
    loc : ArrayLike
        Location parameter of the Gaussian component.
    scale : ArrayLike
        Scale parameter of the Gaussian component.
    minimum : ArrayLike, optional
        Minimum cosine tilt angle, by default -1.0
    validate_args : Optional[bool], optional
        Whether to validate the arguments, by default None

    Returns
    -------
    MixtureGeneral
        Mixture model of spin orientations.
    """
    return NDIsotropicAndTruncatedNormalMixture(
        zeta=zeta,
        loc=loc,
        scale=scale,
        isotropic_low=minimum,
        isotropic_high=jnp.ones((2,)),
        gaussian_low=minimum,
        gaussian_high=jnp.ones((2,)),
        validate_args=validate_args,
    )


def MinimumTiltModelExtended(
    zeta: ArrayLike,
    loc1: ArrayLike,
    loc2: ArrayLike,
    scale1: ArrayLike,
    scale2: ArrayLike,
    minimum1: ArrayLike = -1.0,
    minimum2: ArrayLike = -1.0,
    *,
    validate_args: Optional[bool] = None,
) -> MixtureGeneral:
    """A mixture model of spin orientations with isotropic and normally distributed
    components, with a minimum tilt constraint for each spin.

    Parameters
    ----------
    zeta : ArrayLike
        Weight of the Gaussian component.
    loc1 : ArrayLike
        Location parameter of the first Gaussian component.
    loc2 : ArrayLike
        Location parameter of the second Gaussian component.
    scale1 : ArrayLike
        Scale parameter of the first Gaussian component.
    scale2 : ArrayLike
        Scale parameter of the second Gaussian component.
    minimum1 : ArrayLike, optional
        Minimum cosine tilt angle of the first component, by default -1.0
    minimum2 : ArrayLike, optional
        Minimum cosine tilt angle of the second component, by default -1.0
    validate_args : Optional[bool], optional
        Whether to validate the arguments, by default None

    Returns
    -------
    MixtureGeneral
        Mixture model of spin orientations with minimum tilt constraints for each spin.
    """
    return NDIsotropicAndTruncatedNormalMixture(
        zeta=zeta,
        loc=jnp.stack([loc1, loc2], axis=-1),
        scale=jnp.stack([scale1, scale2], axis=-1),
        isotropic_low=jnp.stack([minimum1, minimum2], axis=-1),
        isotropic_high=1.0,
        gaussian_low=jnp.stack([minimum1, minimum2], axis=-1),
        gaussian_high=1.0,
        validate_args=validate_args,
    )


class GWTC4EffectiveSpinSkewNormalModel(Distribution):
    r"""GWTC-4 effective spin skew normal model.

    This class implements effective spin skew normal model introduced in equation (B37)
    `GWTC-4.0: Population Properties of Merging Compact Binaries
    <https://arxiv.org/abs/2508.18083>`_.

    .. math::

        p(\chi_\mathrm{eff} | \mu, \sigma, \epsilon) \propto \begin{cases}
            (1 + \epsilon) \mathcal{N}_{[-1,1]}(\chi_\mathrm{eff} | \mu, \sigma (1 + \epsilon)), & \chi_\mathrm{eff} \leq 0 \\
            (1 - \epsilon) \mathcal{N}_{[-1,1]}(\chi_\mathrm{eff} | \mu, \sigma (1 - \epsilon)), & \chi_\mathrm{eff} > 0
        \end{cases}

    where :math:`\mathcal{N}_{[-1,1]}(x | \mu, \sigma)` is the truncated normal distribution
    with mean :math:`\mu` and standard deviation :math:`\sigma`, truncated to the interval
    :math:`[-1, 1]`.

    The normalization constant is expressed as:

    .. math::
        \mathcal{Z} = (1 + \epsilon) \left[ \frac{\Phi\left( -\displaystyle\frac{\mu}{\sigma (1 + \epsilon)} \right) - \Phi\left( \displaystyle\frac{-1 - \mu}{\sigma (1 + \epsilon)} \right)}{\Phi\left( \displaystyle\frac{1 - \mu}{\sigma (1 + \epsilon)} \right) - \Phi\left( \displaystyle\frac{-1 - \mu}{\sigma (1 + \epsilon)} \right)} \right]
            + (1 - \epsilon) \left[1 - \frac{\Phi\left( -\displaystyle\frac{\mu}{\sigma (1 - \epsilon)} \right) - \Phi\left( \displaystyle\frac{-1 - \mu}{\sigma (1 - \epsilon)} \right)}{\Phi\left( \displaystyle\frac{1 - \mu}{\sigma (1 - \epsilon)} \right) - \Phi\left( \displaystyle\frac{-1 - \mu}{\sigma (1 - \epsilon)} \right)} \right]

    where, :math:`\Phi(x)` is the cumulative distribution function of the standard normal distribution.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "epsilon": constraints.interval(-1.0, 1.0),
    }
    support = constraints.interval(-1.0, 1.0)
    pytree_data_fields = ("loc", "scale", "epsilon")

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        epsilon: ArrayLike,
        *,
        validate_args: Optional[bool] = None,
    ):
        self.loc, self.scale, self.epsilon = promote_shapes(loc, scale, epsilon)
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(self.loc), jnp.shape(self.scale), jnp.shape(self.epsilon)
        )
        super(GWTC4EffectiveSpinSkewNormalModel, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        scale1 = self.scale * (1.0 + self.epsilon)
        scale2 = self.scale * (1.0 - self.epsilon)
        normalization_constant = (1.0 + self.epsilon) * truncnorm.cdf(
            0.0,
            -(1.0 + self.loc) / scale1,
            (1.0 - self.loc) / scale1,
            loc=self.loc,
            scale=scale1,
        ) + (1.0 - self.epsilon) * truncnorm.sf(
            0.0,
            -(1.0 + self.loc) / scale2,
            (1.0 - self.loc) / scale2,
            loc=self.loc,
            scale=scale2,
        )

        factor = 1.0 + jnp.where(value <= 0.0, 1.0, -1.0) * self.epsilon
        scale = self.scale * factor

        log_pdf_unnorm = jnp.log(factor) + truncnorm.logpdf(
            value,
            -(1.0 + self.loc) / scale,
            (1.0 - self.loc) / scale,
            loc=self.loc,
            scale=scale,
        )

        log_pdf = log_pdf_unnorm - jnp.log(normalization_constant)

        return log_pdf
