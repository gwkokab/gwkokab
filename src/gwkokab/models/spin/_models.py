# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import numpy as jnp
from jax.typing import ArrayLike
from numpyro.distributions import (
    Beta,
    CategoricalProbs,
    constraints,
    Independent,
    MixtureGeneral,
    MultivariateNormal,
    TruncatedNormal,
    Uniform,
)


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
    mixing_probs = jnp.stack([1.0 - zeta, zeta], axis=-1)
    low = -jnp.ones((2,))
    high = jnp.ones((2,))
    batch_dim = 1
    isotropic_component = Independent(
        Uniform(
            low=low,
            high=high,
            validate_args=validate_args,
        ),
        batch_dim,
        validate_args=validate_args,
    )
    gaussian_component = Independent(
        TruncatedNormal(
            loc=high,  # set to high because high=1
            scale=jnp.stack([scale1, scale2], axis=-1),
            low=low,
            high=high,
            validate_args=validate_args,
        ),
        batch_dim,
        validate_args=validate_args,
    )
    return MixtureGeneral(
        mixing_distribution=CategoricalProbs(
            probs=mixing_probs, validate_args=validate_args
        ),
        component_distributions=[isotropic_component, gaussian_component],
        support=constraints.independent(constraints.interval(-1.0, 1.0), batch_dim),
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
    mixing_probs = jnp.stack([1.0 - zeta, zeta], axis=-1)
    low = jnp.full((2,), minimum)
    high = jnp.ones((2,))
    batch_dim = 1
    isotropic_component = Independent(
        Uniform(
            low=low,
            high=high,
            validate_args=validate_args,
        ),
        batch_dim,
        validate_args=validate_args,
    )
    gaussian_component = Independent(
        TruncatedNormal(
            loc=loc,
            scale=jnp.stack([scale, scale], axis=-1),
            low=low,
            high=high,
            validate_args=validate_args,
        ),
        batch_dim,
        validate_args=validate_args,
    )
    return MixtureGeneral(
        mixing_distribution=CategoricalProbs(
            probs=mixing_probs, validate_args=validate_args
        ),
        component_distributions=[isotropic_component, gaussian_component],
        support=constraints.independent(constraints.interval(minimum, 1.0), batch_dim),
        validate_args=validate_args,
    )
