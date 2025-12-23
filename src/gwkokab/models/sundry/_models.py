# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import numpy as jnp
from jaxtyping import ArrayLike
from numpyro.distributions import (
    CategoricalProbs,
    constraints,
    Independent,
    MixtureGeneral,
    TruncatedNormal,
    Uniform,
)

from ..constraints import any_constraint


def TwoTruncatedNormalMixture(
    comp1_high: Optional[ArrayLike],
    comp1_loc: ArrayLike,
    comp1_low: Optional[ArrayLike],
    comp1_scale: ArrayLike,
    comp2_high: Optional[ArrayLike],
    comp2_loc: ArrayLike,
    comp2_low: Optional[ArrayLike],
    comp2_scale: ArrayLike,
    zeta: ArrayLike,
    *,
    validate_args: Optional[bool] = None,
) -> MixtureGeneral:
    """Create a mixture model of two truncated normal distributions.

    Parameters
    ----------
    comp1_high : Optional[ArrayLike]
        Upper truncation for the first component.
    comp1_loc : ArrayLike
        Location parameter for the first component.
    comp1_low : Optional[ArrayLike]
        Lower truncation for the first component.
    comp1_scale : ArrayLike
        Scale parameter for the first component.
    comp2_high : Optional[ArrayLike]
        Upper truncation for the second component.
    comp2_loc : ArrayLike
        Location parameter for the second component.
    comp2_low : Optional[ArrayLike]
        Lower truncation for the second component.
    comp2_scale : ArrayLike
        Scale parameter for the second component.
    zeta : ArrayLike
        Mixing proportion for the second component.
    validate_args : Optional[bool], optional
        Whether to validate the arguments, by default None

    Returns
    -------
    MixtureGeneral
        A mixture of two truncated normal distributions.
    """
    mixing_distribution = CategoricalProbs(
        probs=jnp.stack((1.0 - zeta, zeta), axis=-1), validate_args=validate_args
    )
    trun_norm1 = TruncatedNormal(
        low=comp1_low,
        high=comp1_high,
        loc=comp1_loc,
        scale=comp1_scale,
        validate_args=validate_args,
    )
    trun_norm2 = TruncatedNormal(
        low=comp2_low,
        high=comp2_high,
        loc=comp2_loc,
        scale=comp2_scale,
        validate_args=validate_args,
    )
    component_distributions = [trun_norm1, trun_norm2]
    return MixtureGeneral(
        mixing_distribution=mixing_distribution,
        component_distributions=component_distributions,
        support=any_constraint((trun_norm1.support, trun_norm2.support)),
        validate_args=validate_args,
    )


def NDIsotropicAndTruncatedNormalMixture(
    zeta: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    isotropic_low: ArrayLike,
    isotropic_high: ArrayLike,
    gaussian_low: Optional[ArrayLike],
    gaussian_high: Optional[ArrayLike],
    *,
    batch_dim: int = 1,
    validate_args: Optional[bool] = None,
) -> MixtureGeneral:
    r"""General N-dimensional mixture model of an isotropic uniform distribution and a
    truncated normal distribution.

    .. math::
        p(\mathbf{x}\mid\zeta,\boldsymbol{\mu},\boldsymbol{\sigma}) =
        (1-\zeta)\mathcal{U}(\mathbf{x}\mid
        \boldsymbol{a},\boldsymbol{b}) +
        \zeta\mathcal{N}_{[\boldsymbol{L}, \boldsymbol{U}]}(\mathbf{x}\mid
        \boldsymbol{\mu},\boldsymbol{\sigma})

    where :math:`\mathcal{U}(\cdot)` is the isotropic uniform distribution between
    :math:`\boldsymbol{a}=\left< a_1, a_2, \ldots, a_N \right>` and
    :math:`\boldsymbol{b}=\left< b_1, b_2, \ldots, b_N \right>`, and
    :math:`\mathcal{N}_{[\boldsymbol{L}, \boldsymbol{U}]}(\cdot)` is the truncated normal
    distribution with mean :math:`\boldsymbol{\mu}=\left< \mu_1, \mu_2, \ldots,
    \mu_N \right>`, standard deviation
    :math:`\boldsymbol{\sigma}=\left< \sigma_1, \sigma_2, \ldots, \sigma_N \right>`,
    lower bound :math:`\boldsymbol{L}=\left< L_1, L_2, \ldots, L_N \right>`, and upper bound
    :math:`\boldsymbol{U}=\left< U_1, U_2, \ldots, U_N \right>`.


    Parameters
    ----------
    zeta : ArrayLike
        The mixing probability of the second component.
    loc : ArrayLike
        The mean of the truncated normal distribution.
    scale : ArrayLike
        The standard deviation of the truncated normal distribution.
    isotropic_low : ArrayLike
        The lower bound of the isotropic uniform distribution.
    isotropic_high : ArrayLike
        The upper bound of the isotropic uniform distribution.
    gaussian_low : Optional[ArrayLike]
        The lower bound of the truncated normal distribution.
    gaussian_high : Optional[ArrayLike]
        The upper bound of the truncated normal distribution.
    batch_dim : int, optional
        The batch dimension of the distributions, by default 1
    validate_args : Optional[bool], optional
        Whether to validate the parameters of the distributions, by default None

    Returns
    -------
    MixtureGeneral
        N-dimensional mixture model of an isotropic uniform distribution and a truncated
        normal distribution.
    """
    mixing_probs = jnp.stack((1.0 - zeta, zeta), axis=-1)
    isotropic_component = Independent(
        Uniform(
            low=isotropic_low,
            high=isotropic_high,
            validate_args=validate_args,
        ),
        batch_dim,
        validate_args=validate_args,
    )
    gaussian_component = Independent(
        TruncatedNormal(
            loc=loc,
            scale=scale,
            low=gaussian_low,
            high=gaussian_high,
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
        support=any_constraint(
            (
                constraints.independent(
                    constraints.interval(isotropic_low, isotropic_high), batch_dim
                ),
                constraints.independent(
                    constraints.interval(gaussian_low, gaussian_high), batch_dim
                ),
            )
        ),
        validate_args=validate_args,
    )
