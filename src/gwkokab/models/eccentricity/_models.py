# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import numpy as jnp
from jaxtyping import ArrayLike
from numpyro.distributions import CategoricalProbs, MixtureGeneral, TruncatedNormal

from ..constraints import any_constraint


def EccentricMixtureModel(
    high1: ArrayLike,
    high2: ArrayLike,
    loc1: ArrayLike,
    loc2: ArrayLike,
    low1: ArrayLike,
    low2: ArrayLike,
    scale1: ArrayLike,
    scale2: ArrayLike,
    zeta: ArrayLike,
    *,
    validate_args: Optional[bool] = None,
) -> MixtureGeneral:
    """Create a mixture model of two truncated normal distributions.

    Parameters
    ----------
    high1 : ArrayLike
        Upper truncation for the first component.
    high2 : ArrayLike
        Upper truncation for the second component.
    loc1 : ArrayLike
        Location parameter for the first component.
    loc2 : ArrayLike
        Location parameter for the second component.
    low1 : ArrayLike
        Lower truncation for the first component.
    low2 : ArrayLike
        Lower truncation for the second component.
    scale1 : ArrayLike
        Scale parameter for the first component.
    scale2 : ArrayLike
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
        low=low1,
        high=high1,
        loc=loc1,
        scale=scale1,
        validate_args=validate_args,
    )
    trun_norm2 = TruncatedNormal(
        low=low2,
        high=high2,
        loc=loc2,
        scale=scale2,
        validate_args=validate_args,
    )
    component_distributions = [trun_norm1, trun_norm2]
    return MixtureGeneral(
        mixing_distribution=mixing_distribution,
        component_distributions=component_distributions,
        support=any_constraint((trun_norm1.support, trun_norm2.support)),
        validate_args=validate_args,
    )
