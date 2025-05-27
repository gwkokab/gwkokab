# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import TruncatedNormal
from numpyro.distributions.constraints import real_vector

from gwkokab.models import Wysocki2019MassModel
from gwkokab.models.utils import JointDistribution, ScaledMixture


def EccentricityMattersModel(
    log_rate: Array,
    alpha_m: Array,
    mmin: Array,
    mmax: Array,
    loc: Array,
    scale: Array,
    low: Array,
    high: Array,
    *,
    validate_args: Optional[bool] = None,
) -> ScaledMixture:
    return ScaledMixture(
        log_scales=jnp.array([log_rate]),
        component_distributions=[
            JointDistribution(
                Wysocki2019MassModel(
                    alpha_m=alpha_m, mmin=mmin, mmax=mmax, validate_args=validate_args
                ),
                TruncatedNormal(
                    loc=loc,
                    scale=scale,
                    low=low,
                    high=high,
                    validate_args=validate_args,
                ),
            )
        ],
        support=real_vector,
        validate_args=validate_args,
    )


def constraint(x: Array) -> Array:
    m1 = x[..., 0]
    m2 = x[..., 1]
    ecc = x[..., 2]
    mask = m2 <= m1
    mask &= m2 > 0.0
    mask &= m1 > 0.0
    mask &= ecc >= 0.0
    mask &= ecc <= 1.0
    return mask
