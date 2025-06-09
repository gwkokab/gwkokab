# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import TruncatedNormal

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
    comp_dist = JointDistribution(
        Wysocki2019MassModel(
            alpha_m=alpha_m, mmin=mmin, mmax=mmax, validate_args=validate_args
        ),
        TruncatedNormal(
            loc=loc, scale=scale, low=low, high=high, validate_args=validate_args
        ),
    )
    return ScaledMixture(
        log_scales=jnp.array([log_rate]),
        component_distributions=[comp_dist],
        support=comp_dist.support,
        validate_args=validate_args,
    )
