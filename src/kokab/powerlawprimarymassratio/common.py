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


from __future__ import annotations

from jax import numpy as jnp, vmap
from jaxtyping import Array, Bool
from numpyro.distributions import TransformedDistribution

from gwkokab.models import PowerLawPrimaryMassRatio
from gwkokab.models.transformations import (
    PrimaryMassAndMassRatioToComponentMassesTransform,
)
from gwkokab.utils.transformations import m1_q_to_m2
from gwkokab.vts import load_model


def get_logVT(vt_path):
    _, logVT = load_model(vt_path)

    def m1q_logVT(x: Array) -> Array:
        m1 = x[..., 0]
        q = x[..., 1]
        m2 = m1_q_to_m2(m1=m1, q=q)
        m1m2 = jnp.column_stack([m1, m2])
        return vmap(logVT)(m1m2)

    return m1q_logVT


def constraint_m1q(x: Array) -> Bool:
    m1 = x[..., 0]
    q = x[..., 1]
    mask = m1 > 0.0
    mask &= q >= 0.0
    mask &= q <= 1.0
    return mask


def constraint_m1m2(x: Array) -> Bool:
    m1 = x[..., 0]
    m2 = x[..., 1]
    mask = m2 > 0.0
    mask &= m1 >= m2
    return mask


def TransformedPowerLawPrimaryMassRatio(
    alpha, beta, mmin, mmax
) -> TransformedDistribution:
    r"""Transformed Power Law Primary Mass Ratio model.

    :param alpha: Power law index
    :param beta: Power law index
    :param mmin: Minimum mass
    :param mmax: Maximum mass
    :return: Transformed distribution
    """
    return TransformedDistribution(
        base_distribution=PowerLawPrimaryMassRatio(
            alpha=alpha, beta=beta, mmin=mmin, mmax=mmax
        ),
        transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
    )
