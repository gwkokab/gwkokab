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


from jaxtyping import ArrayLike
from numpyro.distributions import Distribution, TransformedDistribution, TruncatedNormal

from gwkokab.models import SmoothedPowerlawAndPeak
from gwkokab.models.transformations import (
    PrimaryMassAndMassRatioToComponentMassesTransform,
)
from gwkokab.models.utils import JointDistribution


def create_smoothed_powerlaw_and_peak_raw(
    alpha: ArrayLike,
    beta: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    mmin: ArrayLike,
    mmax: ArrayLike,
    delta: ArrayLike,
    lambda_peak: ArrayLike,
) -> SmoothedPowerlawAndPeak:
    """Create a smoothed powerlaw and peak model with raw parameters."""
    return SmoothedPowerlawAndPeak(
        alpha=alpha,
        beta=beta,
        loc=loc,
        scale=scale,
        mmin=mmin,
        mmax=mmax,
        delta=delta,
        lambda_peak=lambda_peak,
    )


def create_smoothed_powerlaw_and_peak(
    alpha: ArrayLike,
    beta: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    mmin: ArrayLike,
    mmax: ArrayLike,
    delta: ArrayLike,
    lambda_peak: ArrayLike,
) -> TransformedDistribution:
    """Create a smoothed powerlaw and peak model with raw parameters."""
    return TransformedDistribution(
        base_distribution=create_smoothed_powerlaw_and_peak_raw(
            alpha=alpha,
            beta=beta,
            loc=loc,
            scale=scale,
            mmin=mmin,
            mmax=mmax,
            delta=delta,
            lambda_peak=lambda_peak,
        ),
        transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
    )


build_smoothing_powerlaw_and_peak = create_smoothed_powerlaw_and_peak


def create_model(
    use_spin: bool = False,
    **params,
) -> Distribution:
    powerlaw = build_smoothing_powerlaw_and_peak(
        params["alpha"],
        params["beta"],
        params["loc"],
        params["scale"],
        params["mmin"],
        params["mmax"],
        params["delta"],
        params["lambda_peak"],
    )
    if not use_spin:
        return powerlaw

    a1_dist = TruncatedNormal(
        params["chi1_loc"],
        params["chi1_scale"],
        0.0,
        1.0,
    )
    a2_dist = TruncatedNormal(
        params["chi2_loc"],
        params["chi2_scale"],
        0.0,
        1.0,
    )
    return JointDistribution(powerlaw, a1_dist, a2_dist)
