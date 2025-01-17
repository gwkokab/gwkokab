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
from numpyro.distributions import TransformedDistribution

from gwkokab.models import SmoothedPowerlawAndPeak
from gwkokab.models.transformations import (
    PrimaryMassAndMassRatioToComponentMassesTransform,
)


def create_smoothed_powerlaw_and_peak_raw(
    alpha: ArrayLike,
    beta: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    mmin: ArrayLike,
    mmax: ArrayLike,
    low: ArrayLike,
    high: ArrayLike,
    delta: ArrayLike,
    lambda_peak: ArrayLike,
    log_rate: ArrayLike,
) -> SmoothedPowerlawAndPeak:
    """Create a smoothed powerlaw and peak model with raw parameters."""
    return SmoothedPowerlawAndPeak(
        alpha=alpha,
        beta=beta,
        loc=loc,
        scale=scale,
        mmin=mmin,
        mmax=mmax,
        low=low,
        high=high,
        delta=delta,
        lambda_peak=lambda_peak,
        log_rate=log_rate,
    )


def create_smoothed_powerlaw_and_peak(
    alpha: ArrayLike,
    beta: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
    mmin: ArrayLike,
    mmax: ArrayLike,
    low: ArrayLike,
    high: ArrayLike,
    delta: ArrayLike,
    lambda_peak: ArrayLike,
    log_rate: ArrayLike,
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
            low=low,
            high=high,
            delta=delta,
            lambda_peak=lambda_peak,
            log_rate=log_rate,
        ),
        transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
    )
