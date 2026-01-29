# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
from numpyro import distributions as dist

from gwkanal.utils.priors import _available_prior


@pytest.mark.parametrize(
    "dist_name",
    [
        "HalfNormal",
        "Normal",
        "Uniform",
        pytest.param(
            "InvalidDist", marks=pytest.mark.xfail(strict=True)
        ),  # Invalid distribution
    ],
)
def test_available_prior(dist_name: str) -> None:
    parsed_dist = _available_prior(dist_name)
    assert issubclass(parsed_dist, dist.Distribution)
