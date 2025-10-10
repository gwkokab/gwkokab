# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
from numpyro import distributions as dist

from kokab.utils.common import _available_prior, get_posterior_data


def test_get_posterior_data():
    with pytest.raises(ValueError, match=r"No files found to read posterior data"):
        get_posterior_data([], ["mass_1_source", "mass_2_source"])


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
