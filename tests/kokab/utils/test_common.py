# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Dict

import pytest
from numpyro import distributions as dist

from kokab.utils.common import get_dist, get_posterior_data


def test_get_posterior_data():
    with pytest.raises(ValueError, match=r"No files found to read posterior data"):
        get_posterior_data([], ["mass_1_source", "mass_2_source"])


@pytest.mark.parametrize(
    "meta_dict",
    [
        {"dist": "Uniform", "low": -5, "high": 10},
        {"dist": "Normal", "loc": 0.1, "scale": 10},
        {"dist": "HalfNormal", "scale": 10},
        {"dist": "Uniform", "low": 0, "high": 0},
        {"dist": "Normal", "loc": 0, "scale": 0},
        pytest.param(
            {"dist": "InvalidDist", "param": 1}, marks=pytest.mark.xfail(strict=True)
        ),  # Invalid distribution
    ],
)
def test_get_dist(meta_dict: Dict) -> None:
    parsed_dist = get_dist(meta_dict.copy())
    assert isinstance(parsed_dist, dist.Distribution)
    dist_name: str = meta_dict.pop("dist")
    assert parsed_dist.__class__.__name__.lower() == dist_name.lower()
    for key, value in meta_dict.items():
        assert getattr(parsed_dist, key) == value
