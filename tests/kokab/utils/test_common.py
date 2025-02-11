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
    ],
)
def test_get_dist(meta_dict: Dict) -> None:
    parsed_dist = get_dist(meta_dict.copy())
    assert isinstance(parsed_dist, dist.Distribution)
    dist_name: str = meta_dict.pop("dist")
    assert parsed_dist.__class__.__name__.lower() == dist_name.lower()
    for key, value in meta_dict.items():
        assert getattr(parsed_dist, key) == value
