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


from kokab.utils.regex import match_all


def test_match_all():
    strings = ["alpha_0", "beta_0", "beta_1", "gamma_10", "alpha_1", "delta_1"]
    pattern_dict_with_val = {
        "beta_[0-9]+": 0.1,
        "gamma_[0-9]+": 0.2,
        "delta_[0-9]+": {0.3, 0.4},
        "alpha_0": "beta_0",
        "alpha_1": "delta_1",
    }
    matches = match_all(strings, pattern_dict_with_val)
    assert matches["alpha_0"] == matches["beta_0"]
    assert matches["beta_1"] == 0.1
    assert matches["gamma_10"] == 0.2
    assert matches["alpha_1"] == "delta_1"
    assert matches["delta_1"] == pattern_dict_with_val["delta_[0-9]+"]
