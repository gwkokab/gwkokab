# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from gwkanal.utils.regex import match_all


def test_match_all():
    strings = ["alpha_0", "beta_0", "beta_1", "gamma_10", "alpha_1", "delta_1"]
    pattern_dict_with_val = {
        "beta_[0-9]+": 0.1,
        "gamma_[0-9]+": 0.2,
        "delta_[0-9]+": {1: 0.4},
        "alpha_0": "beta_0",
        "alpha_1": "delta_1",
    }
    matches = match_all(strings, pattern_dict_with_val)
    assert matches["alpha_0"] == matches["beta_0"]
    assert matches["beta_1"] == 0.1
    assert matches["gamma_10"] == 0.2
    assert matches["alpha_1"] == "delta_1"
    assert matches["delta_1"] == pattern_dict_with_val["delta_[0-9]+"]
