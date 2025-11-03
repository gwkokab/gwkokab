# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import pytest

from kokab.utils.common import get_posterior_data


def test_get_posterior_data():
    with pytest.raises(ValueError, match=r"No files found to read posterior data"):
        get_posterior_data([], ["mass_1_source", "mass_2_source"])
