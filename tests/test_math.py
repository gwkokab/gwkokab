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


import numpy as np
import pytest
from gwkokab.utils.math import (
    beta_dist_concentrations_to_mean_variance,
    beta_dist_mean_variance_to_concentrations,
)


@pytest.mark.parametrize(
    "alpha, beta", [(10, 20), (30, 40), (60, 50), (70, 80), (100, 90)]
)
def test_beta_dist1(alpha, beta):
    mean, variance = beta_dist_concentrations_to_mean_variance(alpha, beta)
    alpha_, beta_ = beta_dist_mean_variance_to_concentrations(mean, variance)
    assert np.allclose(alpha, alpha_)
    assert np.allclose(beta, beta_)


@pytest.mark.parametrize(
    "mean, var", [(0.1, 0.02), (0.2, 0.05), (0.3, 0.07), (0.4, 0.1), (0.5, 0.12)]
)
def test_beta_dist2(mean, var):
    alpha, beta = beta_dist_mean_variance_to_concentrations(mean, var)
    mean_, var_ = beta_dist_concentrations_to_mean_variance(alpha, beta)
    assert np.allclose(mean, mean_)
    assert np.allclose(var, var_)
