#  Copyright 2023 The Jaxtro Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import sys

sys.path.append("../jaxtro")

from jaxtro.models import EccentricityModel


class TestEccentricityModel:
    def test_init(self):
        model = EccentricityModel(sigma_ecc=0.1)
        assert model._scale == 0.1
        assert model._name is None

    def test_rvs(self):
        model = EccentricityModel(sigma_ecc=0.1)
        rvs = model.samples(1000)
        assert rvs.shape == (1000,)
        assert rvs.min() >= 0
        assert rvs.max() <= 1
