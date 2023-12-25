# Copyright 2023 The Jaxtro Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path.append("../jaxtro")

from jaxtro.models import Wysocki2019MassModel


def test_init() -> None:
    model = Wysocki2019MassModel(
        alpha=0.8,
        k=0,
        mmin=5.0,
        mmax=40.0,
        Mmax=80.0,
        name="test",
    )
    assert model._name == "test"
    assert model._alpha == 0.8
    assert model._k == 0
    assert model._mmin == 5.0
    assert model._mmax == 40.0
    assert model._Mmax == 80.0
