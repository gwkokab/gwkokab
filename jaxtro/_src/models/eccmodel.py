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

from __future__ import annotations

from typing import Optional

from jaxampler.rvs import TruncNormal
from jaxampler.typing import Numeric
from jaxtyping import Array

from .abstracteccentricitymodel import AbstractEccentricityModel


class EccentricityModel(TruncNormal, AbstractEccentricityModel):
    def __init__(self, sigma_ecc: Numeric, name: Optional[str] = None) -> None:
        super().__init__(loc=0.0, scale=sigma_ecc, low=0.0, high=1.0, name=name)

    def samples(self, num_of_samples: int) -> Array:
        return super().rvs(shape=(num_of_samples,), key=None)

    def __repr__(self) -> str:
        string = f"EccentricityModel(sigma_ecc={self._scale}"
        if self._name is not None:
            string += f", name={self._name})"
        return string
