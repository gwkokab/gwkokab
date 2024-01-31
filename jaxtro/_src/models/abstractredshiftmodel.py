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

from jaxampler.rvs import Normal
from jaxtyping import Array

from .abstractmodel import AbstractModel


class AbstractRedShiftModel(AbstractModel):
    @staticmethod
    def add_error(x: Array, scale: float = 0.01, size: int = 10) -> Array:
        return Normal(loc=x, scale=scale).rvs(shape=(size,))
