#  Copyright 2023 The GWKokab Authors
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

from abc import abstractmethod

from jaxtyping import Array

from ..typing import Numeric


class AbstractModel(object):
    @abstractmethod
    def samples(self, num_of_samples: int) -> Array:
        raise NotImplementedError

    @abstractmethod
    def add_error(self, x: Array, scale: float, size: int) -> Numeric:
        raise NotImplementedError
