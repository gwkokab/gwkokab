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

from abc import abstractmethod
from collections.abc import Callable, Sequence

import equinox as eqx
from jaxtyping import Array


class VolumeTimeSensitivityInterface(eqx.Module):
    parameters: Sequence[str]
    filename: str
    shuffle_indices: Sequence[int] = eqx.field(init=False)

    @abstractmethod
    def get_logVT(self) -> Callable[[Array], Array]:
        """Gets the logVT function."""
        raise NotImplementedError

    @abstractmethod
    def get_vmapped_logVT(self) -> Callable[[Array], Array]:
        """Gets the vmapped logVT function for batch processing."""
        raise NotImplementedError
