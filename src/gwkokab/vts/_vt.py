# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import List

from ._abc import VolumeTimeSensitivityInterface
from ._injvt import (
    RealInjectionVolumeTimeSensitivity as RealInjectionVolumeTimeSensitivity,
)
from ._neuralvt import NeuralNetVolumeTimeSensitivity
from ._pdet import pdet_O3
from ._popmodelvt import (
    PopModelsCalibratedVolumeTimeSensitivity,
    PopModelsVolumeTimeSensitivity,
)


def __getattr__(name):
    match name:
        case "NeuralNetVolumeTimeSensitivity":
            return NeuralNetVolumeTimeSensitivity
        case "PopModelsVolumeTimeSensitivity":
            return PopModelsVolumeTimeSensitivity
        case "PopModelsCalibratedVolumeTimeSensitivity":
            return PopModelsCalibratedVolumeTimeSensitivity
        case "RealInjectionVolumeTimeSensitivity":
            return RealInjectionVolumeTimeSensitivity
        case "pdet_O3":
            return pdet_O3
        case _:
            raise AttributeError(f"module {__name__} has no attribute {name}")


# Copyright (c) 2024 Colm Talbot
# SPDX-License-Identifier: MIT


class _Available:
    names: List[str] = [
        "NeuralNetVolumeTimeSensitivity",
        "PopModelsCalibratedVolumeTimeSensitivity",
        "PopModelsVolumeTimeSensitivity",
        "RealInjectionVolumeTimeSensitivity",
        "pdet_O3",
    ]

    def keys(self) -> List[str]:
        return self.names

    def __getitem__(self, key) -> VolumeTimeSensitivityInterface:
        return sys.modules[__name__].__getattr__(key)

    def __repr__(self) -> str:
        return repr(self.keys())


available = _Available()
"""Available volume-time sensitivity models."""
