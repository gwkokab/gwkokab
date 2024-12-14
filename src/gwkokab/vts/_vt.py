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
from ._neuralvt import NeuralNetVolumeTimeSensitivity
from ._popmodelvt import (
    PopModelsCalibratedVolumeTimeSensitivity,
    PopModelsVolumeTimeSensitivity,
)


def __getattr__(name):
    if name == "NeuralNetVolumeTimeSensitivity":
        return NeuralNetVolumeTimeSensitivity
    elif name == "PopModelsVolumeTimeSensitivity":
        return PopModelsVolumeTimeSensitivity
    elif name == "PopModelsCalibratedVolumeTimeSensitivity":
        return PopModelsCalibratedVolumeTimeSensitivity
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


# Copyright (c) 2024 Colm Talbot
# SPDX-License-Identifier: MIT


class _Available:
    names: List[str] = [
        "NeuralNetVolumeTimeSensitivity",
        "PopModelsCalibratedVolumeTimeSensitivity",
        "PopModelsVolumeTimeSensitivity",
    ]

    def keys(self) -> List[str]:
        return self.names

    def __getitem__(self, key) -> VolumeTimeSensitivityInterface:
        return sys.modules[__name__].__getattr__(key)

    def __repr__(self) -> str:
        return repr(self.keys())


available = _Available()
