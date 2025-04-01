# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


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
