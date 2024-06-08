#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from dataclasses import dataclass
from enum import auto, Enum, unique
from typing_extensions import Callable, Optional

from jaxtyping import Array, Float, Int


@unique
class ModelMeta(Enum):
    NAME = auto()
    PARAMETERS = auto()
    OUTPUT = auto()
    SAVE_AS = auto()


@unique
class Parameter(Enum):
    PRIMARY_MASS_SOURCE = "m1_source"
    SECONDARY_MASS_SOURCE = "m2_source"

    MASS_RATIO = "q"
    CHIRP_MASS = "M_c"
    SYMMETRIC_MASS_RATIO = "eta"
    REDUCED_MASS = "M_r"

    ECCENTRICITY = "ecc"

    PRIMARY_ALIGNED_SPIN = "a1"
    SECONDARY_ALIGNED_SPIN = "a2"

    PRIMARY_SPIN_X = "a1x"
    PRIMARY_SPIN_Y = "a1y"
    PRIMARY_SPIN_Z = "a1z"
    SECONDARY_SPIN_X = "a2x"
    SECONDARY_SPIN_Y = "a2y"
    SECONDARY_SPIN_Z = "a2z"


@dataclass(repr=True)
class PopInfo:
    ROOT_DIR: str
    EVENT_FILENAME: str
    CONFIG_FILENAME: str
    RATE: Float
    TIME: Optional[Float] = None
    VT_FILE: Optional[str] = None
    VT_PARAMS: Optional[list[Parameter]] = None
    NUM_REALIZATIONS: Int = 5


@dataclass(repr=True)
class NoisePopInfo:
    FILENAME_REGEX: str
    OUTPUT_DIR: str
    HEADER: list[Parameter]
    SIZE: Int
    ERROR_FUNCS: list[tuple[Int, Callable[[Float, Int], Array]]]