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
from typing_extensions import Callable, Optional

from jaxtyping import Array, Float, Int


@dataclass(frozen=True)
class PopInfo:
    r"""Meta data class for the population.

    :param ROOT_DIR: The root directory of the population.
    :param EVENT_FILENAME: The name of the event file.
    :param CONFIG_FILENAME: The name of the configuration file.
    :param RATE: The rate of the population.
    :param TIME: The time of the population.
    :param LOG_VT: The log-vt function.
    :param VT_PARAMS: The parameters of the log-vt function.
    :param NUM_REALIZATIONS: The number of realizations.
    """

    ROOT_DIR: str
    EVENT_FILENAME: str
    CONFIG_FILENAME: str
    RATE: Float
    TIME: Optional[Float] = None
    LOG_VT: Optional[Callable[[Array], Array]] = None
    VT_PARAMS: Optional[list[str]] = None
    NUM_REALIZATIONS: Int = 5


@dataclass(frozen=True)
class NoisePopInfo:
    r"""Meta data class for the noise population.

    :param ROOT_DIR: The root directory of the population.
    :param EVENT_FILENAME: The name of the event file.
    :param CONFIG_FILENAME: The name of the configuration file.
    :param RATE: The rate of the population.
    """

    FILENAME_REGEX: str
    OUTPUT_DIR: str
    HEADER: list[str]
    SIZE: Int
    ERROR_FUNCS: dict[tuple[str, ...], tuple[Int, Callable[[Float, Int], Array]]]
