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


#
"""The population analysis and estimation of merger rates of compact binaries is one of
the important topics in gravitational wave astronomy.

The primary ingredient in these analyses is the population-averaged sensitive volume.
"""

from ._abc import VolumeTimeSensitivityInterface as VolumeTimeSensitivityInterface
from ._injvt import (
    RealInjectionVolumeTimeSensitivity as RealInjectionVolumeTimeSensitivity,
)
from ._neuralvt import NeuralNetVolumeTimeSensitivity as NeuralNetVolumeTimeSensitivity
from ._pdet import pdet_O3 as pdet_O3
from ._popmodelvt import (
    PopModelsCalibratedVolumeTimeSensitivity as PopModelsCalibratedVolumeTimeSensitivity,
    PopModelsVolumeTimeSensitivity as PopModelsVolumeTimeSensitivity,
)
from ._train import train_regressor as train_regressor
from ._utils import (
    load_model as load_model,
    make_model as make_model,
    mse_loss_fn as mse_loss_fn,
    predict as predict,
    read_data as read_data,
    save_model as save_model,
)
from ._vt import available as available
