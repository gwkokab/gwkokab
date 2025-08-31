# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


#
"""The population analysis and estimation of merger rates of compact binaries is one of
the important topics in gravitational wave astronomy.

The primary ingredient in these analyses is the population-averaged sensitive volume.
"""

from ._abc import VolumeTimeSensitivityInterface as VolumeTimeSensitivityInterface
from ._neuralpdet import (
    NeuralNetProbabilityOfDetection as NeuralNetProbabilityOfDetection,
)
from ._neuralvt import NeuralNetVolumeTimeSensitivity as NeuralNetVolumeTimeSensitivity
from ._pdet import pdet_O3 as pdet_O3
from ._popmodelvt import (
    PopModelsCalibratedVolumeTimeSensitivity as PopModelsCalibratedVolumeTimeSensitivity,
    PopModelsVolumeTimeSensitivity as PopModelsVolumeTimeSensitivity,
)
from ._semianalyticalinjvt import (
    SemiAnalyticalRealInjectionVolumeTimeSensitivity as SemiAnalyticalRealInjectionVolumeTimeSensitivity,
)
from ._syninjvt import (
    SyntheticInjectionVolumeTimeSensitivity as SyntheticInjectionVolumeTimeSensitivity,
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
