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


from collections.abc import Sequence
from typing import Optional

import equinox as eqx
import h5py
import jax
import numpy as np
from jaxtyping import Array

from gwkokab import parameters

from ..utils.tools import error_if
from ._abc import VolumeTimeSensitivityInterface


_PARAM_MAPPING = {
    parameters.PRIMARY_MASS_SOURCE.name: "mass1_source",
    parameters.PRIMARY_SPIN_X.name: "spin1x",
    parameters.PRIMARY_SPIN_Y.name: "spin1y",
    parameters.PRIMARY_SPIN_Z.name: "spin1z",
    parameters.REDSHIFT.name: "redshift",
    parameters.SECONDARY_MASS_SOURCE.name: "mass2_source",
    parameters.SECONDARY_SPIN_X.name: "spin2x",
    parameters.SECONDARY_SPIN_Y.name: "spin2y",
    parameters.SECONDARY_SPIN_Z.name: "spin2z",
}


class RealInjectionVolumeTimeSensitivity(VolumeTimeSensitivityInterface):
    injections: Array = eqx.field(init=False)
    """Array of real injections of shape (n_injections, n_features)."""
    sampling_prob: Array = eqx.field(init=False)
    """Array of sampling probabilities of shape (n_injections,)."""

    def __init__(
        self,
        parameters: Sequence[str],
        filename: str,
        batch_size: Optional[int] = None,
    ) -> None:
        """Convenience class for loading a neural vt.

        Parameters
        ----------
        parameters : Sequence[str]
            The names of the parameters that the model expects.
        filename : str
            The filename of the neural vt.
        """
        error_if(not parameters, "parameters sequence cannot be empty")
        error_if(
            not isinstance(parameters, Sequence),
            f"parameters must be a Sequence, got {type(parameters)}",
        )
        error_if(
            not set(parameters).difference(_PARAM_MAPPING.values()),
            f"parameters must be one of the following: {set(_PARAM_MAPPING.values())}",
        )
        error_if(
            not all(isinstance(p, str) for p in parameters),
            "all parameters must be strings",
        )
        if batch_size is not None:
            error_if(
                not isinstance(batch_size, int),
                f"batch_size must be an integer, got {type(batch_size)}",
            )
            error_if(
                batch_size < 1,
                f"batch_size must be a positive integer, got {batch_size}",
            )

        self.batch_size = batch_size

        with h5py.File(filename, "r") as f:
            self.injections = jax.device_put(
                np.stack(
                    [f["injections"][_PARAM_MAPPING[p]] for p in parameters], axis=-1
                )
            )
            self.sampling_prob = jax.device_put(f["injections"]["sampling_pdf"][:])

    def get_logVT(self):
        raise NotImplementedError("Injection based VTs do not have a logVT method.")

    def get_mapped_logVT(self):
        raise NotImplementedError("Injection based VTs do not have a logVT method.")
