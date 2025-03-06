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
from typing import Literal, Optional

import equinox as eqx
import h5py
import jax
import numpy as np
from jaxtyping import Array

from gwkokab import parameters as gwk_parameters

from ..utils.tools import error_if, warn_if
from ._abc import VolumeTimeSensitivityInterface


_PARAM_MAPPING = {
    gwk_parameters.PRIMARY_MASS_SOURCE.name: "mass1_source",
    gwk_parameters.PRIMARY_SPIN_MAGNITUDE.name: "spin1z",
    gwk_parameters.PRIMARY_SPIN_X.name: "spin1x",
    gwk_parameters.PRIMARY_SPIN_Y.name: "spin1y",
    gwk_parameters.PRIMARY_SPIN_Z.name: "spin1z",
    gwk_parameters.REDSHIFT.name: "redshift",
    gwk_parameters.SECONDARY_MASS_SOURCE.name: "mass2_source",
    gwk_parameters.SECONDARY_SPIN_MAGNITUDE.name: "spin2z",
    gwk_parameters.SECONDARY_SPIN_X.name: "spin2x",
    gwk_parameters.SECONDARY_SPIN_Y.name: "spin2y",
    gwk_parameters.SECONDARY_SPIN_Z.name: "spin2z",
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
        spin_case: Literal["aligned_spin", "full_spin"] = "aligned_spin",
    ) -> None:
        """Convenience class for loading a neural vt.

        Parameters
        ----------
        parameters : Sequence[str]
            The names of the parameters that the model expects.
        filename : str
            The filename of the neural vt.
        batch_size: Optional[int]
            The batch size of the neural vt.
        spin_case : Literal[&quot;aligned_spin&quot;, &quot;full_spin&quot;]
            The spin case of the injections. Default is 'aligned_spin'.
        """
        error_if(not parameters, msg="parameters sequence cannot be empty")
        error_if(
            not isinstance(parameters, Sequence),
            msg=f"parameters must be a Sequence, got {type(parameters)}",
        )
        error_if(
            not set(parameters).difference(_PARAM_MAPPING.values()),
            msg=f"parameters must be one of the following: {set(_PARAM_MAPPING.values())}",
        )
        error_if(
            not all(isinstance(p, str) for p in parameters),
            msg="all parameters must be strings",
        )
        warn_if(
            batch_size is not None,
            msg="batch_size is not used for injection based VTs",
        )
        error_if(
            spin_case not in ["aligned_spin", "full_spin"],
            msg=f"spin_case must be one of 'aligned_spin' or 'full_spin', got {spin_case}",
        )

        if spin_case == "aligned_spin":
            spin_converter = lambda sx, sy, sz: np.abs(sz)
        else:
            spin_converter = lambda sx, sy, sz: np.sqrt(
                np.square(sx) + np.square(sy) + np.square(sz)
            )

        with h5py.File(filename, "r") as f:
            injs = []
            for p in parameters:
                if p == gwk_parameters.PRIMARY_MASS_SOURCE.name:
                    _inj = spin_converter(
                        f["injections"]["spin1x"][:],
                        f["injections"]["spin1y"][:],
                        f["injections"]["spin1z"][:],
                    )
                elif p == gwk_parameters.SECONDARY_MASS_SOURCE.name:
                    _inj = spin_converter(
                        f["injections"]["spin2x"][:],
                        f["injections"]["spin2y"][:],
                        f["injections"]["spin2z"][:],
                    )
                else:
                    _inj = f["injections"][_PARAM_MAPPING[p]][:]
                injs.append(_inj)
            self.injections = jax.device_put(np.stack(injs, axis=-1), may_alias=True)
            self.sampling_prob = jax.device_put(
                f["injections"]["sampling_pdf"][:],
                may_alias=True,
            )

    def get_logVT(self):
        raise NotImplementedError("Injection based VTs do not have a logVT method.")

    def get_mapped_logVT(self):
        raise NotImplementedError("Injection based VTs do not have a logVT method.")
