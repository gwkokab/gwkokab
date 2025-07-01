# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Optional

import equinox as eqx
import h5py
import jax
import numpy as np
from jaxtyping import Array

from gwkokab.constants import SECONDS_PER_YEAR

from ..utils.tools import error_if, warn_if
from ._abc import VolumeTimeSensitivityInterface


class SyntheticInjectionVolumeTimeSensitivity(VolumeTimeSensitivityInterface):
    injections: Array = eqx.field(init=False)
    """Array of real injections of shape (n_injections, n_features)."""
    sampling_prob: Array = eqx.field(init=False)
    """Array of sampling probabilities of shape (n_injections,)."""
    analysis_time_years: float = eqx.field(init=False)
    """Analysis time in years."""
    total_injections: int = eqx.field(init=False)
    """Total number of injections."""

    def __init__(
        self, parameters: Sequence[str], filename: str, batch_size: Optional[int] = None
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
            not all(isinstance(p, str) for p in parameters),
            msg="all parameters must be strings",
        )
        warn_if(
            batch_size is not None,
            msg="batch_size is not used for injection based VTs",
        )

        with h5py.File(filename, "r") as f:
            self.analysis_time_years = (
                float(f.attrs["analysis_time_s"]) / SECONDS_PER_YEAR
            )
            injs = [f[p][:] for p in parameters]
            self.injections = jax.device_put(np.stack(injs, axis=-1), may_alias=True)
            self.sampling_prob = jax.device_put(f["sampling_pdf"][:], may_alias=True)
            self.total_injections = self.sampling_prob.shape[0]

    def get_logVT(self):
        raise NotImplementedError("Injection based VTs do not have a logVT method.")

    def get_mapped_logVT(self):
        raise NotImplementedError("Injection based VTs do not have a logVT method.")
