# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Dict, Optional, Union

import equinox as eqx
import h5py
import jax
import numpy as np
from jaxtyping import Array
from loguru import logger

from gwkokab import parameters as gwk_parameters
from gwkokab.constants import SECONDS_PER_YEAR

from ..utils.tools import error_if
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


class SemiAnalyticalRealInjectionVolumeTimeSensitivity(VolumeTimeSensitivityInterface):
    injections: Array = eqx.field(init=False)
    """Array of real injections of shape (n_injections, n_features)."""
    sampling_prob: Array = eqx.field(init=False)
    """Array of sampling probabilities of shape (n_injections,)."""
    analysis_time_years: float = eqx.field(init=False)
    """Analysis time in years."""
    total_injections: int = eqx.field(init=False)
    """Total number of injections.

    This includes both accepted and rejected injections.
    """

    def __init__(
        self,
        parameters: Sequence[str],
        filename: str,
        batch_size: Optional[int] = None,
        far_cut: float = 1.0,
        snr_cut: float = 10.0,
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
        far_cut : float
            The FAR cut to apply to the injections. Default is 1.0.
        snr_cut : float
            The SNR cut to apply to the injections. Default is 10.0.
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

        self.batch_size = batch_size

        logger.debug("Loading injection from {}", filename)

        with h5py.File(filename, "r") as f:
            self.analysis_time_years = (
                float(f.attrs["analysis_time_s"]) / SECONDS_PER_YEAR
            )
            self.total_injections = int(f.attrs["total_generated"])

            injections = f["injections"]

            ifar = np.max(
                [injections[k][:] for k in injections.keys() if "far" in k],
                axis=0,
            )
            snr = injections["optimal_snr_net"][:]

            runs = injections["name"][:].astype(str)
            found = np.where(runs == "o3", ifar > 1 / far_cut, snr > snr_cut)

            n_total = found.shape[0]
            n_found = np.sum(found)
            logger.debug(
                "Found {} out of {} injections with FAR < {} and SNR > {}",
                n_found,
                n_total,
                far_cut,
                snr_cut,
            )

            sampling_prob = (
                injections["sampling_pdf"][found][:]
                / injections["mixture_weight"][found][:]
            )

            χ_1x = injections["spin1x"][found][:]
            χ_1y = injections["spin1y"][found][:]
            χ_1z = injections["spin1z"][found][:]
            χ_2x = injections["spin2x"][found][:]
            χ_2y = injections["spin2y"][found][:]
            χ_2z = injections["spin2z"][found][:]
            a1 = np.sqrt(np.square(χ_1x) + np.square(χ_1y) + np.square(χ_1z))
            a2 = np.sqrt(np.square(χ_2x) + np.square(χ_2y) + np.square(χ_2z))

            injs = []
            for p in parameters:
                if p == "cos_tilt_1":
                    _inj = χ_1z / a1
                elif p == "cos_tilt_2":
                    _inj = χ_2z / a2
                elif p == gwk_parameters.PRIMARY_SPIN_MAGNITUDE.name:
                    sampling_prob *= 2.0 * np.pi * np.square(a1)
                    _inj = a1
                elif p == gwk_parameters.SECONDARY_SPIN_MAGNITUDE.name:
                    sampling_prob *= 2.0 * np.pi * np.square(a2)
                    _inj = a2
                else:
                    _inj = injections[_PARAM_MAPPING[p]][found][:]
                injs.append(_inj)
            self.injections = jax.device_put(np.stack(injs, axis=-1), may_alias=True)

            self.sampling_prob = jax.device_put(sampling_prob, may_alias=True)

            param_ranges: Dict[str, Union[int, float]] = {}

            for i, p in enumerate(parameters):
                try:
                    minimum = f.attrs[p + "_min"]
                except KeyError:
                    minimum = np.min(self.injections[:, i])
                try:
                    maximum = f.attrs[p + "_max"]
                except KeyError:
                    maximum = np.max(self.injections[:, i])

                param_ranges[p + "_min"] = minimum
                param_ranges[p + "_max"] = maximum
            self.parameter_ranges = param_ranges

    def get_logVT(self):
        raise NotImplementedError("Injection based VTs do not have a logVT method.")

    def get_mapped_logVT(self):
        raise NotImplementedError("Injection based VTs do not have a logVT method.")
