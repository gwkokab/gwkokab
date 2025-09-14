# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Dict, Optional, Union

import h5py
import jax
import numpy as np
from jaxtyping import Array
from loguru import logger

from gwkokab.constants import SECONDS_PER_YEAR
from gwkokab.parameters import Parameters

from ..utils.tools import error_if
from ._abc import VolumeTimeSensitivityInterface


_PARAM_MAPPING = {
    Parameters.PRIMARY_MASS_SOURCE.value: "mass1_source",
    Parameters.PRIMARY_SPIN_X.value: "spin1x",
    Parameters.PRIMARY_SPIN_Y.value: "spin1y",
    Parameters.PRIMARY_SPIN_Z.value: "spin1z",
    Parameters.REDSHIFT.value: "redshift",
    Parameters.SECONDARY_MASS_SOURCE.value: "mass2_source",
    Parameters.SECONDARY_SPIN_X.value: "spin2x",
    Parameters.SECONDARY_SPIN_Y.value: "spin2y",
    Parameters.SECONDARY_SPIN_Z.value: "spin2z",
}


class SemiAnalyticalRealInjectionVolumeTimeSensitivity(VolumeTimeSensitivityInterface):
    injections: Array
    """Array of real injections of shape (n_injections, n_features)."""
    sampling_prob: Array
    """Array of sampling probabilities of shape (n_injections,)."""
    analysis_time_years: float
    """Analysis time in years."""
    total_injections: int
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
        ifar_pipelines: Optional[Sequence[str]] = None,
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
        ifar_pipelines: Optional[Sequence[str]]
            Pipelines name to select ifar from.
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

        self.batch_size = batch_size

        logger.debug("Loading injection from {}", filename)

        with h5py.File(filename, "r") as f:
            self.analysis_time_years = (
                float(f.attrs["analysis_time_s"]) / SECONDS_PER_YEAR
            )
            logger.debug("Analysis time: {:.2f} years", self.analysis_time_years)

            self.total_injections = int(f.attrs["total_generated"])
            logger.debug("Total injections: {}", self.total_injections)

            injections = f["injections"]

            if ifar_pipelines is None:
                ifar_pipelines = [k for k in injections.keys() if "ifar" in k]
                logger.debug(
                    "No pipelines specified for ifar, using all available: {}",
                    ", ".join(ifar_pipelines),
                )
            else:
                logger.debug(
                    "Selecting ifar from pipelines: {}", ", ".join(ifar_pipelines)
                )

            ifar = np.max([injections[k][:] for k in ifar_pipelines], axis=0)
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
                if p == Parameters.COS_TILT_1.value:
                    _inj = χ_1z / a1
                elif p == Parameters.COS_TILT_2.value:
                    _inj = χ_2z / a2
                elif p == Parameters.PRIMARY_SPIN_MAGNITUDE.value:
                    _inj = a1
                    # We parameterize spins in spherical coordinates, neglecting azimuthal
                    # parameters. The injections are parameterized in terms of cartesian
                    # spins. The Jacobian is `1 / (2 pi magnitude ** 2)`.
                    logger.debug(
                        "Correcting sampling probability for spherical spin "
                        "parameterization of primary spin."
                    )
                    sampling_prob *= 2.0 * np.pi * np.square(a1)
                elif p == Parameters.SECONDARY_SPIN_MAGNITUDE.value:
                    _inj = a2
                    # We parameterize spins in spherical coordinates, neglecting azimuthal
                    # parameters. The injections are parameterized in terms of cartesian
                    # spins. The Jacobian is `1 / (2 pi magnitude ** 2)`.
                    logger.debug(
                        "Correcting sampling probability for spherical spin "
                        "parameterization of secondary spin."
                    )
                    sampling_prob *= 2.0 * np.pi * np.square(a2)
                else:
                    _inj = injections[_PARAM_MAPPING[p]][found][:]
                injs.append(_inj)

            if (
                Parameters.PRIMARY_SPIN_MAGNITUDE.value not in parameters
                and Parameters.SECONDARY_SPIN_MAGNITUDE.value not in parameters
            ):
                # Eliminating the probability of cartesian spins
                logger.debug(
                    "Eliminating the probability of cartesian spins from sampling probability."
                )
                sampling_prob *= np.square(4.0 * np.pi * a1 * a2)

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
