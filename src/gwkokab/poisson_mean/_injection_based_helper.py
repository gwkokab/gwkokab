# Copyright (c) 2021 Colm Talbot
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Dict, List, Tuple

import h5py
import numpy as np
from jaxtyping import Array
from loguru import logger

from ..cosmology import PLANCK_2015_Cosmology
from ..parameters import Parameters as P
from ..utils.transformations import (
    chi_p_from_components,
    m1_m2_chi1_chi2_costilt1_costilt2_to_chieff,
)
from ._analytic_spin_prior import (
    chi_effective_prior_from_isotropic_spins,
    prior_chieff_chip_isotropic,
)


def aligned_spin_prior(spin):
    r"""The standard prior for aligned spin assuming the spin prior extends to maximal.

    .. math::

        p(\chi) = \frac{1}{2} \log(|\chi|)

    Parameters
    ----------
    spin: array_like
        The aligned spin values to evaluate the prior for.

    Returns
    -------
    prior: array_like
        The prior evaluated at the input spin.
    """
    return -np.log(np.abs(spin)) / 2


def primary_mass_to_chirp_mass_jacobian(q):
    r"""Compute the Jacobian for the primary mass to chirp mass transformation.

    .. math::

        \frac{d m_c}{d m_1} = \frac{q^{3/5}}{(1 + q)^{1/5}}

    Parameters
    ----------
    samples: dict
        Samples containing `mass_1` and `mass_ratio`.

    Returns
    -------
    jacobian: array_like
        The Jacobian for the transformation.
    """
    return (1 + q) ** 0.2 / q**0.6


def get_found_injections(
    data: Dict[str, Array],
    shape: Tuple[int, ...],
    ifar_threshold: float = 1.0,
    snr_threshold: float = 10.0,
):
    found = np.zeros(shape, dtype=bool)
    has_ifar = any(["ifar" in key.lower() for key in data.keys()])

    far_keys = list(
        filter(
            lambda key: (
                key.lower().startswith("far_")
                or key.lower().endswith("_far")
                or "_far_" in key.lower()
            ),
            data,
        )
    )

    if not has_ifar and len(far_keys) > 0:
        for far_key in far_keys:
            data[far_key.replace("far", "ifar")] = 1 / data[far_key][()]
        has_ifar = True
    if ifar_threshold is None:
        ifar_threshold = 1e300
    if has_ifar:
        for key in data:
            if "ifar" in key.lower():
                found |= data[key][()] > ifar_threshold
            if "name" in data.keys():
                gwtc1 = (data["name"][()] == b"o1") | (data["name"][()] == b"o2")
                found |= gwtc1 & (data["optimal_snr_net"][()] > snr_threshold)
        if "semianalytic_observed_phase_maximized_snr_net" in data.keys():
            found |= (
                data["semianalytic_observed_phase_maximized_snr_net"][()]
                > snr_threshold
            )
        return found
    elif snr_threshold is not None:
        if "observed_phase_maximized_snr_net" in data.keys():
            found |= data["observed_phase_maximized_snr_net"][()] > snr_threshold
        elif "observed_snr_net" in data.keys():
            found |= data["observed_snr_net"][()] > snr_threshold
        return found
    else:
        raise ValueError("Cannot find keys to filter sensitivity injections.")


def load_injection_data(
    vt_file: str, ifar_threshold: float = 1.0, snr_threshold: float = 10.0
) -> Dict[str, Array]:
    """Load the injection file in the O3 injection file format.

    For mixture files and multiple observing run files we only
    have the full `sampling_pdf`.

    We use a different parameterization than the default so we require a few
    changes.

    - we parameterize spins in spherical coordinates, neglecting azimuthal
      parameters. The injections are parameterized in terms of cartesian
      spins. The Jacobian is `1 / (2 pi magnitude ** 2)`.

    For O3 injections we threshold on FAR.
    For O1/O2 injections we threshold on SNR as there is no FAR
    provided by the search pipelines.

    Parameters
    ----------
    vt_file: str
        The path to the hdf5 file containing the injections.
    ifar_threshold: float
        The threshold on inverse false alarm rate in years. Default=1.
    snr_threshold: float
        The SNR threshold when there is no FAR. Default=10.

    Returns
    -------
    gwpop_data: dict
        Data required for evaluating the selection function.
    """
    logger.info(f"Loading VT data from {vt_file}.")

    with h5py.File(vt_file, "r") as ff:
        if "injections" in ff:
            data = ff["injections"]
            total_generated = int(data.attrs["total_generated"][()])
            analysis_time = data.attrs["analysis_time_s"][()] / 365.25 / 24 / 60 / 60
        elif "events" in ff:
            keys_of_interest = {
                "mass1_source",
                "mass2_source",
                "mass_1_source",
                "mass_2_source",
                "spin1x",
                "spin1y",
                "spin1z",
                "spin2x",
                "spin2y",
                "spin2z",
                "redshift",
                "z",
                "sampling_pdf",
                "lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z",
                "lnpdraw_mass1_source",
                "lnpdraw_mass2_source_GIVEN_mass1_source",
                "lnpdraw_z",
                "lnpdraw_spin1_magnitude",
                "lnpdraw_spin2_magnitude",
                "lnpdraw_spin1_polar_angle",
                "lnpdraw_spin2_polar_angle",
                "v1_1ifo",
                "weights",
                "weights_1ifo",
                "name",
                "observed_phase_maximized_snr_net",
                "observed_snr_net",
                "optimal_snr_net",
                "semianalytic_observed_phase_maximized_snr_net",
            }
            keys = list(keys_of_interest.intersection(ff["events"].dtype.names))
            for substr in ["far", "ifar"]:
                keys += [
                    key
                    for key in ff["events"].dtype.names
                    if any(
                        [
                            key.startswith(f"{substr}_"),
                            key.endswith(f"_{substr}"),
                            f"_{substr}_" in key,
                        ]
                    )
                ]

            data = {key: np.array(ff["events"][key][()]) for key in keys}
            total_generated = int(ff.attrs["total_generated"][()])
            # the name applied to the analysis time changes between files, so we
            # loop over all plausible values and break once we find one
            for key in [
                "total_analysis_time",
                "analysis_time",
                "total_analysis_time_1ifo",
            ]:
                if key in ff.attrs:
                    analysis_time = ff.attrs[key][()] / 365.25 / 24 / 60 / 60
                    break
            else:
                raise AttributeError(
                    "Provided injection file does not provide analysis time"
                )
            if analysis_time == 0:
                analysis_time = 1 / 12
        else:
            raise KeyError(f"Unable to identify injections from {ff.keys()}")

        if "mass1_source" in data:
            mass_1_key = "mass1_source"
            mass_2_key = "mass2_source"
        else:
            mass_1_key = "mass_1_source"
            mass_2_key = "mass_2_source"
        if "redshift" in data:
            redshift_key = "redshift"
        else:
            redshift_key = "z"
        found_shape = data[mass_1_key][()].shape
        found = get_found_injections(data, found_shape, ifar_threshold, snr_threshold)
        n_found = sum(found)
        if n_found == 0:
            raise ValueError("No sensitivity injections pass threshold.")
        gwpop_data = dict(
            mass_1=np.asarray(data[mass_1_key][()][found]),
            mass_ratio=np.asarray(
                data[mass_2_key][()][found] / data[mass_1_key][()][found]
            ),
            redshift=np.asarray(data[redshift_key][()][found]),
            total_generated=total_generated,
            analysis_time=analysis_time,
            idx=np.arange(data[mass_1_key].shape[0]),
        )
        for ii in [1, 2]:
            gwpop_data[f"a_{ii}"] = (
                np.asarray(
                    data.get(f"spin{ii}x", np.zeros(n_found))[()][found] ** 2
                    + data.get(f"spin{ii}y", np.zeros(n_found))[()][found] ** 2
                    + data[f"spin{ii}z"][()][found] ** 2
                )
                ** 0.5
            )
            gwpop_data[f"cos_tilt_{ii}"] = (
                np.asarray(data[f"spin{ii}z"][()][found]) / gwpop_data[f"a_{ii}"]
            )
        if (
            "sampling_pdf" in data
        ):  # O1+O2+O3 mixture and endO3 injections (https://dcc.ligo.org/LIGO-T2100377, https://dcc.ligo.org/LIGO-T2100113)
            gwpop_data["prior"] = np.asarray(
                data["sampling_pdf"][()][found]
            ) * np.square(2 * np.pi * gwpop_data["a_1"] * gwpop_data["a_2"])
        elif (
            "lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
            in data
        ):  # O1+O2+O3+O4a mixture (https://dcc.ligo.org/LIGO-T2400110)
            gwpop_data["prior"] = np.exp(
                np.asarray(
                    data[
                        "lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
                    ][()][found]
                )
                + 2.0 * np.log(2 * np.pi * gwpop_data["a_1"] * gwpop_data["a_2"])
            )
        else:  # O4a sensitivity injections (https://dcc.ligo.org/LIGO-T2400073)
            gwpop_data["prior"] = np.exp(
                np.sum(
                    [
                        np.asarray(data[f"lnpdraw_{key}"][()][found])
                        for key in [
                            "mass1_source",
                            "mass2_source_GIVEN_mass1_source",
                            "z",
                            "spin1_magnitude",
                            "spin2_magnitude",
                            "spin1_polar_angle",
                            "spin2_polar_angle",
                        ]
                    ],
                    axis=0,
                )
            )
            gwpop_data["prior"] /= np.sin(np.arccos(gwpop_data["cos_tilt_1"]))
            gwpop_data["prior"] /= np.sin(np.arccos(gwpop_data["cos_tilt_2"]))

        weights = np.ones(())
        if "v1_1ifo" in vt_file:
            weights *= np.asarray(data["weights_1ifo"][()][found])
        elif "weights" in data:
            weights *= np.asarray(data["weights"][()][found])
        gwpop_data["prior"] /= weights
    return gwpop_data


def apply_injection_prior(data: Dict[str, Array], parameters: List[str]):
    """We assume the injection prior in terms of the source frame primary mass and mass
    ratio.
    """

    if P.SECONDARY_MASS_SOURCE.value in parameters:
        data[P.SECONDARY_MASS_SOURCE.value] = (
            data[P.PRIMARY_MASS_SOURCE.value] * data[P.MASS_RATIO.value]
        )
        data["prior"] /= data[P.PRIMARY_MASS_SOURCE.value]
    if P.CHIRP_MASS.value in parameters:
        jacobian = primary_mass_to_chirp_mass_jacobian(data[P.MASS_RATIO.value])
        data[P.CHIRP_MASS.value] = data[P.PRIMARY_MASS_SOURCE.value] / jacobian
        data["prior"] *= jacobian
    if P.EFFECTIVE_SPIN_MAGNITUDE.value in parameters:
        data[P.EFFECTIVE_SPIN_MAGNITUDE.value] = (
            m1_m2_chi1_chi2_costilt1_costilt2_to_chieff(
                m1=data[P.PRIMARY_MASS_SOURCE.value],
                m2=data[P.SECONDARY_MASS_SOURCE.value],
                chi1=data[P.PRIMARY_SPIN_MAGNITUDE.value],
                chi2=data[P.SECONDARY_SPIN_MAGNITUDE.value],
                costilt1=data[P.COS_TILT_1.value],
                costilt2=data[P.COS_TILT_2.value],
            )  # type: ignore
        )
        if P.PRECESSING_SPIN_MAGNITUDE.value in parameters:
            data[P.PRECESSING_SPIN_MAGNITUDE.value] = chi_p_from_components(
                a_1=data[P.PRIMARY_SPIN_MAGNITUDE.value],
                cos_tilt_1=data[P.COS_TILT_1.value],
                a_2=data[P.SECONDARY_SPIN_MAGNITUDE.value],
                cos_tilt_2=data[P.COS_TILT_2.value],
                mass_ratio=data[P.MASS_RATIO.value],
            )  # type: ignore
            amax = 1
            logger.info(
                f"Applying isotropic prior to chi_eff and chi_p, assuming injections with amax={amax}."
            )
            p_chi_iso = prior_chieff_chip_isotropic(
                data[P.EFFECTIVE_SPIN_MAGNITUDE.value],
                data[P.PRECESSING_SPIN_MAGNITUDE.value],
                data[P.MASS_RATIO.value],
                amax=amax,
            )
        else:
            amax = 1
            logger.info(
                f"Applying isotropic prior to chi_eff, assuming injections with amax={amax}."
            )
            p_chi_iso = chi_effective_prior_from_isotropic_spins(
                data[P.EFFECTIVE_SPIN_MAGNITUDE.value],
                data[P.MASS_RATIO.value],
                amax=amax,
            )
        p_magnitude_costilt_iso = (1 / 2) ** 2 * (1 / amax) ** 2
        data["prior"] *= p_chi_iso / p_magnitude_costilt_iso
    if P.CHI_1.value in parameters:
        data[P.CHI_1.value] = (
            data[P.PRIMARY_SPIN_MAGNITUDE.value] * data[P.COS_TILT_1.value]
        )
        data["prior"] *= 2 * aligned_spin_prior(data[P.CHI_1.value])
    if P.CHI_2.value in parameters:
        data[P.CHI_2.value] = (
            data[P.SECONDARY_SPIN_MAGNITUDE.value] * data[P.COS_TILT_2.value]
        )
        data["prior"] *= 2 * aligned_spin_prior(data[P.CHI_2.value])
    if P.PRIMARY_MASS_DETECTED.value in parameters:
        data[P.PRIMARY_MASS_DETECTED.value] = data[P.PRIMARY_MASS_SOURCE.value] * (
            1 + data[P.REDSHIFT.value]
        )
        data["prior"] /= 1 + data[P.REDSHIFT.value]
    if P.SECONDARY_MASS_DETECTED.value in parameters:
        data[P.SECONDARY_MASS_DETECTED.value] = (
            data[P.PRIMARY_MASS_DETECTED.value] * data[P.MASS_RATIO.value]
        )
        data["prior"] /= data[P.PRIMARY_MASS_DETECTED.value]
    if P.CHIRP_MASS_DETECTOR.value in parameters:
        jacobian = primary_mass_to_chirp_mass_jacobian(data[P.MASS_RATIO.value])
        try:
            data[P.CHIRP_MASS_DETECTOR.value] = (
                data[P.PRIMARY_MASS_DETECTED.value] / jacobian
            )
            data["prior"] *= jacobian
        except (KeyError, AttributeError, TypeError):
            data[P.CHIRP_MASS_DETECTOR.value] = (
                data[P.PRIMARY_MASS_SOURCE.value]
                * (1 + data[P.REDSHIFT.value])
                / jacobian
            )
            data["prior"] *= jacobian / (1 + data[P.REDSHIFT.value])
    if P.LUMINOSITY_DISTANCE.value in parameters:
        cosmo = PLANCK_2015_Cosmology()

        data[P.LUMINOSITY_DISTANCE.value] = cosmo.z_to_DL(data[P.REDSHIFT.value])  # type: ignore
        data["prior"] /= cosmo.dDLdz(data[P.REDSHIFT.value])  # type: ignore
    return data
