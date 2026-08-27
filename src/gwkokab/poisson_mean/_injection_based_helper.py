# Copyright (c) 2021 Colm Talbot
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Reading and reweighting sensitivity injection files.

Injection releases differ between observing runs in their column names, in whether they
report a false alarm rate at all, and in how the draw density is stored -- as a single
``sampling_pdf``, as one joint ``lnpdraw_*`` column, or as a set of factors to be
summed. :func:`load_injection_data` absorbs those differences and returns one dictionary
in a common form.

:func:`apply_injection_prior` then changes coordinates from the ones the injections were
drawn in (component masses, Cartesian spins, redshift) to whichever the analysis uses,
carrying the Jacobian into the ``prior`` entry so the importance weights stay correct.

This module is adapted from
`gwpopulation <https://github.com/ColmTalbot/gwpopulation>`_
and carries its own copyright notice.
"""

from typing import Dict, List, Tuple

import h5py
import numpy as np
from jaxtyping import Array
from loguru import logger

from ..constants import SECONDS_PER_YEAR
from ..cosmology import default_cosmology
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
    """Select the injections a search recovered.

    Whichever criterion the file supports is used: an inverse false alarm rate threshold
    where FAR or IFAR columns are present -- an injection counts as found if *any*
    pipeline recovers it -- and an SNR threshold otherwise, which is the case for O1 and
    O2 where the pipelines report no FAR.

    Parameters
    ----------
    data : Dict[str, Array]
        The injection columns, or a read-only HDF5 group standing in for them.
    shape : Tuple[int, ...]
        Shape of the boolean mask to build, i.e. of one injection column.
    ifar_threshold : float, optional
        Threshold on inverse false alarm rate, in years. Defaults to ``1.0``;
        :data:`None` disables the cut.
    snr_threshold : float, optional
        SNR threshold. Defaults to ``10.0``.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``shape``, true for found injections.

    Raises
    ------
    ValueError
        If the file carries neither a FAR/IFAR column nor a recognised SNR column.
    """
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

    if has_ifar:
        ifar_values = {
            key: data[key][()] for key in data.keys() if "ifar" in key.lower()
        }
    elif len(far_keys) > 0:
        # `data` may be a read-only HDF5 group, so the inverse false alarm rates
        # derived from the FAR columns are kept locally rather than written back.
        ifar_values = {
            far_key.replace("far", "ifar"): 1 / data[far_key][()]
            for far_key in far_keys
        }
        has_ifar = True
    else:
        ifar_values = {}
    if ifar_threshold is None:
        ifar_threshold = 1e300
    if has_ifar:
        for ifar in ifar_values.values():
            found |= ifar > ifar_threshold
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
            analysis_time = data.attrs["analysis_time_s"][()] / SECONDS_PER_YEAR
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
                    if any([
                        key.startswith(f"{substr}_"),
                        key.endswith(f"_{substr}"),
                        f"_{substr}_" in key,
                    ])
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
                    analysis_time = ff.attrs[key][()] / SECONDS_PER_YEAR
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
            mass_2=np.asarray(data[mass_2_key][()][found]),
            redshift=np.asarray(data[redshift_key][()][found]),
            total_generated=total_generated,
            analysis_time=analysis_time,
            idx=np.arange(data[mass_1_key].shape[0]),
        )
        for ii in [1, 2]:
            gwpop_data[f"a_{ii}"] = (
                np.asarray(
                    data.get(f"spin{ii}x", np.zeros(found_shape))[()][found] ** 2
                    + data.get(f"spin{ii}y", np.zeros(found_shape))[()][found] ** 2
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

        weights = 1.0  # type: ignore
        if "v1_1ifo" in vt_file:
            weights *= np.asarray(data["weights_1ifo"][()][found])
        elif "weights" in data:
            weights *= np.asarray(data["weights"][()][found])
        gwpop_data["prior"] /= weights
    return gwpop_data


def apply_injection_prior(data: Dict[str, Array], parameters: List[str]):
    r"""Derive the analysis coordinates and carry the draw density across.

    The injections are drawn in source-frame component masses, Cartesian spins and
    redshift. For each coordinate the analysis asks for, this adds the corresponding
    column to ``data`` and multiplies the ``prior`` entry by the Jacobian of the change
    of variables, so the importance weights of
    :func:`~gwkokab.poisson_mean.poisson_mean_from_sensitivity_injections` remain
    correct.

    The spin coordinates are the interesting case: :math:`\chi_{\text{eff}}` and
    :math:`\chi_p` are not invertible functions of the drawn spins, so the induced
    density has no Jacobian and is instead supplied analytically by
    :mod:`~gwkokab.poisson_mean._analytic_spin_prior`, assuming isotropic spin
    orientations with :math:`a_{\max} = 1`.

    Parameters
    ----------
    data : Dict[str, Array]
        Injection data as returned by :func:`load_injection_data`. Modified in place.
    parameters : List[str]
        Names of the coordinates the analysis needs.

    Returns
    -------
    Dict[str, Array]
        ``data``, with the requested coordinates added and ``prior`` updated.
    """
    if P.MASS_RATIO in parameters:
        data[P.MASS_RATIO] = data[P.SECONDARY_MASS_SOURCE] / data[P.PRIMARY_MASS_SOURCE]
        data["prior"] *= data[P.PRIMARY_MASS_SOURCE]
    if P.CHIRP_MASS in parameters:
        jacobian = primary_mass_to_chirp_mass_jacobian(data[P.MASS_RATIO])
        data[P.CHIRP_MASS] = data[P.PRIMARY_MASS_SOURCE] / jacobian
        data["prior"] *= jacobian
    if P.EFFECTIVE_SPIN in parameters:
        data[P.EFFECTIVE_SPIN] = m1_m2_chi1_chi2_costilt1_costilt2_to_chieff(
            m1=data[P.PRIMARY_MASS_SOURCE],
            m2=data[P.SECONDARY_MASS_SOURCE],
            chi1=data[P.PRIMARY_SPIN_MAGNITUDE],
            chi2=data[P.SECONDARY_SPIN_MAGNITUDE],
            costilt1=data[P.COS_TILT_1],
            costilt2=data[P.COS_TILT_2],
        )  # type: ignore

        if P.MASS_RATIO not in data:
            data[P.MASS_RATIO] = (
                data[P.SECONDARY_MASS_SOURCE] / data[P.PRIMARY_MASS_SOURCE]
            )

        if P.PRECESSING_SPIN in parameters:
            data[P.PRECESSING_SPIN] = chi_p_from_components(
                a_1=data[P.PRIMARY_SPIN_MAGNITUDE],
                cos_tilt_1=data[P.COS_TILT_1],
                a_2=data[P.SECONDARY_SPIN_MAGNITUDE],
                cos_tilt_2=data[P.COS_TILT_2],
                mass_ratio=data[P.MASS_RATIO],
            )  # type: ignore
            amax = 1
            logger.info(
                f"Applying isotropic prior to chi_eff and chi_p, assuming injections with amax={amax}."
            )
            p_chi_iso = prior_chieff_chip_isotropic(
                data[P.EFFECTIVE_SPIN],
                data[P.PRECESSING_SPIN],
                data[P.MASS_RATIO],
                amax=amax,
            )
        else:
            amax = 1
            logger.info(
                f"Applying isotropic prior to chi_eff, assuming injections with amax={amax}."
            )
            p_chi_iso = chi_effective_prior_from_isotropic_spins(
                data[P.EFFECTIVE_SPIN],
                data[P.MASS_RATIO],
                amax=amax,
            )
        p_magnitude_costilt_iso = (1 / 2) ** 2 * (1 / amax) ** 2
        data["prior"] *= p_chi_iso / p_magnitude_costilt_iso
    if P.CHI_1 in parameters:
        data[P.CHI_1] = data[P.PRIMARY_SPIN_MAGNITUDE] * data[P.COS_TILT_1]
        data["prior"] *= 2 * aligned_spin_prior(data[P.CHI_1])
    if P.CHI_2 in parameters:
        data[P.CHI_2] = data[P.SECONDARY_SPIN_MAGNITUDE] * data[P.COS_TILT_2]
        data["prior"] *= 2 * aligned_spin_prior(data[P.CHI_2])
    if P.PRIMARY_MASS_DETECTED in parameters:
        data[P.PRIMARY_MASS_DETECTED] = data[P.PRIMARY_MASS_SOURCE] * (
            1 + data[P.REDSHIFT]
        )
        data["prior"] /= 1 + data[P.REDSHIFT]
    if P.SECONDARY_MASS_DETECTED in parameters:
        data[P.SECONDARY_MASS_DETECTED] = (
            data[P.PRIMARY_MASS_DETECTED] * data[P.MASS_RATIO]
        )
        data["prior"] /= data[P.PRIMARY_MASS_DETECTED]
    if P.CHIRP_MASS_DETECTOR in parameters:
        jacobian = primary_mass_to_chirp_mass_jacobian(data[P.MASS_RATIO])
        try:
            data[P.CHIRP_MASS_DETECTOR] = data[P.PRIMARY_MASS_DETECTED] / jacobian
            data["prior"] *= jacobian
        except (KeyError, AttributeError, TypeError):
            data[P.CHIRP_MASS_DETECTOR] = (
                data[P.PRIMARY_MASS_SOURCE] * (1 + data[P.REDSHIFT]) / jacobian
            )
            data["prior"] *= jacobian / (1 + data[P.REDSHIFT])
    if P.LUMINOSITY_DISTANCE in parameters:
        cosmo = default_cosmology()

        data[P.LUMINOSITY_DISTANCE] = cosmo.z_to_DL(data[P.REDSHIFT])  # type: ignore
        data["prior"] /= cosmo.dDLdz(data[P.REDSHIFT])  # type: ignore
    return data
