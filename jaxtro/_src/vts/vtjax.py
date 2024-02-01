#  Copyright 2023 The Jaxtro Authors
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


from typing_extensions import Optional

import astropy.cosmology as cosmo
import astropy.units as u
import lal
import lalsimulation as ls
import scipy.integrate as si  # Optimizeed integral needed
from jax import numpy as jnp


def next_pow_two(x: int) -> int:
    # """Return the next (integer) power of two above `x`."""
    x2 = 1
    while x2 < x:
        x2 = x2 << 1
    return x2


def optimal_snr(
    m1: float,
    m2: float,
    a1z: float,
    a2z: float,
    z: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    psd_fn: Optional[int] = None,
    approximant: Optional[int] = None,
) -> float:
    """Return the optimal SNR of a signal.

    :param m1: The source-frame mass 1.

    :param m2: The source-frame mass 2.

    :param a1z: The z-component of spin 1.

    :param a2z: The z-component of spin 2.

    :param z: The redshift.

    :param fmin: The starting frequency for waveform generation.

    :param psd_fn: A function that returns the detector PSD at a given
    frequency, you can choose any given in lalsimulation.
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_noise_p_s_d__c.html
    #ga02d7e9443530dbd0957dff1a08a0ab3c

    :return: The SNR of a face-on, overhead source.

    """

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDAdVO4T1800545  # psd of single detector for O4

    if approximant is None:
        approximant = ls.IMRPhenomB  # alligned spin approximant
    # https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_i_m_r_phenom__c.html
    # #ga9117e5a155732b9922eab602930377d7

    # Get dL, Gpc
    dL = cosmo.Planck15.luminosity_distance(z).to(u.Gpc).value

    tmax = ls.SimInspiralChirpTimeBound(fmin, m1 * (1 + z) * lal.MSUN_SI, m2 * (1 + z) * lal.MSUN_SI, a1z, a2z) + 2.0

    df = max(1.0 / next_pow_two(tmax), dfmin)
    fmax = 2048.0  # Hz --- based on max freq of 5-5 inspiral

    # Compute the GW strain. g(t) = hp*Fps + hc*Fxs

    hp, hc = ls.SimInspiralChooseFDWaveform(
        ((1 + z) * m1 * lal.MSUN_SI),  # REAL8_const_m1
        ((1 + z) * m2 * lal.MSUN_SI),  # REAL8_const_m2
        0.0,  # REAL8_const_S1x
        0.0,  # REAL8_const_S1y
        a1z,  # REAL8_const_S1z
        0.0,  # REAL8_const_S2x
        0.0,  # REAL8_const_S2y
        a2z,  # REAL8_const_S2z
        dL * 1e9 * lal.PC_SI,  # REAL8_const_distance
        0.0,  # REAL8_const_inclination
        0.0,  # REAL8_const_phiRef
        0.0,  # REAL8_const_longAscNodes
        0.0,  # REAL8_const_eccentricity
        0.0,  # REAL8_const_meanPerAno
        df,  # REAL8_const_deltaF
        fmin,  # REAL8_const_f_min
        fmax,  # REAL8_const_f_max
        fref,  # REAL8_f_ref
        None,  # Dict_LALpars
        approximant,  # Approximant_const_approximant
    )

    Nf = int(round(fmax / df)) + 1
    fs = jnp.linspace(0, fmax, Nf)

    # PSD is in units of 1/Hz
    # sffs is the frequency series of the PSD
    sffs = lal.CreateREAL8FrequencySeries("psds", 0, 0.0, df, lal.DimensionlessUnit, fs.shape[0])
    psd_fn(sffs, psdstart)
    return ls.MeasureSNRFD(hp, sffs, psdstart, -1.0)


# Computing p_{det}.
def fraction_above_threshold(
    m1: float,
    m2: float,
    a1z: float,
    a2z: float,
    z: float,
    snr_thresh: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    psd_fn: Optional[int] = None,
    approximant: Optional[int] = None,
) -> float:
    if z == 0.0:
        return 1.0

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDAdVO4T1800545

    rho_max = optimal_snr(
        m1,
        m2,
        a1z,
        a2z,
        z,
        fmin=fmin,
        dfmin=dfmin,
        fref=fref,
        psdstart=psdstart,
        psd_fn=psd_fn,
        approximant=approximant,
    )

    # w = (jnp.asarray(snr_thresh) / rho_max)[()]
    w = snr_thresh / rho_max
    if w > 1:  # no detection
        return 0.0
    a2, a4, a8 = 0.374222, 2.04216, -2.63948
    # slightly higher than Dan's value
    P_det = a2 * ((1 - w) ** 2) + a4 * ((1 - w) ** 4) + a8 * ((1 - w) ** 8) + (1 - a2 - a4 - a8) * ((1 - w) ** 10)
    return P_det


# Computing VT
def vt_from_mass_spin(
    m1: float,
    m2: float,
    a1z: float,
    a2z: float,
    thresh: float,
    analysis_time: float,
    fmin: float = 19.0,
    dfmin: float = 0.0,
    fref: float = 40.0,
    psdstart: float = 20.0,
    zmax: float = 1.5,
    psd_fn: Optional[int] = None,
    approximant: Optional[int] = None,
) -> float:
    """Returns the sensitive time-volume for a given system: A float.

    :param m1: Source-frame mass 1.

    :param m2: Source-frame mass 2.

    :param a1z: The z-component of spin 1.

    :param a2z: The z-component of spin 2.

    :param analysis_time: The total detector-frame searched time.

    :param fmin: The starting frequency for waveform generation.

    :param psd_fn: Function giving the assumed single-detector PSD
      (see :func:`optimal_snr`).

    :return: The sensitive time-volume in comoving Gpc^3-yr (assuming
      analysis_time is given in years).

    """

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDAdVO4T1800545

    def integrand(z) -> float:
        if z == 0.0:
            return 0.0
        return (
            4
            * jnp.pi
            * cosmo.Planck15.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value
            / (1 + z)
            * fraction_above_threshold(
                m1,
                m2,
                a1z,
                a2z,
                z,
                thresh,
                fmin=fmin,
                dfmin=dfmin,
                fref=fref,
                psdstart=psdstart,
                psd_fn=psd_fn,
                approximant=approximant,
            )
        )

    zmin = 0.001
    assert (
        fraction_above_threshold(
            m1,
            m2,
            a1z,
            a2z,
            zmax,
            thresh,
            fmin=fmin,
            dfmin=dfmin,
            fref=fref,
            psdstart=psdstart,
            psd_fn=psd_fn,
            approximant=approximant,
        )
        == 0.0
    )

    while zmax - zmin > 1e-3:
        zhalf = 0.5 * (zmax + zmin)
        fhalf = fraction_above_threshold(
            m1,
            m2,
            a1z,
            a2z,
            zhalf,
            thresh,
            fmin=fmin,
            dfmin=dfmin,
            fref=fref,
            psdstart=psdstart,
            psd_fn=psd_fn,
            approximant=approximant,
        )

        if fhalf > 0.0:
            zmin = zhalf
        else:
            zmax = zhalf

    vol_integral, _ = si.quad(integrand, zmin, zmax)  # This is the sensitive volume. out is tuple(integral, error)

    return analysis_time * vol_integral
