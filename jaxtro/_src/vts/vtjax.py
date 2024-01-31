from jax.numpy import pi,cos,sin,square,sqrt,linspace,asarray
from jax.random import uniform
from jax import random
import lalsimulation as ls
import lal
import astropy.units as u
import astropy.cosmology as cosmo
import scipy.integrate as si # Optimizeed integral needed

def next_pow_two(x):
    """Return the next (integer) power of two above `x`.
    """

    x2 = 1
    while x2 < x:
        x2 = x2 << 1
    return x2

def optimal_snr(
        m1, m2, a1z, a2z, z,
        fmin=19.0, dfmin=0.0, fref=40.0, psdstart=20.0,
        psd_fn=None,
        approximant=None,
    ):
    """Return the optimal SNR of a signal.

    :param m1: The source-frame mass 1.

    :param m2: The source-frame mass 2.

    :param a1z: The z-component of spin 1.

    :param a2z: The z-component of spin 2.

    :param z: The redshift.

    :param fmin: The starting frequency for waveform generation.

    :param psd_fn: A function that returns the detector PSD at a given
    frequency, you can choose any given in lalsimulation.
    https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_noise_p_s_d__c.html#ga02d7e9443530dbd0957dff1a08a0ab3c

    :return: The SNR of a face-on, overhead source.

    """

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDAdVO4T1800545 #psd of single detector for O4

    if approximant is None:
        approximant = ls.IMRPhenomB #alligned spin approximant
    #https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_i_m_r_phenom__c.html#ga9117e5a155732b9922eab602930377d7
    
    # Get dL, Gpc
    dL = cosmo.Planck15.luminosity_distance(z).to(u.Gpc).value

    tmax = ls.SimInspiralChirpTimeBound(
        fmin,
        m1*(1+z)*lal.MSUN_SI, m2*(1+z)*lal.MSUN_SI,
        a1z, a2z,
    ) + 2.0

    df = max(1.0/next_pow_two(tmax), dfmin)
    fmax = 2048.0 # Hz --- based on max freq of 5-5 inspiral
    
    # Compute the GW strain. g(t) = hp*Fps + hc*Fxs
    hp, hc = ls.SimInspiralChooseFDWaveform(
        (1+z)*m1*lal.MSUN_SI, (1+z)*m2*lal.MSUN_SI,
        0.0, 0.0, a1z, 0.0, 0.0, a2z,
        dL*1e9*lal.PC_SI,
        0.0, 0.0, 0.0, 0.0, 0.0,
        df, fmin, fmax, fref, None, approximant,
    )
     
    Nf = int(round(fmax/df)) + 1
    fs = linspace(0, fmax, Nf)

    # PSD is in units of 1/Hz
    # sffs is the frequency series of the PSD
    sffs = lal.CreateREAL8FrequencySeries(
        "psds", 0, 0.0, df, lal.DimensionlessUnit, fs.shape[0],
    )
    psd_fn(sffs, psdstart)
    return ls.MeasureSNRFD(hp, sffs, psdstart, -1.0)

#print(optimal_snr(10,10,0.5,0.5,0.1))

# Computing p_{det}.
def fraction_above_threshold( 
        m1, m2, a1z, a2z, z,
        snr_thresh,
        fmin=19.0, dfmin=0.0, fref=40.0, psdstart=20.0,
        psd_fn=None, approximant=None,
    ):

    global _thetas

    if psd_fn is None:
        psd_fn = ls.SimNoisePSDAdVO4T1800545

    if z == 0.0:
        return 1.0

    rho_max = optimal_snr(
        m1, m2, a1z, a2z, z,
        fmin=fmin, dfmin=dfmin, fref=fref, psdstart=psdstart,
        psd_fn=psd_fn,
        approximant=approximant,
    )

    a2, a4, a8 = 0.374222, 2.04216, -2.63948 
    w = (asarray(snr_thresh) / rho_max)[()]
    if w > 1: # no detection
        return 0.0
    else:
        P_det = a2*((1-w)**2)+a4*((1-w)**4)+a8*((1-w)**8)+(1-a2-a4-a8)*((1-w)**(10)) #slightly higher than Dan's value
        return P_det
#print(fraction_above_threshold(30,30,0.5,0.5,0.1,10)) 

# Computing VT
def vt_from_mass_spin(
        m1, m2, a1z, a2z,
        thresh,
        analysis_time,
        fmin=19.0, dfmin=0.0, fref=40.0, psdstart=20.0,
        zmax=1.5,
        psd_fn=None,
        approximant=None,
    ):
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

    def integrand(z):
        if z == 0.0:
            return 0.0
        else:
            return (
                4*pi *
                cosmo.Planck15.differential_comoving_volume(z)
                  .to(u.Gpc**3 / u.sr).value /
                (1+z) *
                fraction_above_threshold(
                    m1, m2, a1z, a2z, z,
                    thresh,
                    fmin=fmin, dfmin=dfmin, fref=fref, psdstart=psdstart,
                    psd_fn=psd_fn,
                    approximant=approximant,
                )
            )

    zmin = 0.001
    assert fraction_above_threshold(
          m1, m2, a1z, a2z, zmax,
          thresh,
          fmin=fmin, dfmin=dfmin, fref=fref, psdstart=psdstart,
          psd_fn=psd_fn,
          approximant=approximant,
      ) == 0.0
  
    while zmax - zmin > 1e-3:
        zhalf = 0.5*(zmax+zmin)
        fhalf = fraction_above_threshold(
            m1, m2, a1z, a2z, zhalf,
            thresh,
            fmin=fmin, dfmin=dfmin, fref=fref, psdstart=psdstart,
            psd_fn=psd_fn,
            approximant=approximant,
        )

        if fhalf > 0.0:
            zmin=zhalf
        else:
            zmax=zhalf

    vol_integral = si.quad(integrand, zmin, zmax)[0] # this is the sensitive volume. out is tupel(integral, error)
    
    return analysis_time * vol_integral

#print(vt_from_mass_spin(30,20,0.4,0.5,8,2))

# Computing VT for a given set of masses and spins.

