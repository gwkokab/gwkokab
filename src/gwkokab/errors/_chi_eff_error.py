# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax.numpy as jnp
import jax.random as jrd
import RIFT.lalsimutils as lalsimutils
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import TruncatedNormal

from ..utils.transformations import (
    chi_a,
    chi_eff,
    chirp_mass,
    delta_m,
    mass_ratio,
    symmetric_mass_ratio,
)


def _psi_coefficient(chi1z: Array, chi2z: Array, m1: Array, m2: Array) -> Array:
    r"""Calculate the :math:`\psi`-coefficient of the 1.5 PN phase term. See equation
    (A2) of `Gravitational-wave astrophysics with effective-spin measurements:
    asymmetries and selection biases <http://arxiv.org/abs/1805.03046>`_

    .. math::
        \psi = \eta^{-\frac{3}{5}}\left(\frac{1}{128} \left[\left(113 - 76 \eta \right) \chi_{\text{eff}} + 76 \delta \eta \chi_a\right] - \frac{3 \pi}{8}\right)

    where, :math:`\eta`, :math:`\chi_{\text{eff}}`, :math:`\delta`, and :math:`\chi_a`
    are defined in :func:`~gwkokab.utils.transformations.symmetric_mass_ratio`,
    :func:`~gwkokab.utils.transformations.chi_eff`,
    :func:`~gwkokab.utils.transformations.delta_m`, and
    :func:`~gwkokab.utils.transformations.chi_a` respectively.

    Parameters
    ----------
    chi1z : Array
        Projection of the primary spin onto the z-axis.
    chi2z : Array
        Projection of the secondary spin onto the z-axis.
    m1 : Array
        Primary mass.
    m2 : Array
        Secondary mass.

    Returns
    -------
    Array
        Coefficient of the 1.5 PN phase term.
    """
    _q = mass_ratio(m1=m1, m2=m2)
    η = symmetric_mass_ratio(q=_q)
    δ = delta_m(m1, m2)
    χ_a = chi_a(chi1z, chi2z)
    χ_eff = chi_eff(chi1z=chi1z, chi2z=chi2z, q=_q)
    return jnp.power(η, -0.6) * (
        ((((113.0 - 76.0 * η) * χ_eff) + (76.0 * δ * η * χ_a)) / 128.0)
        - (0.375 * jnp.pi)
    )


def _psi_from_chi_eff_and_eta(chi_eff: Array, eta: Array) -> Array:
    r"""Calculate :math:`\psi` from :math:`\chi_{\text{eff}}` and :math:`\eta` with the
    assumption of :math:`\chi_{2z} = 0`.
    """
    _q = mass_ratio(eta=eta)
    _δ_η_χ_a = chi_eff * eta * (1 - _q)
    return jnp.power(eta, -0.6) * (
        ((((113.0 - 76.0 * eta) * chi_eff) + (76.0 * _δ_η_χ_a)) / 128.0)
        - (0.375 * jnp.pi)
    )


def _chi_eff_from_psi_and_eta(psi: Array, eta: Array) -> Array:
    r"""Calculation of :math:`\chi_{\text{eff}}` from :math:`\psi` and :math:`\eta`, by
    rearranging equation (A2) of `Gravitational-wave astrophysics with effective-spin
    measurements: asymmetries and selection biases <http://arxiv.org/abs/1805.03046>`_.
    Assuming :math:`\chi_{2z} = 0`.

    Parameters
    ----------
    psi : Array
        :math:`\psi`-coefficient of the 1.5 PN phase term.
    eta : Array
        Symmetric mass ratio.
    Returns
    -------
    Array
        Calculation of :math:`\chi_{\text{eff}}` from :math:`\psi` and :math:`\eta`.
    """
    q = mass_ratio(eta=eta)
    A = 128.0 / (76.0 * eta)
    B = psi * jnp.power(eta, 0.6) + (3.0 * jnp.pi / 8.0)
    C = (113 / (76 * eta) - 1) + ((1 - q) * 0.5)
    return A * (B / C)


def _chi_eff_approximate_prior_prob(chi_eff: Array) -> Array:
    r"""Approximate prior for :math:`\chi_{\text{eff}}`, from equation B7 of
    `Gravitational-wave astrophysics with effective-spin measurements: asymmetries and
    selection biases <http://arxiv.org/abs/1805.03046>`_.

    .. math::
        p(\chi_{\text{eff}}) = \frac{\displaystyle 1-\exp{\left(-\frac{|\chi_{\text{eff}}|-\chi_{*}}{w}\right)}}{\displaystyle 2\chi_{*} + 2w\left(1-\exp{\left(\frac{\chi_{*}}{w}\right)}\right)}

    where, :math:`\chi_{*}` is the maximum value of :math:`\chi_{\text{eff}}`, it is set
    to 1, and :math:`w=0.23` is a value found to fit well to all priors in a study done
    in cited paper.

    Parameters
    ----------
    chi_eff : Array
        The effective spin.

    Returns
    -------
    Array
        Approximate prior for :math:`\chi_{\text{eff}}`.
    """
    χ_max = 1
    w = 0.23  # value found to fit well to all priors by the cited paper in docstring
    A = 1 - jnp.exp(-(jnp.abs(chi_eff) - χ_max) / w)
    B = 2 * χ_max
    C = 2 * w * (1 - jnp.exp(χ_max / w))
    return A / (B + C)


def chi_eff_from_m1_m2_chi1_chi2(
    x: Array,
    size: int,
    key: PRNGKeyArray,
    *,
    scale_eta: Array = 1.0,
    scale_chi_eff: Array = 1.0,
) -> Array:
    m1 = x[..., 0]
    m2 = x[..., 1]
    a1z = x[..., 2]
    a2z = x[..., 3]

    Mc_true = chirp_mass(m1=m1, m2=m2)
    η_true = symmetric_mass_ratio(m1=m1, m2=m2)
    χ_true = chi_eff(chi1z=a1z, chi2z=a2z, m1=m1, m2=m2)

    keys = jrd.split(key, 3)

    ρ = 9.0 * jnp.power(jrd.uniform(key=keys[0]), -1.0 / 3.0)

    v_PN_param = (jnp.pi * Mc_true * 20 * lalsimutils.MsunInSec) ** (
        1.0 / 3.0
    )  # 'v' parameter
    v_PN_param_max = 0.2
    v_PN_param = jnp.min(jnp.array([v_PN_param, v_PN_param_max]))
    snr_fac = ρ / 12.0
    # this ignores range due to redshift / distance, based on a low-order est
    ln_mc_error_pseudo_fisher = (
        1.5 * 0.3 * (v_PN_param / v_PN_param_max) ** (7.0) / snr_fac
    )

    β = jnp.min(jnp.array([0.07 / snr_fac, ln_mc_error_pseudo_fisher]))

    scale_eta *= β
    scale_chi_eff *= β

    # make a gaussian in ψ, then convert to χeff
    ψ_mean = _psi_from_chi_eff_and_eta(χ_true, η_true)

    # estimate sigma on psi from sigma on χeff.
    # χeff maximum range is -1 to 1: 2. psi maximum range is -4.2 to -1.2: 3.0.
    # scale χeff uncertainties by 3.0 / 2.
    ψ_sigma = scale_chi_eff * 3.0 / 2.0

    eta_samps = TruncatedNormal(
        loc=η_true, scale=scale_eta, low=0.0, high=0.25, validate_args=True
    ).sample(key=keys[1], sample_shape=(size,))

    psi_samps = TruncatedNormal(
        loc=ψ_mean, scale=ψ_sigma, low=-4.2, high=-1.2, validate_args=True
    ).sample(key=keys[2], sample_shape=(size,))

    χ_eff_samps = _chi_eff_from_psi_and_eta(psi_samps, eta_samps)

    return χ_eff_samps
