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
import numpy as np
from jaxtyping import Array
from scipy import interpolate

from ..utils.transformations import (
    chi_a,
    chi_eff,
    delta_m,
    mass_ratio,
    symmetric_mass_ratio,
)


def psi_coefficient(chi1z: Array, chi2z: Array, m1: Array, m2: Array) -> Array:
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

    :param chi1z: Projection of the primary spin onto the z-axis.
    :param chi2z: Projection of the secondary spin onto the z-axis.
    :param m1: Primary mass.
    :param m2: Secondary mass.
    :return: Coefficient of the 1.5 PN phase term.
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


def psi_from_chi_eff_and_eta(chi_eff: Array, eta: Array) -> Array:
    r"""Calculate :math:`\psi` from :math:`\chi_{\text{eff}}` and :math:`\eta` with
    the assumption of :math:`\chi_{2z} = 0`."""
    _q = mass_ratio(eta=eta)
    _δ_η_χ_a = chi_eff * eta * (1 - _q)
    return jnp.power(eta, -0.6) * (
        ((((113.0 - 76.0 * eta) * chi_eff) + (76.0 * _δ_η_χ_a)) / 128.0)
        - (0.375 * jnp.pi)
    )


def chi_eff_from_psi_and_eta(psi: Array, eta: Array) -> Array:
    r"""Calculation of :math:`\chi_{\text{eff}}` from :math:`\psi` and :math:`\eta`, by
    rearranging equation (A2) of `Gravitational-wave astrophysics with effective-spin
    measurements: asymmetries and selection biases <http://arxiv.org/abs/1805.03046>`_.
    Assuming :math:`\chi_{2z} = 0`.

    :param psi: :math:`\psi`-coefficient of the 1.5 PN phase term.
    :param eta: Symmetric mass ratio.
    :return: Calculation of :math:`\chi_{\text{eff}}` from :math:`\psi` and :math:`\eta`.
    """
    q = mass_ratio(eta=eta)
    A = 128.0 / (76.0 * eta)
    B = psi * jnp.power(eta, 0.6) + (3.0 * jnp.pi / 8.0)
    C = (113 / (76 * eta) - 1) + ((1 - q) * 0.5)
    return A * (B / C)


def chi_eff_approximate_prior_prob(chi_eff: Array) -> Array:
    r"""Approximate prior for :math:`\chi_{\text{eff}}`, from equation B7 of
    `Gravitational-wave astrophysics with effective-spin measurements: asymmetries and
    selection biases <http://arxiv.org/abs/1805.03046>`_.

    .. math::
        p(\chi_{\text{eff}}) = \frac{\displaystyle 1-\exp{\left(-\frac{|\chi_{\text{eff}}|-\chi_{*}}{w}\right)}}{\displaystyle 2\chi_{*} + 2w\left(1-\exp{\left(\frac{\chi_{*}}{w}\right)}\right)}

    where, :math:`\chi_{*}` is the maximum value of :math:`\chi_{\text{eff}}`, it is set
    to 1, and :math:`w=0.23` is a value found to fit well to all priors in a study done
    in cited paper.

    :param chi_eff: The effective spin.
    :return: Approximate prior for :math:`\chi_{\text{eff}}`.
    """
    χ_max = 1
    w = 0.23  # value found to fit well to all priors by the cited paper in docstring
    A = 1 - jnp.exp(-(jnp.abs(chi_eff) - χ_max) / w)
    B = 2 * χ_max
    C = 2 * w * (1 - jnp.exp(χ_max / w))
    return A / (B + C)


def chi_eff_draw_prob(a1z_samps, q_samps):
    Xeff_samps = a1z_samps / (1 + q_samps)
    # 2d histogram
    H, Xeffedges, qedges = np.histogram2d(Xeff_samps, q_samps, bins=100, density=True)
    # 2D histogram density normalises as bin_count / sample_count / bin_area
    # so multiply by bin_area below, so that sum(H)=1
    H = H * (Xeffedges[1] - Xeffedges[0]) * (qedges[1] - qedges[0])
    # interpolate
    Xeffpts = Xeffedges[0:-1] + ((Xeffedges[1] - Xeffedges[0]) / 2)  # bin midpoints
    qpts = qedges[0:-1] + ((qedges[1] - qedges[0]) / 2)
    func = interpolate.RectBivariateSpline(Xeffpts, qpts, H)
    # return function
    return func
