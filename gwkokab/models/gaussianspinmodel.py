#  Copyright 2023 The GWKokab Authors
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

from __future__ import annotations

from numpyro.distributions import MultivariateNormal


def GaussianSpinModel(mu_eff: float, sigma_eff: float, mu_p: float, sigma_p: float, rho: float) -> MultivariateNormal:
    r"""Bivariate normal distribution for the effective and precessing spins.
    See Eq. (D3) and (D4) in `Population Properties of Compact Objects from
    the Second LIGO-Virgo Gravitational-Wave Transient Catalog
    <https://arxiv.org/abs/2010.14533>`__.

    .. math::
    
        \left(\chi_{\text{eff}}, \chi_{p}\right) \sim \mathcal{N}\left(
            \begin{bmatrix}
                \mu_{\text{eff}} \\ \mu_{p}
            \end{bmatrix},
            \begin{bmatrix}
                \sigma_{\text{eff}}^2 & \rho \sigma_{\text{eff}} \sigma_{p} \\
                \rho \sigma_{\text{eff}} \sigma_{p} & \sigma_{p}^2
            \end{bmatrix}
        \right)

    where :math:`\chi_{\text{eff}}` is the effective spin and :math:`\chi_{\text{eff}}\in[-1,1]` and
    :math:`\chi_{p}` is the precessing spin and :math:`\chi_{p}\in[0,1]`.

    :param mu_eff: mean of the effective spin
    :param sigma_eff: standard deviation of the effective spin
    :param mu_p: mean of the precessing spin
    :param sigma_p: standard deviation of the precessing spin
    :param rho: correlation coefficient between the effective and precessing spins
    :return: Multivariate normal distribution for the effective and precessing spins
    """
    return MultivariateNormal(
        loc=[mu_eff, mu_p],
        covariance_matrix=[
            [sigma_eff**2, rho * sigma_eff * sigma_p],
            [rho * sigma_eff * sigma_p, sigma_p**2],
        ],
        validate_args=True,
    )
