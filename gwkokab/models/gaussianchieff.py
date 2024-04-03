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

from numpyro.distributions import TruncatedNormal


def GaussianChiEff(mu: float, sigma: float, *, validate_args=None) -> TruncatedNormal:
    r"""Truncated normal distribution for the effective spin. See Eq. (3)-(4) in
    `The Low Effective Spin of Binary Black Holes and Implications for Individual
    Gravitational-Wave Events <https://arxiv.org/abs/2001.06051>`__ and
    `Population Properties of Compact Objects from the Second LIGO-Virgo Gravitational-Wave Transient Catalog
    <https://arxiv.org/abs/2010.14533>`__

    .. math::
        \displaystyle p(\chi_{\text{eff}}\mid\mu,\sigma)=
        \frac{\displaystyle\sqrt{\frac{2}{\pi\sigma^2}}}{\displaystyle\operatorname{erf}\left(\frac{1-\mu}{\sqrt{2}\sigma}\right)
        +\operatorname{erf}\left(\frac{1+\mu}{\sqrt{2}\sigma}\right)}
        \exp{\left(-\frac{1}{2}\left(\frac{\chi_{\text{eff}}-\mu}{\sigma}\right)^2\right)}

    where :math:`\chi_{\text{eff}}` is the effective spin and :math:`\chi_{\text{eff}}\in[-1,1]`.

    :param mu: mean of the distribution
    :param sigma: standard deviation of the distribution
    :return: Truncated normal distribution for the effective spin
    """
    return TruncatedNormal(
        mu,
        sigma,
        low=-1.0,
        high=1.0,
        validate_args=validate_args,
    )
