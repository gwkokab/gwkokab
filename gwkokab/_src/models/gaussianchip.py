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

from jaxtyping import Float
from numpyro.distributions import TruncatedNormal


def GaussianChiP(mu: Float, sigma: Float) -> TruncatedNormal:
    r"""Truncated normal distribution for the precessing spin. See Eq. (3)-(4)
    in [The Low Effective Spin of Binary Black Holes and Implications for
    Individual Gravitational-Wave Events](https://arxiv.org/abs/2001.06051) and
    [Population Properties of Compact Objects from the Second LIGO-Virgo
    Gravitational-Wave Transient Catalog](https://arxiv.org/abs/2010.14533).

    $$
        p(\chi_{p}\mid\mu,\sigma)\propto\mathbb{I}_{[0,1]}(\chi_{p})
        \mathcal{N}(\chi_{p}\mid\mu,\sigma)
    $$

    where $\chi_{p}$ is the precessing spin and $\mathbb{I}(\cdot)$
    is the indicator function.

    :param mu: mean of the distribution
    :param sigma: standard deviation of the distribution
    :return: Truncated normal distribution for the precessing spin
    """
    return TruncatedNormal(mu, sigma, low=0.0, high=1.0, validate_args=True)
