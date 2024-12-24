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


from abc import abstractmethod
from typing import Union

import equinox as eqx
from jaxtyping import Array
from numpyro.distributions import Distribution

from ..models.utils import ScaledMixture


class PoissonMeanABC(eqx.Module):
    r"""Abstract class for Poisson mean functions. It is designed to be used with the
    `ScaledMixture` model. The `scale` attribute is a scaling factor for the mean
    function. Usually scale is analysis time. Mathematically it is defined as,

    .. math::

        \mu_{\Omega\mid\Lambda}
        = \mathbb{E}_{\Omega\mid\Lambda}\left[\operatorname{VT}(\Omega)\right]
        = \int_{\Omega} \operatorname{VT}(\omega) \rho_{\Omega\mid\Lambda}(\omega\mid\lambda) \, d\omega

    where :math:`\operatorname{VT}` is the Volume Time Sensitivity function, :math:`\Omega`
    is a vector of random variables which represent physical properties of the source like
    :math:`\Omega = \{m_1, m_2, \chi_1, \chi_2\}`, :math:`\Lambda` is the vector of
    hyperparameters of the model, :math:`\rho_{\Omega\mid\Lambda}` is the parameterized
    probability density function of the source properties given the hyperparameters.

    .. seealso::

        :class:`ImportanceSamplingPoissonMean`
        :class:`InverseTransformSamplingPoissonMean`
    """

    scale: Union[int, float, Array] = eqx.field(init=False, default=1.0, static=True)

    @abstractmethod
    def __call__(self, model: Distribution | ScaledMixture) -> Array:
        r"""Compute the mean of the Poisson distribution."""
        raise NotImplementedError("Abstract method")
