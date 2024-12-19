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


from collections.abc import Callable
from typing import Union

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ._abc import PoissonMeanABC


class InverseTransformSamplingPoissonMean(PoissonMeanABC):
    r"""Samples are generated from :math:`\rho_{\Omega\mid\Lambda}` by using the
    inverse transform sampling method. The estimator is given by,

    .. math::

        \hat{\mu}_{\Omega\mid\Lambda} \approx \frac{1}{N}\sum_{i=1}^{N}\operatorname{VT}(\omega_i),
        \qquad \forall 0<i\leq N, \omega_i \sim \rho_{\Omega\mid\Lambda}.

    This method is very useful when the target distribution is easy to sample from.
    """

    logVT_fn: Callable[[Array], Array] = eqx.field(init=False)
    num_samples: int = eqx.field(init=False, static=True)
    key: PRNGKeyArray = eqx.field(init=False)

    def __init__(
        self,
        logVT_fn: Callable[[Array], Array],
        key: PRNGKeyArray,
        num_samples: int,
        scale: Union[int, float, Array] = 1.0,
    ) -> None:
        r"""
        :param logVT_fn: Log of the Volume Time Sensitivity function.
        :param key: PRNG key.
        :param num_samples: Number of samples
        :param scale: scale factor, defaults to 1.0
        """
        self.scale = scale
        self.key = key
        self.num_samples = num_samples
        self.logVT_fn = jax.vmap(lambda xx: jnp.mean(jnp.exp(logVT_fn(xx))), in_axes=1)

    def __call__(self, model):
        values = model.component_sample(self.key, (self.num_samples,))
        VT = self.logVT_fn(values)
        rates = jnp.exp(model._log_scales)
        return self.scale * jnp.dot(VT, rates)
