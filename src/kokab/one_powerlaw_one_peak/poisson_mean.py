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
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions import TransformedDistribution

from gwkokab.poisson_mean import PoissonMeanABC


class InverseTransformSamplingPoissonMean(PoissonMeanABC):
    r"""Samples are generated from :math:`\rho_{\Omega\mid\Lambda}` by using the inverse
    transform sampling method. The estimator is given by,

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
        Parameters
        ----------
        logVT_fn : Callable[[Array], Array]
            Log of the Volume Time Sensitivity function.
        key : PRNGKeyArray
            PRNG key.
        num_samples : int
            Number of samples
        scale : Union[int, float, Array]
            scale factor, by default 1.0
        """
        self.scale = scale
        self.key = key
        self.num_samples = num_samples
        self.logVT_fn = logVT_fn

    def __call__(self, model: TransformedDistribution) -> Array:
        if isinstance(model, TransformedDistribution):
            values = model.base_dist.sample(self.key, (self.num_samples,))
            log_rate = model.base_dist.log_rate
        else:
            values = model.sample(self.key, (self.num_samples,))
            log_rate = model.log_rate
        logVT_value = jnp.stack(
            [self.logVT_fn(values[:, i, :]) for i in range(2)], axis=-1
        )
        log_exp_rate_component = (
            log_rate + jnn.logsumexp(logVT_value, axis=0) - jnp.log(self.num_samples)
        )
        return self.scale * jnp.sum(jnp.exp(log_exp_rate_component), axis=-1)
