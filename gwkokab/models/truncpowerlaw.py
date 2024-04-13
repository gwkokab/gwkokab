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

from functools import partial
from typing_extensions import Optional

from jax import jit, lax, numpy as jnp
from jax.random import uniform
from jaxtyping import Array
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample

from ..typing import Numeric
from ..utils.misc import get_key


class TruncatedPowerLaw(dist.Distribution):
    r"""A generic double side truncated power law distribution.
    
    .. warning::
        There are many different definition of Power Law that include exponential cut-offs and
        interval cut-offs.  They are just interchangeably. This class is the implementation of
        power law that has been restricted over a closed interval.

    .. math::
        
        p(x\mid\alpha, x_{\text{min}}, x_{\text{max}}):=
        \begin{cases}
            \displaystyle\frac{x^{\alpha}}{\mathcal{Z}} & 0<x_{\text{min}}\leq x\leq x_{\text{max}}\\
            0                                           & \text{otherwise}
        \end{cases}

    where :math:`\mathcal{Z}` is the normalization constant and :math:`\alpha` is the power law index.
    :math:`x_{\text{min}}` and :math:`x_{\text{max}}` are the lower and upper truncation limits,
    respectively. The normalization constant is given by,
    
    .. math::
    
        \mathcal{Z}:=\begin{cases}
            \log{x_{\text{max}}}-\log{x_{\text{min}}}                                         & \alpha = -1 \\
            \displaystyle\frac{x_{\text{max}}^{1+\alpha}-x_{\text{min}}^{1+\alpha}}{1+\alpha} & \text{otherwise}
        \end{cases}
    """

    arg_constraints = {
        "alpha": dist.constraints.real,
        "xmin": dist.constraints.positive,
        "xmax": dist.constraints.positive,
    }

    def __init__(self, alpha: float, xmin: float, xmax: float, *, validate_args=None):
        self.alpha, self.xmin, self.xmax = promote_shapes(alpha, xmin, xmax)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha),
            jnp.shape(xmin),
            jnp.shape(xmax),
        )
        super(TruncatedPowerLaw, self).__init__(
            batch_shape=batch_shape,
            validate_args=validate_args,
        )
        self.support = dist.constraints.interval(self.xmin, self.xmax)
        self._log_Z = self.log_Z()

    @partial(jit, static_argnums=(0,))
    def log_Z(self) -> Numeric:
        """Computes the logarithm of normalization constant.

        :return: The logarithm of normalization constant.
        """
        return jnp.where(
            self.alpha == -1.0,
            jnp.log(jnp.log(self.xmax) - jnp.log(self.xmin)),
            jnp.log(jnp.abs(jnp.power(self.xmax, 1.0 + self.alpha) - jnp.power(self.xmin, 1.0 + self.alpha)))
            - jnp.log(jnp.abs(1.0 + self.alpha)),
        )

    @validate_sample
    def log_prob(self, value: Numeric) -> Numeric:
        return self.alpha * jnp.log(value) - self._log_Z

    def sample(self, key: Optional[Array | int], sample_shape: tuple = ()):
        if key is None or isinstance(key, int):
            key = get_key(key)
        U = uniform(key, sample_shape + self.batch_shape)
        return jnp.where(
            self.alpha == -1.0,
            jnp.exp(jnp.log(self.xmin) + U * (jnp.log(self.xmax) - jnp.log(self.xmin))),
            jnp.power(
                jnp.power(self.xmin, 1.0 + self.alpha)
                + U * (jnp.power(self.xmax, 1.0 + self.alpha) - jnp.power(self.xmin, 1.0 + self.alpha)),
                jnp.reciprocal(1.0 + self.alpha),
            ),
        )
