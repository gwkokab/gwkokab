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

from typing_extensions import Optional

from jax import lax, numpy as jnp
from jaxtyping import Array
from numpyro import distributions as dist
from numpyro.distributions.util import promote_shapes, validate_sample

from ..utils.misc import get_key
from .truncpowerlaw import TruncatedPowerLaw
from .utils.constraints import mass_sandwich


class Wysocki2019MassModel(dist.Distribution):
    r"""It is a double side truncated power law distribution, as described in
    equation 7 of the [Reconstructing phenomenological distributions of compact
    binaries via gravitational wave observations](https://arxiv.org/abs/1805.06442).

    $$
        p(m_1,m_2\mid\alpha,m_{\text{min}},m_{\text{max}},M_{\text{max}})\propto
        \frac{m_1^{-\alpha}}{m_1-m_{\text{min}}}
    $$
    """

    arg_constraints = {
        "alpha_m": dist.constraints.real,
        "mmin": dist.constraints.dependent,
        "mmax": dist.constraints.dependent,
    }
    reparametrized_params = ["alpha_m", "mmin", "mmax"]
    pytree_aux_fields = ("_support",)

    def __init__(self, alpha_m: float, mmin: float, mmax: float) -> None:
        r"""
        :param alpha_m: index of the power law distribution
        :param mmin: lower mass limit
        :param mmax: upper mass limit
        """
        self.alpha_m, self.mmin, self.mmax = promote_shapes(alpha_m, mmin, mmax)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(alpha_m),
            jnp.shape(mmin),
            jnp.shape(mmax),
        )
        self._support = mass_sandwich(mmin, mmax)
        super(Wysocki2019MassModel, self).__init__(
            batch_shape=batch_shape,
            event_shape=(2,),
            validate_args=True,
        )

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    @validate_sample
    def log_prob(self, value):
        m1 = value[..., 0]
        log_prob_m1 = TruncatedPowerLaw(
            alpha=-self.alpha_m,
            xmin=self.mmin,
            xmax=self.mmax,
        ).log_prob(m1)
        log_prob_m2_given_m1 = -jnp.log(m1 - self.mmin)
        return log_prob_m1 + log_prob_m2_given_m1

    def sample(self, key: Optional[Array | int], sample_shape: tuple = ()) -> Array:
        if key is None or isinstance(key, int):
            key = get_key(key)
        m2 = dist.Uniform(
            low=self.mmin,
            high=self.mmax,
        ).sample(key=key, sample_shape=sample_shape + self.batch_shape)
        key = get_key(key)
        m1 = TruncatedPowerLaw(
            alpha=-self.alpha_m,
            xmin=m2,
            xmax=self.mmax,
        ).sample(key=key, sample_shape=())
        return jnp.column_stack((m1, m2))
