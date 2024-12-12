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


from typing_extensions import List

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float
from numpyro.distributions import CategoricalProbs, Distribution
from numpyro.distributions.mixtures import MixtureGeneral
from numpyro.distributions.util import validate_sample


class ScaledMixture(MixtureGeneral):
    r"""A finite mixture of component distributions from different families. This is
    a generalization of :class:`~numpyro.distributions.Mixture` where the component
    distributions are scaled by a set of rates.

    **Example**

    .. doctest::

       >>> import jax
       >>> import jax.random as jrd
       >>> import numpyro.distributions as dist
       >>> from gwkokab.models.utils import ScaledMixture
       >>> log_scales = jrd.uniform(jrd.PRNGKey(42), (3,), minval=0, maxval=5)
       >>> component_dists = [
       ...     dist.Normal(loc=0.0, scale=1.0),
       ...     dist.Normal(loc=-0.5, scale=0.3),
       ...     dist.Normal(loc=0.6, scale=1.2),
       ... ]
       >>> mixture = ScaledMixture(log_scales, component_dists)
       >>> mixture.sample(jax.random.PRNGKey(42)).shape
       ()
    """

    pytree_data_fields = ("_log_scales",)

    def __init__(
        self,
        log_scales: Float[Array, "..."],
        component_distributions: List[Distribution],
        *,
        support=None,
        validate_args=None,
    ):
        self._log_scales = log_scales
        mixing_distribution = CategoricalProbs(
            probs=jax.nn.softmax(log_scales), validate_args=validate_args
        )
        super(ScaledMixture, self).__init__(
            mixing_distribution=mixing_distribution,
            component_distributions=component_distributions,
            support=support,
            validate_args=validate_args,
        )

    def component_log_probs(self, value):
        # modified implementation of numpyro.distributions.MixtureGeneral.component_log_probs
        component_log_probs = []
        for d in self.component_distributions:
            log_prob = d.log_prob(value)
            if (self._support is not None) and (not d._validate_args):
                mask = d.support(value)
                log_prob = jnp.where(mask, log_prob, -jnp.inf)
            component_log_probs.append(log_prob)
        component_log_probs = jnp.stack(component_log_probs, axis=-1)
        return self._log_scales + component_log_probs

    @validate_sample
    def log_prob(
        self, value: Float[Array, "..."], intermediates=None
    ) -> Float[Array, "..."]:
        # https://github.com/pyro-ppl/numpyro/issues/1870
        del intermediates
        sum_log_probs = self.component_log_probs(value)
        safe_sum_log_probs = jnp.where(
            jnp.isneginf(sum_log_probs), -jnp.inf, sum_log_probs
        )
        return jax.nn.logsumexp(safe_sum_log_probs, axis=-1)
