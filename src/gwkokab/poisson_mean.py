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
from typing import List, Literal, Optional, Tuple, Union

import equinox as eqx
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.distributions.distribution import Distribution, DistributionLike

from .logger import logger
from .models.utils import ScaledMixture


class PoissonMean(eqx.Module):
    r"""
    Inverse Transform Sampling
    --------------------------

    Samples are generated from :math:`\rho_{\Omega\mid\Lambda}` by using the inverse
    transform sampling method. The estimator is given by,

    .. math::

        \hat{\mu}_{\Omega\mid\Lambda} \approx \frac{1}{N}\sum_{i=1}^{N}\operatorname{VT}(\omega_i),
        \qquad \forall 0<i\leq N, \omega_i \sim \rho_{\Omega\mid\Lambda}.

    This method is very useful when the target distribution is easy to sample from.

    Importance Sampling
    -------------------

    It is very cheap to generate random samples from the proposal distribution and
    evaluate the importance weights and reusing them for rest of the calculations. We
    assume a proposal distribution :math:`\rho_{\phi}` and calculate the importance
    weights as,

    .. math::
        w_i = \frac{\operatorname{VT(\omega_i)}}{\rho_{\phi}(\omega_i)},
        \qquad \forall 0<i\leq N, \omega_i \sim \rho_{\phi}.

    We save these weights and samples for the later use and estimate
    :math:`\mu_{\Omega\mid\Lambda}` as,

    .. math::
        \hat{\mu}_{\Omega\mid\Lambda} \approx \frac{1}{N} \sum_{i=1}^{N}
        w_i \cdot \rho_{\Omega\mid\Lambda}(\omega_i\mid\lambda).

    Importance sampling is a variance reduction technique and it is very useful when the
    proposal distribution is close to the target distribution. Choice of a good proposal
    distribution is a tedious task and it is not always possible to find a good proposal
    distribution. To overcome this problem, we have used a mixture of proposal
    distributions to sample from. Usually, we use a uniform distribution and a
    log-uniform distribution over an appropriate range of the parameters. Some times we
    also add a peak at the maxima and at the expectation of the proposal distribution to
    improve the performance of the importance sampling.
    """

    logVT_fn: Callable[[Array], Array] = eqx.field(init=False)
    num_samples: int = eqx.field(init=False, static=True)
    key: PRNGKeyArray = eqx.field(init=False)
    proposal_log_weights_and_samples: List[Optional[Tuple[Array, Array]]] = eqx.field(
        init=False
    )
    scale: Union[int, float, Array] = eqx.field(init=False, default=1.0)

    def __init__(
        self,
        logVT_fn: Callable[[Array], Array],
        proposal_dists: List[Union[Literal["self"], DistributionLike]],
        key: PRNGKeyArray,
        num_samples: int,
        scale: Union[int, float, Array] = 1.0,
    ) -> None:
        r"""
        Parameters
        ----------
        logVT_fn : Callable[[Array], Array]
            Log of the Volume Time Sensitivity function.
        proposal_dists : List[Union[Literal[&quot;self&quot;], DistributionLike]]
            List of proposal distributions. If "self" is given, the proposal distribution
            is the target distribution itself.
        key : PRNGKeyArray
            PRNG key.
        num_samples : int
            Number of samples
        scale : Union[int, float, Array]
            scale factor, by default 1.0

        Raises
        ------
        ValueError
            If the proposal distribution is unknown.
        ValueError
            If the proposal distribution is not a distribution.
        """
        self.scale = scale
        self.num_samples = num_samples
        self.logVT_fn = logVT_fn

        proposal_log_weights_and_samples = []
        for dist in proposal_dists:
            if isinstance(dist, str):
                if dist.strip().lower() == "self":
                    proposal_log_weights_and_samples.append(None)
                else:
                    raise ValueError(f"Unknown proposal distribution: {dist}")
            elif isinstance(dist, Distribution):
                samples = dist.sample(key, (num_samples,))
                proposal_log_prob: Array = dist.log_prob(samples).reshape(num_samples)
                logVT_samples = logVT_fn(samples).reshape(num_samples)
                log_weights = logVT_samples - proposal_log_prob
                assert log_weights.shape == (num_samples,), (
                    f"Expected log_weights to have shape {(num_samples,)}, "
                    f"but got {log_weights.shape}"
                )
                assert samples.shape[0] == num_samples, (
                    f"Expected samples to have shape {(num_samples, -1)}, "
                    f"but got {samples.shape}"
                )
                assert jnp.all(jnp.isfinite(log_weights)), (
                    f"Expected log_weights to be finite, but got {log_weights}"
                )
                assert jnp.all(jnp.isfinite(samples)), (
                    f"Expected samples to be finite, but got {samples}"
                )
                proposal_log_weights_and_samples.append((log_weights, samples))
                key, _ = jrd.split(key)
            else:
                raise ValueError(f"Unknown proposal distribution: {dist}")
        self.proposal_log_weights_and_samples = proposal_log_weights_and_samples
        self.key = key

    def __call__(self, model: ScaledMixture) -> Array:
        r"""Estimate the rate/s by using the given model.

        Parameters
        ----------
        model : ScaledMixture
            Model instance.

        Returns
        -------
        Array
            Estimated rate/s.
        """
        assert isinstance(model, ScaledMixture), (
            f"Expected model to be an instance of ScaledMixture, but got {type(model)}"
        )
        assert len(self.proposal_log_weights_and_samples) == model.mixture_size, (
            f"Expected {model.mixture_size} proposal distributions, "
            f"but got {len(self.proposal_log_weights_and_samples)}"
        )

        per_component_per_sample_estimated_rates = []

        for i in range(model.mixture_size):
            log_weights_and_samples = self.proposal_log_weights_and_samples[i]
            component_dist: DistributionLike = model._component_distributions[i]
            if (
                log_weights_and_samples is None
            ):  # case 1: "self" meaning inverse transform sampling
                samples = component_dist.sample(self.key, (self.num_samples,))
                per_component_per_sample_estimated_rates.append(self.logVT_fn(samples))
            else:  # case 2: importance sampling
                log_weights, samples = log_weights_and_samples
                proposal_log_prob = component_dist.log_prob(samples).reshape(
                    self.num_samples
                )
                per_component_per_sample_estimated_rates.append(
                    log_weights + proposal_log_prob
                )

        per_component_per_sample_estimated_rates = jnp.stack(
            per_component_per_sample_estimated_rates, axis=-1
        )
        per_component_estimated_rates = (
            model._log_scales
            + jnn.logsumexp(per_component_per_sample_estimated_rates, axis=0)
            - jnp.log(self.num_samples)
        )
        logger.debug(
            "per_component_estimated_rates: {per_component_estimated_rates}",
            per_component_estimated_rates=per_component_estimated_rates,
        )
        return self.scale * jnp.sum(jnp.exp(per_component_estimated_rates), axis=-1)
