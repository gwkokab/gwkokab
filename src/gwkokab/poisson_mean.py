# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import List, Literal, Optional, Tuple, Union

import equinox as eqx
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger
from numpyro.distributions.distribution import Distribution, DistributionLike

from .models.utils import ScaledMixture
from .utils.tools import error_if
from .vts import RealInjectionVolumeTimeSensitivity, VolumeTimeSensitivityInterface


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

    is_injection_based: bool = eqx.field(init=False, default=False)
    """Flag to check if the class is injection based or not."""
    key: PRNGKeyArray = eqx.field(init=False)
    logVT_estimator: Optional[VolumeTimeSensitivityInterface] = eqx.field(
        init=False, default=None
    )
    num_samples_per_component: Optional[List[int]] = eqx.field(
        init=False, static=True, default=None
    )
    proposal_log_weights_and_samples: Tuple[Optional[Tuple[Array, Array]], ...] = (
        eqx.field(init=False)
    )
    time_scale: Union[int, float, Array] = eqx.field(init=False, default=1.0)

    def __init__(
        self,
        logVT_estimator: VolumeTimeSensitivityInterface,
        key: PRNGKeyArray,
        proposal_dists: Optional[List[Union[Literal["self"], DistributionLike]]] = None,
        num_samples: int = 10_000,
        self_num_samples: Optional[int] = None,
        num_samples_per_component: Optional[List[int]] = None,
        time_scale: Union[int, float, Array] = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        logVT_estimator : VolumeTimeSensitivityInterface
            Volume Time Sensitivity estimator.
        key : PRNGKeyArray
            PRNG key.
        proposal_dists : List[Union[Literal[&quot;self&quot;], DistributionLike]]
            List of proposal distributions. If "self" is given, the proposal distribution
            is the target distribution itself.
        num_samples : int
            Number of samples, by default 10_000
        self_num_samples : Optional[int], optional
            Number of samples for distribution using Inverse Transform Sampling, by default None
        num_samples_per_component : Optional[List[int]], optional
            Number of samples for each component, by default None
        time_scale : Union[int, float, Array]
            scale factor for time, by default 1.0

        Raises
        ------
        ValueError
            If the proposal distribution is unknown.
        ValueError
            If the proposal distribution is not a distribution.
        """
        if isinstance(logVT_estimator, RealInjectionVolumeTimeSensitivity):
            self.is_injection_based = True
            self.key = key
            self.proposal_log_weights_and_samples = (
                (jnp.log(logVT_estimator.sampling_prob), logVT_estimator.injections),
            )
            logger.warning(
                "The time scale is not used for injection based VTs. "
                "We parse the injection files to get the time scales."
            )
            self.time_scale = logVT_estimator.analysis_time_years
            self.num_samples_per_component = [logVT_estimator.total_injections]

        else:
            self.logVT_estimator = logVT_estimator
            self.__init_for_per_component_rate__(
                key=key,
                proposal_dists=proposal_dists,
                num_samples=num_samples,
                self_num_samples=self_num_samples,
                num_samples_per_component=num_samples_per_component,
                time_scale=time_scale,
            )

    def __init_for_per_component_rate__(
        self,
        key: PRNGKeyArray,
        proposal_dists: List[Union[Literal["self"], DistributionLike]],
        num_samples: int = 10_000,
        self_num_samples: Optional[int] = None,
        num_samples_per_component: Optional[List[int]] = None,
        time_scale: Union[int, float, Array] = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        key : PRNGKeyArray
            PRNG key.
        proposal_dists : List[Union[Literal[&quot;self&quot;], DistributionLike]]
            List of proposal distributions. If "self" is given, the proposal distribution
            is the target distribution itself.
        num_samples : int
            Number of samples, by default 10_000
        self_num_samples : Optional[int], optional
            Number of samples for distribution using Inverse Transform Sampling, by default None
        num_samples_per_component : Optional[List[int]], optional
            Number of samples for each component, by default None
        time_scale : Union[int, float, Array]
            scale factor for time, by default 1.0

        Raises
        ------
        ValueError
            If the proposal distribution is unknown.
        ValueError
            If the proposal distribution is not a distribution.
        """
        if num_samples_per_component is not None:
            error_if(
                len(proposal_dists) != len(num_samples_per_component),
                AssertionError,
                "Mismatch between the number of proposal distributions "
                f"({len(proposal_dists)}) and the number of samples per component "
                f"({len(num_samples_per_component)})",
            )
            self.num_samples_per_component = num_samples_per_component
        else:
            self.num_samples_per_component = [
                num_samples for _ in range(len(proposal_dists))
            ]
        self.time_scale = time_scale
        logVT_fn = self.logVT_estimator.get_mapped_logVT()  # type: ignore

        proposal_log_weights_and_samples = []
        for index, dist in enumerate(proposal_dists):
            if isinstance(dist, str):
                if dist.strip().lower() == "self":
                    proposal_log_weights_and_samples.append(None)
                    if self_num_samples is not None:
                        self.num_samples_per_component[index] = self_num_samples
                else:
                    raise ValueError(f"Unknown proposal distribution: {dist}")
            elif isinstance(dist, Distribution):
                _num_samples = self.num_samples_per_component[index]
                samples = dist.sample(key, (_num_samples,))
                proposal_log_prob: Array = dist.log_prob(samples).reshape(_num_samples)
                logVT_samples = logVT_fn(samples).reshape(_num_samples)
                log_weights = logVT_samples - proposal_log_prob
                error_if(
                    log_weights.shape != (_num_samples,),
                    AssertionError,
                    f"Expected log_weights to have shape {(_num_samples,)}, "
                    f"but got {log_weights.shape}",
                )
                error_if(
                    samples.shape[0] != _num_samples,
                    AssertionError,
                    f"Expected samples to have shape {(_num_samples, -1)}, "
                    f"but got {samples.shape}",
                )
                error_if(
                    not jnp.all(jnp.isfinite(log_weights)),
                    AssertionError,
                    f"Expected log_weights to be finite, but got {log_weights}",
                )
                error_if(
                    not jnp.all(jnp.isfinite(samples)),
                    AssertionError,
                    f"Expected samples to be finite, but got {samples}",
                )
                proposal_log_weights_and_samples.append((log_weights, samples))
                key, _ = jrd.split(key)
            else:
                raise ValueError(f"Unknown proposal distribution: {dist}")
        self.proposal_log_weights_and_samples = tuple(proposal_log_weights_and_samples)
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
        if self.is_injection_based:  # injection based sampling method
            # log w_i, ω_i
            log_weights, samples = self.proposal_log_weights_and_samples[0]  # type: ignore
            # n_total = n_accepted + n_rejected
            num_samples = self.num_samples_per_component[0]  # type: ignore
            # log p(ω_i|λ)
            model_log_prob = model.log_prob(samples).reshape(log_weights.shape[0])
            # log p(ω_i|λ) - log w_i
            log_prob = model_log_prob - log_weights
            # (T / n_total) * exp(log Σ exp(log p(ω_i|λ) - log w_i))
            return (self.time_scale / num_samples) * jnp.exp(
                jnn.logsumexp(log_prob, where=~jnp.isneginf(log_prob), axis=-1)
            )
        else:  # per component rate estimation
            return self.calculate_per_component_rate(model)

    def calculate_per_component_rate(self, model: ScaledMixture) -> Array:
        r"""Estimate the per component rate/s by using the given model.

        Parameters
        ----------
        model : ScaledMixture
            Model instance.

        Returns
        -------
        Array
            Estimated rate/s.
        """
        error_if(
            not isinstance(model, ScaledMixture),
            msg=f"Expected model to be an instance of ScaledMixture, but got {type(model)}",
        )
        error_if(
            len(self.proposal_log_weights_and_samples) != model.mixture_size,
            msg=f"Expected {model.mixture_size} proposal distributions, "
            f"but got {len(self.proposal_log_weights_and_samples)}",
        )

        per_component_log_estimated_rates = []

        logVT_fn = self.logVT_estimator.get_mapped_logVT()  # type: ignore

        for i in range(model.mixture_size):
            log_weights_and_samples = self.proposal_log_weights_and_samples[i]
            component_dist: DistributionLike = model.component_distributions[i]
            num_samples = self.num_samples_per_component[i]  # type: ignore
            if (
                log_weights_and_samples is None
            ):  # case 1: "self" meaning inverse transform sampling
                samples = component_dist.sample(self.key, (num_samples,))
                per_sample_log_estimated_rates = logVT_fn(samples)
            else:  # case 2: importance sampling
                log_weights, samples = log_weights_and_samples
                component_log_prob = component_dist.log_prob(samples).reshape(
                    num_samples
                )
                per_sample_log_estimated_rates = log_weights + component_log_prob
            per_component_log_estimated_rate = jnn.logsumexp(
                per_sample_log_estimated_rates,
                where=~jnp.isneginf(per_sample_log_estimated_rates),
            ) - jnp.log(num_samples)
            per_component_log_estimated_rates.append(per_component_log_estimated_rate)

        per_component_log_estimated_rates = model.log_scales + jnp.stack(
            per_component_log_estimated_rates, axis=-1
        )  # type: ignore
        per_component_estimated_rates = jnp.exp(per_component_log_estimated_rates)  # type: ignore
        return self.time_scale * jnp.sum(per_component_estimated_rates, axis=-1)
