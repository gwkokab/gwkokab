# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import numpy as np
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from numpyro._typing import DistributionT

from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean._injection_based_helper import (
    apply_injection_prior,
    load_injection_data,
)
from gwkokab.utils.tools import batch_and_remainder


def custom_poisson_mean_estimator(
    key: PRNGKeyArray,
    parameters: Sequence[str],
    filename: str,
    batch_size: Optional[int] = None,
    far_cut: float = 1.0,
    snr_cut: float = 10.0,
    ifar_pipelines: Sequence[str] | None = None,
) -> Tuple[
    Optional[Callable[[Array], Array]],
    Callable[[ScaledMixture], Array],
    float | Array,
    Callable[[ScaledMixture], Array],
]:
    """Custom injection-based Poisson mean estimator which ignores eccentricity."""
    del key  # Unused.

    parameters_without_ecc = list(
        filter(lambda p: p != P.ECCENTRICITY.value, parameters)
    )

    injections_dict = load_injection_data(filename, 1.0 / far_cut, snr_cut)

    _PARAM_MAPPING = {
        "mass_1": P.PRIMARY_MASS_SOURCE.value,
        "mass_2": P.SECONDARY_MASS_SOURCE.value,
        "mass1_source": P.PRIMARY_MASS_SOURCE.value,
        "mass2_source": P.SECONDARY_MASS_SOURCE.value,
        "redshift": P.REDSHIFT.value,
        "spin1x": P.PRIMARY_SPIN_X.value,
        "spin1y": P.PRIMARY_SPIN_Y.value,
        "spin1z": P.PRIMARY_SPIN_Z.value,
        "spin2x": P.SECONDARY_SPIN_X.value,
        "spin2y": P.SECONDARY_SPIN_Y.value,
        "spin2z": P.SECONDARY_SPIN_Z.value,
        "z": P.REDSHIFT.value,
    }

    injections_dict = {_PARAM_MAPPING.get(k, k): v for k, v in injections_dict.items()}
    injections_dict = apply_injection_prior(injections_dict, parameters_without_ecc)

    samples = jnp.stack(
        [jnp.asarray(injections_dict[param]) for param in parameters_without_ecc],
        axis=-1,
    )
    log_weights = np.log(injections_dict["prior"])
    analysis_time_years = injections_dict["analysis_time"]
    total_injections = injections_dict["total_generated"]
    n_accepted = samples.shape[0]

    def _poisson_mean(scaled_mixture: ScaledMixture) -> Array:
        scaled_mixture_without_ecc: List[DistributionT] = list(
            scaled_mixture.component_distributions[0].marginal_distributions
        )
        # eccentricity is the second last distribution in both powerlawpeak and
        # bp2pfull models
        scaled_mixture_without_ecc.pop(-2)
        new_scaled_mixture = ScaledMixture(
            log_scales=scaled_mixture.log_scales,
            component_distributions=[
                JointDistribution(
                    *scaled_mixture_without_ecc,
                    validate_args=scaled_mixture.component_distributions[
                        0
                    ]._validate_args,
                )
            ],
            validate_args=scaled_mixture._validate_args,
        )
        log_prob_fn = eqx.filter_jit(eqx.filter_vmap(new_scaled_mixture.log_prob))

        def _f(carry_logsumexp: Array, data: Tuple[Array, Array]) -> Tuple[Array, None]:
            log_weights, samples = data
            # log p(θ_i|λ)
            model_log_prob = log_prob_fn(samples).reshape(log_weights.shape[0])
            # log p(θ_i|λ) - log w_i
            log_prob = model_log_prob - log_weights
            safe_log_prob = jnp.where(
                jnp.isneginf(log_prob) | jnp.isnan(log_prob),
                -jnp.inf,
                log_prob,
            )

            partial_logsumexp = jnn.logsumexp(
                safe_log_prob,
                where=~jnp.isneginf(safe_log_prob),
                axis=-1,
            )
            safe_carry_logsumexp = jnp.where(
                jnp.isneginf(carry_logsumexp) | jnp.isnan(carry_logsumexp),
                -jnp.inf,
                carry_logsumexp,
            )
            return jnp.logaddexp(safe_carry_logsumexp, partial_logsumexp), None

        initial_logprob = jnp.asarray(-jnp.inf)
        if batch_size is None or n_accepted <= batch_size:
            # If the number of accepted injections is less than or equal to the batch size,
            # we can process them all at once.
            log_prob, _ = _f(initial_logprob, (log_weights, samples))
        else:
            batched_log_weights, remainder_log_weights = batch_and_remainder(
                log_weights, batch_size
            )
            batched_samples, remainder_samples = batch_and_remainder(
                samples, batch_size
            )
            batched_logprob, _ = jax.lax.scan(
                _f,
                initial_logprob,
                (batched_log_weights, batched_samples),
            )
            log_prob, _ = _f(
                batched_logprob, (remainder_log_weights, remainder_samples)
            )

        # (T / n_total) * exp(log Σ exp(log p(θ_i|λ) - log w_i))
        return (analysis_time_years / total_injections) * jnp.exp(log_prob)

    def _variance_of_estimator(scaled_mixture: ScaledMixture) -> Array:  # place holder
        return jnp.array(0.0)

    return None, _poisson_mean, analysis_time_years, _variance_of_estimator
