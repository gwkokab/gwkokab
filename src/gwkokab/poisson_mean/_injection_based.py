# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import List, Optional, Tuple

import equinox as eqx
import h5py
import jax
import numpy as np
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from loguru import logger

from ..constants import SECONDS_PER_YEAR
from ..models.utils import ScaledMixture
from ..parameters import Parameters as P
from ..utils.tools import batch_and_remainder, error_if, warn_if


_PARAM_MAPPING = {
    P.PRIMARY_MASS_SOURCE.value: "mass1_source",
    P.PRIMARY_SPIN_X.value: "spin1x",
    P.PRIMARY_SPIN_Y.value: "spin1y",
    P.PRIMARY_SPIN_Z.value: "spin1z",
    P.REDSHIFT.value: "redshift",
    P.SECONDARY_MASS_SOURCE.value: "mass2_source",
    P.SECONDARY_SPIN_X.value: "spin2x",
    P.SECONDARY_SPIN_Y.value: "spin2y",
    P.SECONDARY_SPIN_Z.value: "spin2z",
}


def load_o1o2o3_or_endO_injection_data(
    filename: str,
    parameters: Sequence[str],
    far_cut: float = 1.0,
    snr_cut: float = 10.0,
    ifar_pipelines: Sequence[str] | None = None,
):
    if P.ECCENTRICITY.value in parameters:
        _PARAM_MAPPING[P.ECCENTRICITY.value] = P.ECCENTRICITY.value
        warn_if(
            True,
            msg="Eccentricity injections are not part of O1, O2 or O3 injections. "
            f"Make sure you have altered the injection set with {P.ECCENTRICITY.value} "
            "key accordingly.",
        )
    with h5py.File(filename, "r") as f:
        analysis_time_years = float(f.attrs["analysis_time_s"]) / SECONDS_PER_YEAR
        logger.debug("Analysis time: {:.2f} years", analysis_time_years)

        total_injections = int(f.attrs["total_generated"])
        logger.debug("Total injections: {}", total_injections)

        injections = f["injections"]

        if ifar_pipelines is None:
            ifar_pipelines = [k for k in injections.keys() if "ifar" in k]
            logger.debug(
                "No pipelines specified for ifar, using all available: {}",
                ", ".join(ifar_pipelines),
            )
        else:
            logger.debug("Selecting ifar from pipelines: {}", ", ".join(ifar_pipelines))

        ifar = np.max([injections[k][:] for k in ifar_pipelines], axis=0)
        snr = injections["optimal_snr_net"][:]

        runs = injections["name"][:].astype(str)
        found = np.where(runs == "o3", ifar > 1 / far_cut, snr > snr_cut)

        n_total = found.shape[0]
        n_found = np.sum(found)
        logger.debug(
            "Found {} out of {} injections with FAR < {} and SNR > {}",
            n_found,
            n_total,
            far_cut,
            snr_cut,
        )

        sampling_prob = (
            injections["sampling_pdf"][found][:]
            / injections["mixture_weight"][found][:]
        )

        χ_1x = injections["spin1x"][found][:]
        χ_1y = injections["spin1y"][found][:]
        χ_1z = injections["spin1z"][found][:]
        χ_2x = injections["spin2x"][found][:]
        χ_2y = injections["spin2y"][found][:]
        χ_2z = injections["spin2z"][found][:]
        a1 = np.sqrt(np.square(χ_1x) + np.square(χ_1y) + np.square(χ_1z))
        a2 = np.sqrt(np.square(χ_2x) + np.square(χ_2y) + np.square(χ_2z))

        injs = []
        for p in parameters:
            if p == P.COS_TILT_1.value:
                _inj = χ_1z / a1
            elif p == P.COS_TILT_2.value:
                _inj = χ_2z / a2
            elif p == P.PRIMARY_SPIN_MAGNITUDE.value:
                _inj = a1
                # We parameterize spins in spherical coordinates, neglecting azimuthal
                # P. The injections are parameterized in terms of cartesian
                # spins. The Jacobian is `1 / (2 pi magnitude ** 2)`.
                logger.debug(
                    "Correcting sampling probability for spherical spin "
                    "parameterization of primary spin."
                )
                sampling_prob *= 2.0 * np.pi * np.square(a1)
            elif p == P.SECONDARY_SPIN_MAGNITUDE.value:
                _inj = a2
                # We parameterize spins in spherical coordinates, neglecting azimuthal
                # P. The injections are parameterized in terms of cartesian
                # spins. The Jacobian is `1 / (2 pi magnitude ** 2)`.
                logger.debug(
                    "Correcting sampling probability for spherical spin "
                    "parameterization of secondary spin."
                )
                sampling_prob *= 2.0 * np.pi * np.square(a2)
            else:
                _inj = injections[_PARAM_MAPPING[p]][found][:]
            injs.append(_inj)

        if (
            P.PRIMARY_SPIN_MAGNITUDE.value not in parameters
            and P.SECONDARY_SPIN_MAGNITUDE.value not in parameters
        ):
            # Eliminating the probability of cartesian spins
            logger.debug(
                "Eliminating the probability of cartesian spins from sampling probability."
            )
            sampling_prob *= np.square(4.0 * np.pi * a1 * a2)

        injections = jax.device_put(np.stack(injs, axis=-1))

        sampling_log_prob = jax.device_put(np.log(sampling_prob))

    return injections, sampling_log_prob, analysis_time_years, total_injections


def load_o1o2o3o4a_injection_data(
    filename: str,
    parameters: Sequence[str],
    far_cut: float = 1.0,
    snr_cut: float = 10.0,
):
    expected_params = {
        P.COS_TILT_1.value,
        P.COS_TILT_2.value,
        P.PRIMARY_MASS_SOURCE.value,
        P.PRIMARY_SPIN_MAGNITUDE.value,
        P.REDSHIFT.value,
        P.SECONDARY_MASS_SOURCE.value,
        P.SECONDARY_SPIN_MAGNITUDE.value,
    }
    if P.ECCENTRICITY.value in parameters:
        _PARAM_MAPPING[P.ECCENTRICITY.value] = P.ECCENTRICITY.value
        expected_params.add(P.ECCENTRICITY.value)
        warn_if(
            True,
            msg="Eccentricity injections are not part of O1, O2, O3, or O4a injections. "
            f"Make sure you have altered the injection set with {P.ECCENTRICITY.value} "
            "key accordingly.",
        )

    error_if(
        set(parameters) != expected_params,
        msg=(
            "For O1, O2, O3, and O4a injections, the following parameters must be used: "
            + ", ".join(_PARAM_MAPPING.keys())
        ),
    )

    with h5py.File(filename, "r") as f:
        raw_events = f["events"]
        keys: List[str] = list(raw_events.dtype.names)

        analysis_time_years = float(f.attrs["total_analysis_time"]) / SECONDS_PER_YEAR
        logger.debug("Analysis time: {:.2f} years", analysis_time_years)

        total_injections = int(f.attrs["total_generated"])
        logger.debug("Total injections: {}", total_injections)

        far_pipelines = [
            k
            for k in keys
            if any([k.startswith("far_"), k.endswith("_far"), "_far_" in k])
        ]

        ifar_pipelines = [
            k
            for k in keys
            if any([k.startswith("ifar_"), k.endswith("_ifar"), "_ifar_" in k])
        ]

        if len(far_pipelines) > 0:
            logger.debug("Selecting far from pipelines: {}", ", ".join(far_pipelines))
        if len(ifar_pipelines) > 0:
            logger.debug("Selecting ifar from pipelines: {}", ", ".join(ifar_pipelines))

        ifar = np.max(
            [1.0 / raw_events[k][:] for k in far_pipelines]
            + [raw_events[k][:] for k in ifar_pipelines],
            axis=0,
        )
        snr = raw_events["semianalytic_observed_phase_maximized_snr_net"][:]

        found = (snr > snr_cut) | (ifar > 1 / far_cut)

        n_total = int(f.attrs["num_accepted"])
        n_found = np.sum(found)
        logger.debug(
            "Found {} out of {} injections with FAR < {} and SNR > {}",
            n_found,
            n_total,
            far_cut,
            snr_cut,
        )

        χ_1x = raw_events["spin1x"][found][:]
        χ_1y = raw_events["spin1y"][found][:]
        χ_1z = raw_events["spin1z"][found][:]
        χ_2x = raw_events["spin2x"][found][:]
        χ_2y = raw_events["spin2y"][found][:]
        χ_2z = raw_events["spin2z"][found][:]
        a1 = np.sqrt(np.square(χ_1x) + np.square(χ_1y) + np.square(χ_1z))
        a2 = np.sqrt(np.square(χ_2x) + np.square(χ_2y) + np.square(χ_2z))

        sampling_log_prob = (
            raw_events[
                "lnpdraw_mass1_source_mass2_source_redshift_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z"
            ][found][:]
            - np.log(raw_events["weights"][found][:])
            # We parameterize spins in spherical coordinates, neglecting azimuthal
            # P. The injections are parameterized in terms of cartesian
            # spins. The Jacobian is `1 / (2 pi magnitude ** 2)`.
            + 2.0 * (np.log(2.0 * np.pi) + np.log(a1) + np.log(a2))
        )

        injs = []
        for p in parameters:
            if p == P.COS_TILT_1.value:
                _inj = χ_1z / a1
            elif p == P.COS_TILT_2.value:
                _inj = χ_2z / a2
            elif p == P.PRIMARY_SPIN_MAGNITUDE.value:
                _inj = a1
            elif p == P.SECONDARY_SPIN_MAGNITUDE.value:
                _inj = a2
            else:
                _inj = raw_events[_PARAM_MAPPING[p]][found][:]
            injs.append(_inj)

        events = jax.device_put(np.stack(injs, axis=-1))

        sampling_log_prob = jax.device_put(sampling_log_prob)

    return events, sampling_log_prob, analysis_time_years, total_injections


def poisson_mean_from_sensitivity_injections(
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
    del key  # Unused.

    with h5py.File(filename, "r") as f:
        is_o1o2o3o4a = "events" in f

    if is_o1o2o3o4a:
        del ifar_pipelines  # Unused.
        # θ_i, log w_i, T, N_total
        samples, log_weights, analysis_time_years, total_injections = (
            load_o1o2o3o4a_injection_data(
                filename,
                parameters,
                far_cut,
                snr_cut,
            )
        )
    else:
        # θ_i, log w_i, T, N_total
        samples, log_weights, analysis_time_years, total_injections = (
            load_o1o2o3_or_endO_injection_data(
                filename,
                parameters,
                far_cut,
                snr_cut,
                ifar_pipelines,
            )
        )

    n_accepted = log_weights.shape[0]

    def _poisson_mean(scaled_mixture: ScaledMixture) -> Array:
        log_prob_fn = eqx.filter_jit(eqx.filter_vmap(scaled_mixture.log_prob))

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

    @eqx.filter_jit
    def _variance_of_estimator(scaled_mixture: ScaledMixture) -> Array:
        """See equation 9 and 11 of https://arxiv.org/abs/2406.16813."""
        log_prob_fn = eqx.filter_jit(eqx.filter_vmap(scaled_mixture.log_prob))

        def _f(
            carry: Tuple[Array, Array], data: Tuple[Array, Array]
        ) -> Tuple[Tuple[Array, Array], None]:
            carry_logsumexp, carry_logsumexp2 = carry
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
            partial_logsumexp2 = jnn.logsumexp(
                2.0 * safe_log_prob,
                where=~jnp.isneginf(safe_log_prob),
                axis=-1,
            )

            safe_carry_logsumexp = jnp.where(
                jnp.isneginf(carry_logsumexp) | jnp.isnan(carry_logsumexp),
                -jnp.inf,
                carry_logsumexp,
            )
            safe_carry_logsumexp2 = jnp.where(
                jnp.isneginf(carry_logsumexp2) | jnp.isnan(carry_logsumexp2),
                -jnp.inf,
                carry_logsumexp2,
            )

            return (
                jnp.logaddexp(safe_carry_logsumexp, partial_logsumexp),
                jnp.logaddexp(safe_carry_logsumexp2, partial_logsumexp2),
            ), None

        initial_logprob = jnp.asarray(-jnp.inf)
        if batch_size is None or n_accepted <= batch_size:
            # If the number of accepted injections is less than or equal to the batch size,
            # we can process them all at once.
            (log_prob, log_prob2), _ = _f(
                (initial_logprob, initial_logprob), (log_weights, samples)
            )
        else:
            batched_log_weights, remainder_log_weights = batch_and_remainder(
                log_weights, batch_size
            )
            batched_samples, remainder_samples = batch_and_remainder(
                samples, batch_size
            )
            (batched_logprob, batched_logprob2), _ = jax.lax.scan(
                _f,
                (initial_logprob, initial_logprob),
                (batched_log_weights, batched_samples),
            )
            (log_prob, log_prob2), _ = _f(
                (batched_logprob, batched_logprob2),
                (remainder_log_weights, remainder_samples),
            )

        term2 = jnp.exp(
            2.0 * jnp.log(analysis_time_years)
            - 3.0 * jnp.log(total_injections)
            + 2.0 * log_prob
        )
        term1 = jnp.exp(
            2.0 * jnp.log(analysis_time_years)
            - 2.0 * jnp.log(total_injections)
            + log_prob2
        )
        return term1 - term2

    return None, _poisson_mean, analysis_time_years, _variance_of_estimator
