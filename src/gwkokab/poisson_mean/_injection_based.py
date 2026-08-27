# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Poisson mean from a found-injection campaign.

Rather than modelling the sensitivity, this estimator reweights the recovered
injections of a search campaign. Each found injection carries the density it was
drawn from, so importance sampling gives

.. math::
    \mu(\Lambda) = \frac{T_{\text{obs}}}{N_{\text{total}}}
    \sum_{i \in \text{found}} \frac{p(\omega_i \mid \Lambda)}{\pi(\omega_i)},

with :math:`N_{\text{total}}` the number of injections *generated*, not found -- the
ones that were missed contribute zero. The draw density :math:`\pi` is reconstructed
from the injection file and converted into the analysis coordinates by
:func:`~gwkokab.poisson_mean._injection_based_helper.apply_injection_prior`.
"""

from collections.abc import Callable
from typing import Any, List, Optional, Tuple

import jax
import numpy as np
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from loguru import logger

from ..models.utils import ScaledMixture
from ..parameters import Parameters as P
from ._injection_based_helper import apply_injection_prior, load_injection_data


def poisson_mean_from_sensitivity_injections(
    key: PRNGKeyArray,
    parameters: List[str],
    filename: str,
    batch_size: Optional[int] = None,
    far_cut: float = 1.0,
    snr_cut: float = 10.0,
) -> Tuple[
    Optional[Callable[[Array], Array]],
    Callable[..., Array],
    dict[str, Any],
]:
    """Build a Poisson mean estimator from a sensitivity injection campaign.

    The injection file is read once at build time: injections are filtered by inverse
    false alarm rate and SNR, their column names are mapped onto
    :class:`~gwkokab.parameters.Parameters`, and their draw density is transformed into
    the coordinates named by ``parameters``. Everything that does not depend on the
    population is therefore precomputed.

    Parameters
    ----------
    key : PRNGKeyArray
        Unused; accepted so every estimator factory shares one signature.
    parameters : List[str]
        Names of the event coordinates, in the order the population model expects them.
    filename : str
        Path to the HDF5 injection file.
    batch_size : Optional[int], optional
        Chunk size for evaluating the population density over the injections. Defaults
        to :data:`None`, meaning no chunking.
    far_cut : float, optional
        False alarm rate threshold, in inverse years; an injection is found if its FAR is
        below this. Defaults to ``1.0``.
    snr_cut : float, optional
        SNR threshold, used for the observing runs that report no FAR. Defaults to
        ``10.0``.

    Returns
    -------
    Tuple[Optional[Callable[[Array], Array]], Callable[..., Array], dict[str, Any]]
        A triple of:

        - the log sensitivity function, exposed so it can be plotted or reused (or
          :data:`None` when the estimator has no such function);
        - the estimator itself, which maps a
          :class:`~gwkokab.models.utils.ScaledMixture` and the extra arguments below to
          ``(mean, variance)``;
        - a dictionary of those extra arguments, to be splatted into the estimator at
          call time.

        Here the first element is :data:`None`, since the sensitivity is tabulated rather
        than functional, and the extra arguments carry the injection samples, their log
        draw densities and the campaign's analysis time.

    Raises
    ------
    ValueError
        If no injection passes the thresholds, or the file's format cannot be identified.

    Notes
    -----
    The Monte Carlo variance is returned alongside the mean so the sampler can detect
    when the estimate is too noisy to trust; see the ``--variance-cut-threshold`` flag
    and Equations 9 and 11 of `arXiv:2406.16813 <https://arxiv.org/abs/2406.16813>`_.
    """
    del key  # Unused.

    injections_dict = load_injection_data(filename, 1.0 / far_cut, snr_cut)

    _PARAM_MAPPING = {
        "mass_1": P.PRIMARY_MASS_SOURCE,
        "mass_2": P.SECONDARY_MASS_SOURCE,
        "mass1_source": P.PRIMARY_MASS_SOURCE,
        "mass2_source": P.SECONDARY_MASS_SOURCE,
        "redshift": P.REDSHIFT,
        "spin1x": P.PRIMARY_SPIN_X,
        "spin1y": P.PRIMARY_SPIN_Y,
        "spin1z": P.PRIMARY_SPIN_Z,
        "spin2x": P.SECONDARY_SPIN_X,
        "spin2y": P.SECONDARY_SPIN_Y,
        "spin2z": P.SECONDARY_SPIN_Z,
        "z": P.REDSHIFT,
    }

    injections_dict = {_PARAM_MAPPING.get(k, k): v for k, v in injections_dict.items()}
    injections_dict = apply_injection_prior(injections_dict, parameters)

    samples = jnp.stack(
        [jnp.asarray(injections_dict[param]) for param in parameters],
        axis=-1,
    )
    log_weights = np.log(injections_dict["prior"])
    analysis_time_years = injections_dict["analysis_time"]
    total_injections = injections_dict["total_generated"]

    logger.debug("Analysis time (years): {}", analysis_time_years)
    logger.debug(
        "Found {} out of {} injections with FAR < {} and SNR > {}",
        samples.shape[0],
        total_injections,
        far_cut,
        snr_cut,
    )

    def _poisson_mean(
        scaled_mixture: ScaledMixture, samples: Array, log_weights: Array, T_obs: float
    ) -> tuple[Array, Array]:
        r"""Estimate the Poisson mean and its Monte Carlo variance by importance
        sampling.

        Injections where the population density vanishes contribute :math:`-\infty` to the
        log sum and are masked out of the reduction, so they cost nothing rather than
        poisoning the gradient.

        Parameters
        ----------
        scaled_mixture : ScaledMixture
            The population model, evaluated at each found injection.
        samples : Array
            The found injections, with parameters along the last axis.
        log_weights : Array
            Log draw density :math:`\log\pi` of each found injection.
        T_obs : float
            Analysis time of the campaign, in years.

        Returns
        -------
        tuple[Array, Array]
            The expected number of detections and the variance of that estimate.
        """
        model_log_prob = jax.lax.map(
            scaled_mixture.log_prob,
            samples,
            batch_size=batch_size,
        )

        log_prob = model_log_prob - log_weights

        safe_log_prob = jnp.where(
            jnp.isneginf(log_prob) | jnp.isnan(log_prob),
            -jnp.inf,
            log_prob,
        )

        logsumexp_log_prob = jnn.logsumexp(
            safe_log_prob,
            where=~jnp.isneginf(safe_log_prob),
            axis=-1,
        )

        logsumexp_log_prob2 = jnn.logsumexp(
            2.0 * safe_log_prob,
            where=~jnp.isneginf(safe_log_prob),
            axis=-1,
        )

        term2 = jnp.exp(
            2.0 * jnp.log(T_obs)
            - 3.0 * jnp.log(total_injections)
            + 2.0 * logsumexp_log_prob
        )
        term1 = jnp.exp(
            2.0 * jnp.log(T_obs) - 2.0 * jnp.log(total_injections) + logsumexp_log_prob2
        )
        # See equation 9 and 11 of https://arxiv.org/abs/2406.16813
        variance = term1 - term2

        # (T / n_total) * exp(log Σ exp(log p(θ_i|λ) - log w_i))
        mean = (T_obs * jnp.exp(logsumexp_log_prob)) / total_injections

        return mean, variance

    return (
        None,
        _poisson_mean,
        {
            "samples": samples,
            "log_weights": log_weights,
            "T_obs": analysis_time_years,
        },
    )
