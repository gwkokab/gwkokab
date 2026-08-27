# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Poisson mean from a neural sensitive volume-time regressor.

The expected number of detections is estimated by Monte Carlo over samples drawn
from the population model itself:

.. math::
    \mu(\Lambda) = T_{\text{obs}} \sum_k \mathcal{R}_k
    \left\langle VT(\omega) \right\rangle_{\omega \sim p_k},

with the average taken over ``num_samples`` draws from each mixture component. The
:math:`VT` surface is supplied by a network trained with
:func:`gwkokab.utils.train.train_regressor`.
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional, Tuple, Union

import jax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from gwkokab.utils.exceptions import LoggedTypeError, LoggedValueError

from ..models.utils import ScaledMixture
from ..utils.train import load_model


def poisson_mean_from_neural_vt(
    key: PRNGKeyArray,
    parameters: Sequence[str],
    filename: str,
    batch_size: Optional[int] = None,
    num_samples: int = 1_000,
    time_scale: Union[int, float, Array] = 1.0,
) -> Tuple[
    Optional[Callable[[Array], Array]],
    Callable[..., Array],
    dict[str, Any],
]:
    r"""Build a Poisson mean estimator backed by a neural :math:`VT` regressor.

    The network is loaded from ``filename`` and its inputs are reordered to match
    ``parameters``, so the caller's parameter order need not agree with the order the
    network was trained in.

    Parameters
    ----------
    key : PRNGKeyArray
        JAX random key, fixed at build time so the Monte Carlo draws are the same on
        every call and the estimator stays a deterministic function of the population.
    parameters : Sequence[str]
        Names of the event coordinates, in the order the population model produces them.
        Must be a superset of the network's inputs.
    filename : str
        Path to the HDF5 file holding the trained network.
    batch_size : Optional[int], optional
        Chunk size for evaluating the network. Defaults to :data:`None`, meaning no
        chunking.
    num_samples : int, optional
        Number of Monte Carlo samples drawn per component. Defaults to ``1000``.
    time_scale : Union[int, float, Array], optional
        Observing time :math:`T_{\text{obs}}`, in the same units the rates are expressed
        in. Defaults to ``1.0``.

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

    Raises
    ------
    LoggedValueError
        If ``parameters`` is empty, ``batch_size`` is not positive, or the network
        expects a parameter that ``parameters`` does not supply.
    LoggedTypeError
        If ``parameters`` is not a sequence of strings, or ``batch_size`` is not an
        integer.

    Notes
    -----
    The Monte Carlo variance is returned alongside the mean so the sampler can detect
    when the estimate is too noisy to trust; see the ``--variance-cut-threshold`` flag
    and Equations 9 and 11 of `arXiv:2406.16813 <https://arxiv.org/abs/2406.16813>`_.
    """
    if not parameters:
        raise LoggedValueError("parameters sequence cannot be empty")
    if not isinstance(parameters, Sequence):
        raise LoggedTypeError(f"parameters must be a Sequence, got {type(parameters)}")
    if not all(isinstance(p, str) for p in parameters):
        raise LoggedTypeError("all parameters must be strings")
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise LoggedTypeError(
                f"batch_size must be an integer, got {type(batch_size)}"
            )
        if batch_size < 1:
            raise LoggedValueError(
                f"batch_size must be a positive integer, got {batch_size}"
            )

    names, neural_vt_model = load_model(filename)
    if any(name not in parameters for name in names):
        raise LoggedValueError(
            f"Model in {filename} expects parameters {names}, but received "
            f"{parameters}. Missing: {set(names) - set(parameters)}"
        )

    shuffle_indices = [parameters.index(name) for name in names]

    @jax.jit
    def log_vt(x: Array) -> Array:
        r"""Evaluate the log sensitive volume-time at a set of event coordinates.

        Parameters
        ----------
        x : Array
            Event coordinates, with the parameters along the last axis in the order given by
            ``parameters``. They are reordered internally to the network's input order.

        Returns
        -------
        Array
            :math:`\log VT`, with the trailing singleton output axis squeezed away.
        """
        x_new = x[..., shuffle_indices]
        return jnp.squeeze(
            jax.lax.map(neural_vt_model, x_new, batch_size=batch_size), axis=-1
        )

    def _poisson_mean(
        scaled_mixture: ScaledMixture, T_obs: Array
    ) -> Tuple[Array, Array]:
        """Estimate the Poisson mean and its Monte Carlo variance.

        Parameters
        ----------
        scaled_mixture : ScaledMixture
            The population model; its log scales supply the component rates.
        T_obs : Array
            Observing time.

        Returns
        -------
        Tuple[Array, Array]
            The expected number of detections and the variance of that estimate.
        """
        component_sample = scaled_mixture.component_sample(key, (num_samples,))
        # vmapping over components
        log_vt_values = jax.vmap(log_vt, in_axes=1)(component_sample)
        mean_per_component = jnp.exp(
            scaled_mixture.log_scales + jax.nn.logsumexp(log_vt_values, axis=-1)
        )
        mean = (T_obs / num_samples) * jnp.sum(mean_per_component, axis=-1)

        term2 = jnp.exp(
            2.0 * jnp.log(T_obs)
            - 3.0 * jnp.log(num_samples)
            + 2.0 * scaled_mixture.log_scales
            + 2.0 * jax.nn.logsumexp(log_vt_values, axis=-1)
        )
        term1 = jnp.exp(
            2.0 * jnp.log(T_obs)
            - 2.0 * jnp.log(num_samples)
            + 2.0 * scaled_mixture.log_scales
            + jax.nn.logsumexp(2.0 * log_vt_values, axis=-1)
        )
        variance = jnp.sum(term1 - term2, axis=-1)
        return mean, variance

    return (
        log_vt,
        _poisson_mean,
        {"T_obs": time_scale},
    )
