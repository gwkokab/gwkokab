# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import Optional

import equinox as eqx
import jax
from jax import lax, numpy as jnp
from jaxtyping import Array
from loguru import logger

from ..utils.tools import error_if
from ._abc import VolumeTimeSensitivityInterface
from ._utils import load_model


class NeuralNetVolumeTimeSensitivity(VolumeTimeSensitivityInterface):
    neural_vt_model: eqx.nn.MLP
    """The neural volume-time sensitivity model."""

    def __init__(
        self,
        parameters: Sequence[str],
        filename: str,
        batch_size: Optional[int] = None,
    ) -> None:
        """Convenience class for loading a neural vt.

        Parameters
        ----------
        parameters : Sequence[str]
            The names of the parameters that the model expects.
        filename : str
            The filename of the neural vt.
        batch_size : Optional[int], optional
            The batch size :func:`jax.lax.map` should use, by default None.
        """
        logger.debug(
            "Starting `NeuralNetVolumeTimeSensitivity` from {} with parameters {}",
            filename,
            parameters,
        )
        error_if(not parameters, msg="parameters sequence cannot be empty")
        error_if(
            not isinstance(parameters, Sequence),
            err=TypeError,
            msg=f"parameters must be a Sequence, got {type(parameters)}",
        )
        error_if(
            not all(isinstance(p, str) for p in parameters),
            err=TypeError,
            msg="all parameters must be strings",
        )
        if batch_size is not None:
            error_if(
                not isinstance(batch_size, int),
                err=TypeError,
                msg=f"batch_size must be an integer, got {type(batch_size)}",
            )
            error_if(
                batch_size < 1,
                msg=f"batch_size must be a positive integer, got {batch_size}",
            )

        self.batch_size = batch_size
        logger.debug("Batch size set to {}", self.batch_size)
        names, self.neural_vt_model = load_model(filename)
        error_if(
            any(name not in parameters for name in names),
            msg=f"Model in {filename} expects parameters {names}, but received "
            f"{parameters}. Missing: {set(names) - set(parameters)}",
        )
        self.shuffle_indices = [parameters.index(name) for name in names]
        logger.debug("Finished `NeuralNetVolumeTimeSensitivity`")

    def get_logVT(self) -> Callable[[Array], Array]:
        """Gets the logVT function."""

        @jax.jit
        def _logVT(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            return self.neural_vt_model(x_new)

        return _logVT

    def get_mapped_logVT(self) -> Callable[[Array], Array]:
        """Gets the vmapped logVT function for batch processing."""
        _batch_size = self.batch_size

        @jax.jit
        def _logVT(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            return jnp.squeeze(
                lax.map(self.neural_vt_model, x_new, batch_size=_batch_size), axis=-1
            )

        return _logVT
