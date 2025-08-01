# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Sequence
from typing import Dict, Optional, Union

import equinox as eqx
import h5py
import jax
from jax import lax, numpy as jnp
from jaxtyping import Array

from ._utils import load_model


class NeuralNetProbabilityOfDetection(eqx.Module):
    shuffle_indices: Optional[Sequence[int]] = eqx.field(static=True, default=None)
    """The indices to shuffle the input to the model."""
    batch_size: Optional[int] = eqx.field(static=True, default=None)
    """The batch size used by :func:`jax.lax.map` in mapped functions."""
    neural_vt_model: eqx.nn.MLP
    """The neural volume-time sensitivity model."""
    parameter_ranges: Optional[Dict[str, Union[int, float]]]
    """Ranges of the parameters expected by the model."""

    def __init__(
        self,
        parameters: Sequence[str],
        filename: str,
        batch_size: Optional[int] = None,
    ) -> None:
        """Convenience class for loading a neural pdet.

        Parameters
        ----------
        parameters : Sequence[str]
            The names of the parameters that the model expects.
        filename : str
            The filename of the neural vt.
        batch_size : Optional[int], optional
            The batch size :func:`jax.lax.map` should use, by default None.
        """
        if not parameters:
            raise ValueError("parameters sequence cannot be empty")
        if not isinstance(parameters, Sequence):
            raise TypeError(f"parameters must be a Sequence, got {type(parameters)}")
        if not all(isinstance(p, str) for p in parameters):
            raise TypeError("all parameters must be strings")
        if batch_size is not None:
            if not isinstance(batch_size, int):
                raise TypeError(
                    f"batch_size must be an integer, got {type(batch_size)}"
                )
            if batch_size < 1:
                raise ValueError(
                    f"batch_size must be a positive integer, got {batch_size}"
                )

        self.batch_size = batch_size
        names, self.neural_vt_model = load_model(filename)
        if any(name not in parameters for name in names):
            raise ValueError(
                f"Model in {filename} expects parameters {names}, but received "
                f"{parameters}. Missing: {set(names) - set(parameters)}"
            )
        self.shuffle_indices = [parameters.index(name) for name in names]
        parameter_ranges = {}
        with h5py.File(filename, "r") as f:
            for k, v in f.attrs.items():
                if k.endswith("_min") or k.endswith("_max"):
                    parameter_ranges[k] = lax.stop_gradient(v)
        if len(parameter_ranges) > 0:
            self.parameter_ranges = parameter_ranges
        else:
            self.parameter_ranges = None

    def get_logVT(self) -> Callable[[Array], Array]:  # TODO: rename to get_pdet
        """Gets the logVT function."""

        @jax.jit
        def _pdet(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            y_new = self.neural_vt_model(x_new)
            mask = jnp.less_equal(y_new, 0.0)
            safe_y_new = jnp.where(mask, 1.0, y_new)
            return jnp.where(mask, -jnp.inf, jnp.log(safe_y_new))

        return _pdet

    def get_mapped_logVT(
        self,
    ) -> Callable[[Array], Array]:  # TODO: rename to get_mapped_pdet
        """Gets the vmapped logVT function for batch processing."""
        _batch_size = self.batch_size

        @jax.jit
        def _pdet(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            y_new = jnp.squeeze(
                lax.map(self.neural_vt_model, x_new, batch_size=_batch_size),
                axis=-1,
            )
            mask = jnp.less_equal(y_new, 0.0)
            safe_y_new = jnp.where(mask, 1.0, y_new)
            return jnp.where(mask, -jnp.inf, jnp.log(safe_y_new))

        return _pdet
