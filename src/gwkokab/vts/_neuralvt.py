#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from collections.abc import Callable, Sequence

import equinox as eqx
import jax
from jax import lax, numpy as jnp
from jaxtyping import Array

from ._abc import VolumeTimeSensitivityInterface
from ._utils import load_model


class NeuralNetVolumeTimeSensitivity(VolumeTimeSensitivityInterface):
    neural_vt_model: eqx.nn.Sequential = eqx.field(init=False)
    """The neural volume-time sensitivity model."""

    def __init__(self, parameters: Sequence[str], filename: str) -> None:
        """Convenience class for loading a neural vt.

        Parameters
        ----------
        parameters : Sequence[str]
            The names of the parameters that the model expects.
        filename : str
            The filename of the neural vt.
        """
        if not parameters:
            raise ValueError("parameters sequence cannot be empty")
        if not isinstance(parameters, Sequence):
            raise TypeError(f"parameters must be a Sequence, got {type(parameters)}")
        if not all(isinstance(p, str) for p in parameters):
            raise TypeError("all parameters must be strings")

        names, self.neural_vt_model = load_model(filename)
        if any(name not in parameters for name in names):
            raise ValueError(
                f"Model in {filename} expects parameters {names}, but received "
                f"{parameters}. Missing: {set(names) - set(parameters)}"
            )
        self.shuffle_indices = [parameters.index(name) for name in names]

    def get_logVT(self) -> Callable[[Array], Array]:
        """Gets the logVT function."""

        @jax.jit
        def _logVT(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            return self.neural_vt_model(x_new)

        return _logVT

    def get_mapped_logVT(self) -> Callable[[Array], Array]:
        """Gets the vmapped logVT function for batch processing."""

        @jax.jit
        def _logVT(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            return jnp.squeeze(
                lax.map(self.neural_vt_model, x_new, batch_size=1000), axis=-1
            )

        return _logVT
