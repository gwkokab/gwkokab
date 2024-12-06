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
from jax import numpy as jnp
from jaxtyping import Array

from ._utils import load_model
from ._vt import VolumeTimeSensitivityInterface


class NeuralNetVolumeTimeSensitivity(VolumeTimeSensitivityInterface):
    model: eqx.nn.Sequential = eqx.field(init=False)

    def __init__(self, parameters: Sequence[str], filename: str) -> None:
        """Convenience class for loading a neural vt.

        :param parameters: The names of the parameters that the model expects.
        :param filename: The filename of the neural vt.
        """
        names, self.model = load_model(filename)
        if any(name not in parameters for name in names):
            raise ValueError(
                f"{filename} only supports {names}. Requested {parameters}."
            )
        self.shuffle_indices = [parameters.index(name) for name in names]

    def get_logVT(self) -> Callable[[Array], Array]:
        """Gets the logVT function."""

        @jax.jit
        def _logVT(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            return self.model(x_new)

        return _logVT

    def get_vmapped_logVT(self) -> Callable[[Array], Array]:
        """Gets the vmapped logVT function for batch processing."""

        model_vmap = jax.vmap(self.model)

        @jax.jit
        def _logVT(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            return jnp.squeeze(model_vmap(x_new), axis=-1)

        return _logVT
