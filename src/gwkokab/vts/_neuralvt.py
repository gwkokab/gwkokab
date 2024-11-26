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

from ._utils import load_model, load_samples


class NeuralVT(eqx.Module):
    parameters: Sequence[str]
    filename: str
    model: eqx.nn.Sequential = eqx.field(init=False)
    shuffle_indices: Sequence[int] = eqx.field(init=False)

    def __post_init__(self):
        names, self.model = load_model(self.filename)
        if any(name not in self.parameters for name in names):
            raise ValueError(
                f"{self.filename} only support {names}. Requested {self.parameters}."
            )
        self.shuffle_indices = [self.parameters.index(name) for name in names]

    def get_logVT(self) -> Callable[[Array], Array]:
        """Gets the logVT function."""

        @jax.jit
        def _logVT(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            return self.model(x_new)

        return _logVT

    def get_vmapped_logVT(self) -> Callable[[Array], Array]:
        """Gets the logVT function."""

        @jax.jit
        def _logVT(x: Array) -> Array:
            x_new = x[..., self.shuffle_indices]
            return jnp.squeeze(jax.vmap(self.model)(x_new), axis=-1)

        return _logVT

    def get_samples(self) -> Array:
        _, samples = load_samples(self.filename)
        return samples[..., self.shuffle_indices]


NeuralVT.__init__.__doc__ = """Convenience class for loading a neural vt.

:param parameters: The names of the parameters that the model expects.
:param filename: The filename of the neural vt.
"""
