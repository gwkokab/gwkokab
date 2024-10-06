# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from functools import partial
from typing_extensions import Callable, List, Tuple

import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Int


def compute_ppd(
    logpdf: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    ranges: List[Tuple[Float[float, ""], Float[float, ""], Int[int, ""]]],
) -> Float[Array, "..."]:
    r"""Compute the posterior predictive distribution (PPD) of a model.

    :param logpdf: A callable that computes the log-probability density function of the model.
    :param ranges: A list of tuples containing the ranges of the parameters to evaluate the PPD.
    :return: The PPD of the model.
    """
    max_axis = int(np.argmax([n for _, _, n in ranges]))

    @partial(jax.vmap, in_axes=(max_axis,), out_axes=max_axis)
    def _ppd_vmapped(x: Float[Array, "..."]) -> Float[Array, "..."]:
        x = jnp.expand_dims(x, axis=-2)
        prob = jnp.exp(logpdf(x))
        ppd = jnp.mean(prob, axis=-1)
        return ppd

    xx = [jnp.linspace(a, b, n) for a, b, n in ranges]
    mesh = jnp.meshgrid(*xx, indexing="ij")
    xx_mesh = jnp.stack(mesh, axis=-1)
    ppd_vec = _ppd_vmapped(xx_mesh)
    return ppd_vec
