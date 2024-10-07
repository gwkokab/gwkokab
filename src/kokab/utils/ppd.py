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

import h5py
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Int


def compute_ppd(
    logpdf: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    ranges: List[Tuple[Float[float, ""], Float[float, ""], Int[int, ""]]],
) -> Float[Array, "..."]:
    r"""Compute the posterior predictive distribution (PPD) of a model.

    The function evaluates the PPD over a grid defined by the provided parameter ranges.

    :param logpdf: A callable that computes the log-probability density function of the model.
    :param ranges: A list of tuples `(start, end, num_points)` for each parameter, defining the grid over which to compute the PPD.
    :return: The PPD of the model as a multidimensional array corresponding to the parameter grid.
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


def save_ppd(
    ppd_array: Float[Array, "..."],
    filename: str,
    ranges: List[Tuple[Float[float, ""], Float[float, ""], Int[int, ""]]],
    headers: List[str],
) -> None:
    assert ppd_array.ndim == len(
        ranges
    ), "Number of ranges must match the number of dimensions of the PPD array."
    assert ppd_array.ndim == len(
        headers
    ), "Number of headers must match the number of dimensions of the PPD array."

    with h5py.File(filename, "w") as f:
        f.create_dataset("range", data=np.array(ranges))
        f.create_dataset("headers", data=np.array(headers, dtype="S"))
        f.create_dataset("ppd", data=ppd_array)
