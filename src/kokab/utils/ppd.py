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


from functools import partial
from typing_extensions import Callable, List, Tuple

import h5py
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array


def compute_probs(
    logpdf: Callable[[Array], Array],
    ranges: List[Tuple[float, float, int]],
) -> Array:
    r"""Compute the probability density function of a model.

    :param logpdf: A callable that computes the log-probability density function of
        the model.
    :param ranges: A list of tuples `(start, end, num_points)` for each parameter,
        defining the grid over which to compute the PPD.
    :return: The PPD of the model as a multidimensional array corresponding to the
        parameter grid.
    """
    max_axis = int(np.argmax([int(n) for _, _, n in ranges]))

    @partial(jax.vmap, in_axes=(max_axis,), out_axes=max_axis)
    def _prob_vmapped(x: Array) -> Array:
        x_expanded = jnp.expand_dims(x, axis=-2)
        prob = jnp.exp(logpdf(x_expanded))
        return prob

    xx = [jnp.linspace(a, b, int(n)) for a, b, n in ranges]
    mesh = jnp.meshgrid(*xx, indexing="ij")
    xx_mesh = jnp.stack(mesh, axis=-1)
    prob_vec = _prob_vmapped(xx_mesh)
    return prob_vec


def _compute_marginal_probs(
    probs_array: Array,
    axis: int,
    domain: List[Tuple[float, float, int]],
) -> Array:
    r"""Compute the marginal probabilities of a model.

    The function computes the marginal probabilities of a model by summing over the
    specified axis.

    :param probs_array: The probabilities of the model.
    :param axis: The axis
    :param domain: The domain of the axis.
    :return: The marginal probabilities of the model.
    """
    assert axis < probs_array.ndim, "Axis must be less than the number of dimensions."
    j = 0
    marginal_density = probs_array
    for i, (start, end, num_points) in enumerate(domain):
        if i == axis:
            continue
        num_points = int(num_points)
        marginal_density = jnp.trapezoid(
            y=marginal_density,
            x=jnp.linspace(start, end, num_points),
            axis=i - j,
        )
        j += 1

    return marginal_density


def get_all_marginals(
    probs: Array,
    domains: List[Tuple[float, float, int]],
) -> List[Array]:
    """Compute marginal probabilities for all axes.

    :param probs: The probability array.
    :param domains: List of domains for each axis.
    :return: List of marginal probability arrays, one for each axis.
    """
    return [_compute_marginal_probs(probs, axis, domains) for axis in range(probs.ndim)]


def get_ppd(probs: Array, axis: int = -1) -> Array:
    """Compute the posterior predictive distribution.

    :param probs: The probability array.
    :param axis: The axis along which to compute the mean (default: -1).
    :return: The posterior predictive distribution.
    """
    return np.mean(probs, axis=axis)


def save_probs(
    ppd_array: Array,
    marginal_probs: List[Array],
    filename: str,
    domains: List[Tuple[float, float, int]],
    headers: List[str],
) -> None:
    assert ppd_array.ndim == len(
        domains
    ), "Number of ranges must match the number of dimensions of the PPD array."
    assert ppd_array.ndim == len(
        headers
    ), "Number of headers must match the number of dimensions of the PPD array."

    with h5py.File(filename, "w") as f:
        f.create_dataset("domains", data=np.array(domains))
        f.create_dataset("headers", data=np.array(headers, dtype="S"))
        f.create_dataset("ppd", data=ppd_array)
        marginal_probs_group = f.create_group("marginals")
        for marginal_prob, head in zip(marginal_probs, headers):
            marginal_probs_group.create_dataset(head, data=marginal_prob)
