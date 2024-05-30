#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from typing_extensions import Callable, Optional

import jax
from jax import numpy as jnp
from jax.scipy.integrate import trapezoid
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Array, Int

from ...utils import get_key


def numerical_inverse_transform_sampling(
    logpdf: Callable[[Array], Array],
    limits: Array,
    n_samples: Int,
    *,
    seed: Optional[Int] = None,
    n_grid_points: Int = 1000,
    method: str = "linear",
    bounds_error: bool = False,
    fill_value: Optional[float] = None,
) -> Array:
    """Numerical inverse transform sampling.

    :param logpdf: log of the probability density function
    :param limits: limits of the domain
    :param n_samples: number of samples
    :param seed: random seed. defaults to None
    :param n_grid_points: number of points on grid, defaults to 1000
    :param points: length-N sequence of arrays specifying the grid coordinates.
    :param values: N-dimensional array specifying the grid values.
    :param method: interpolation method, either ``"linear"`` or ``"nearest"``.
    :param bounds_error: error if points are outside the grid, defaults to False
    :param fill_value: value returned for points outside the grid, defaults to NaN.
    :raises ValueError: if method is not 'linear' or 'nearest'
    :return: samples from the distribution
    """
    if method not in ["linear", "nearest"]:
        raise ValueError("method must be either 'linear' or 'nearest'")
    if seed is None:
        key = get_key()
    else:
        key = jax.random.PRNGKey(seed)
    grid = jnp.linspace(limits[0], limits[1], n_grid_points)  # 1000 grid points
    pdf = jnp.exp(logpdf(grid))  # pdf
    pdf = pdf / trapezoid(y=pdf, x=grid, axis=0)  # normalize
    cdf = jnp.cumsum(pdf, axis=0)  # cdf
    cdf = cdf / cdf[-1]  # normalize
    cdf = RegularGridInterpolator(  # interpolate
        (cdf.flatten(),),
        grid.flatten(),
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    u = jax.random.uniform(key, (n_samples,))  # uniform samples
    return cdf(u)  # inverse transform sampling
