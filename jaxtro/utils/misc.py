# Copyright 2023 The Jaxtro Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from jax import Array
from jax import numpy as jnp
from jaxampler.rvs import Normal


def add_normal_error(*x: tuple[float], scale: float = 0.01, size: int = 10) -> Array:
    """Adds error to the masses of the binaries

    Uses a normal distribution with mean as provided values and standard deviation as scale

    Parameters
    ----------
    *x : ArrayLike
        values to add error to
    scale : float, optional
        error scale (standard deviation of normal distribution), by default 0.01
    size : int, optional
        number of points after adding error, by default 10

    Returns
    -------
    Array
        array of shape (size, len(x)) with error added to each value in x
    """
    return jnp.column_stack([Normal(mu=xi, sigma=scale).rvs(size) for xi in x])


def dump_configurations(filename: str, *args: list[tuple[str, Any]]) -> None:
    """Dump configurations to a csv file

    Parameters
    ----------
    filename : str
        filename to dump the configurations
    """
    with open(filename, "w") as f:
        header = ""
        content = ""
        for h, c in args:
            header += f"{h},"
            content += f"{c},"

        f.write(f"{header[:-1]}\n")
        f.write(f"{content[:-1]}\n")
