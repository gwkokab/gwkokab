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

from __future__ import annotations

from typing import Any
from typing_extensions import Any, Optional, Unpack

import jax
import numpy as np
from jax import lax, numpy as jnp
from jax._src import core
from jaxtyping import Array, Integer


def gwk_shape_cast(
    *args: Any,
) -> tuple[int, ...]:
    # partially taken from the implementation of `jnp.broadcast_arrays`
    shapes = [np.shape(arg) for arg in args]
    if not shapes or all(core.definitely_equal_shape(shapes[0], s) for s in shapes):
        result_shape = shapes[0]
    else:
        result_shape: tuple[int, ...] = lax.broadcast_shapes(*shapes)
    return result_shape


def gwk_array_cast(
    *args: Any,
) -> tuple[tuple[int, ...], Unpack[tuple[Any, ...]]]:
    """Cast provided arguments to `jnp.array` and checks if they can be
    broadcast.

    Parameters
    ----------
    *args:
        Arguments to cast and check.

    Returns
    -------
    list[Array]
        List of cast arguments.
    """
    return gwk_shape_cast(*args), *tuple(jnp.asarray(arg) for arg in args)


fact = [1, 1, 2, 6, 24, 120, 720, 5_040, 40_320, 362_880, 3_628_800]


def nPr(n: Integer, r: Integer) -> Integer:
    """Calculates the number of permutations of `r` objects out of `n`

    Parameters
    ----------
    n : Integer
        total objects
    r : Integer
        selected objects

    Returns
    -------
    Integer
        number of permutations of `r` objects out of `n`
    """
    assert 0 <= r <= n
    if n <= len(fact):
        return fact[n] // fact[n - r]
    for i in range(len(fact), n + 1):
        fact.append(fact[i - 1] * i)
    return fact[n] // fact[n - r]


def nCr(n: Integer, r: Integer) -> Integer:
    """Calculates the number of combinations of `r` objects out of `n`

    Parameters
    ----------
    n : Integer
        total objects
    r : Integer
        selected objects

    Returns
    -------
    Integer
        number of combinations of `r` objects out of `n`
    """
    assert 0 <= r <= n
    if n <= len(fact):
        return (fact[n] // fact[r]) // fact[n - r]
    for i in range(len(fact), n + 1):
        fact.append(fact[i - 1] * i)
    return fact[n] // (fact[r] * fact[n - r])


def dump_configurations(filename: str, *args: tuple[str, Any]) -> None:
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


def get_key(key: Optional[Array | int] = None) -> Array:
    """Get a new JAX random key.

    This function is used to generate a new JAX random key if
    the user does not provide one. The key is generated using
    the JAX random.PRNGKey function. The key is split into
    two keys, the first of which is returned. The second key
    is discarded.

    Parameters
    ----------
    key : Array, optional
        JAX random key, by default None

    Returns
    -------
    Array
        New JAX random key.
    """
    if isinstance(key, int):
        return jax.random.PRNGKey(key)

    if key is None:
        new_key = jax.random.PRNGKey(np.random.randint(0, 1000_000))
    else:
        new_key, _ = jax.random.split(key)

    return new_key
