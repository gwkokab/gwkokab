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

from typing_extensions import Any, Optional

import jax
import numpy as np
from jaxtyping import Array


def dump_configurations(filename: str, *args: tuple[str, Any]) -> None:
    """Write the given configurations to a file.

    This function writes the given configurations to a file. The
    configurations are written as a CSV file. The first row contains
    the header and the second row contains the content.

    :param filename: name of the file
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
    r"""Get a new JAX random key if not provide. The key is
    generated using the `jax.random.PRNGKey` function. It is
    split into two keys, the first of which is returned. The
    second key is discarded.

    :param key: JAX random key or seed value, defaults to `None`
    :return: New JAX random key
    """
    if isinstance(key, int):
        return jax.random.PRNGKey(key)

    if key is None:
        new_key = jax.random.PRNGKey(np.random.randint(0, 1000_000))
    else:
        new_key, _ = jax.random.split(key)

    return new_key
