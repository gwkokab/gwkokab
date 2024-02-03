#  Copyright 2023 The Jaxtro Authors
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

from jax import jit, lax
from jaxampler.typing import Numeric


@jit
def chirp_mass(m1: Numeric, m2: Numeric) -> Numeric:
    # return jnp.power(m1 * m2, 0.6) / jnp.power(m1 + m2, 0.2)
    return lax.div(lax.pow(lax.mul(m1, m2), 0.6), lax.pow(lax.add(m1, m2), 0.2))


@jit
def symmetric_mass_ratio(m1: Numeric, m2: Numeric) -> Numeric:
    # return (m1 * m2) / jnp.power(m1 + m2, 2)
    return lax.div(lax.mul(m1, m2), lax.pow(lax.add(m1, m2), 2))


@jit
def reduced_mass(m1: Numeric, m2: Numeric) -> Numeric:
    # return (m1 * m2) / (m1 + m2)
    return lax.div(lax.mul(m1, m2), lax.add(m1, m2))


@jit
def mass_ratio(m1: Numeric, m2: Numeric) -> Numeric:
    # return m1 / m2
    return lax.div(m1, m2)


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
