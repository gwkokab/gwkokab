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
from typing_extensions import Optional

import jax
import numpy as np
from jax import jit, lax
from jaxtyping import Array

from ..typing import Numeric


@jit
def chirp_mass(m1: Numeric, m2: Numeric) -> Numeric:
    r"""
    .. math::

        M_c = \frac{(m_1 m_2)^{\frac{3}{5}}}{(m_1 + m_2)^{\frac{1}{5}}}
    :param m1: mass 1
    :param m2: mass 2
    :return:
    """
    return lax.mul(lax.pow(lax.mul(m1, m2), 0.6), lax.pow(lax.add(m1, m2), -0.2))


@jit
def symmetric_mass_ratio(m1: Numeric, m2: Numeric) -> Numeric:
    r"""
    .. math::

        \eta = \frac{m_1 m_2}{(m_1 + m_2)^2}
    :param m1: mass 1
    :param m2: mass 2
    :return:
    """
    return lax.mul(lax.mul(m1, m2), lax.pow(lax.add(m1, m2), -2))


@jit
def reduced_mass(m1: Numeric, m2: Numeric) -> Numeric:
    """
    .. math::

        M_r = \frac{m_1 m_2}{m_1 + m_2}
    :param m1: mass 1
    :param m2: mass 2
    :return:
    """
    return lax.div(lax.mul(m1, m2), lax.add(m1, m2))


@jit
def mass_ratio(m1: Numeric, m2: Numeric) -> Numeric:
    r"""
    .. math::

        q=\frac{m_1}{m_2}
    :param m1: mass 1
    :param m2: mass 2
    :return:
    """
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
