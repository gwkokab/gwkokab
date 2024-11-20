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

from collections.abc import Sequence
from typing_extensions import Callable

from jax import numpy as jnp, vmap
from jaxtyping import Array, Bool, Float

from gwkokab.vts import load_model


def get_logVT(
    vt_path: str, selection_indexes: Sequence[int]
) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    _, logVT = load_model(vt_path)

    def trimmed_logVT(x: Float[Array, "..."]) -> Float[Array, "..."]:
        m1q = x[..., selection_indexes]
        m2 = x[..., 0] * x[..., 1]
        m1m2 = m1q.at[..., 1].set(m2)

        return jnp.squeeze(vmap(logVT)(m1m2), axis=-1)

    return trimmed_logVT


def constraint(
    x: Float[Array, "..."],
    has_spin: Bool[bool, "True", "False"],
    has_tilt: Bool[bool, "True", "False"],
    has_eccentricity: Bool[bool, "True", "False"],
    has_redshift: Bool[bool, "True", "False"],
) -> Bool[Array, "..."]:
    """Applies physical constraints to the input array.

    :param x: Input array where:

        - x[..., 0] is mass m1
        - x[..., 1] is mass ratio q
        - x[..., 2] is chi1 (if has_spin is True)
        - x[..., 3] is chi2 (if has_spin is True)
        - x[..., 4] is cos_tilt_1 (if has_tilt is True)
        - x[..., 5] is cos_tilt_2 (if has_tilt is True)
        - x[..., 6] is eccentricity (if has_eccentricity is True)
        - x[..., 7] is redshift z (if has_redshift is True)

    :param has_spin: Whether to apply spin constraints.
    :param has_tilt: Whether to apply tilt constraints.
    :param has_eccentricity: Whether to apply eccentricity constraints.
    :return: Boolean array indicating which samples satisfy the constraints.
    """
    m1 = x[..., 0]
    q = x[..., 1]

    mask = m1 > 0.0
    mask &= q > 0.0
    mask &= q <= 1.0

    i = 2

    if has_spin:
        chi1 = x[..., i]
        chi2 = x[..., i + 1]

        mask &= chi1 >= 0.0
        mask &= chi1 <= 1.0

        mask &= chi2 >= 0.0
        mask &= chi2 <= 1.0

        i += 2

    if has_tilt:
        cos_tilt_1 = x[..., i]
        cos_tilt_2 = x[..., i + 1]

        mask &= cos_tilt_1 >= -1.0
        mask &= cos_tilt_1 <= 1.0

        mask &= cos_tilt_2 >= -1.0
        mask &= cos_tilt_2 <= 1.0

        i += 2

    if has_eccentricity:
        ecc = x[..., i]

        mask &= ecc >= 0.0
        mask &= ecc <= 1.0

        i += 1

    if has_redshift:
        z = x[..., i]

        mask &= z >= 1e-3

    return mask