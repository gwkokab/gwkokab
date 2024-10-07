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

from typing_extensions import Callable

from jax import vmap
from jaxtyping import Array, Bool, Float

from gwkokab.vts import load_model


def get_logVT(vt_path) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    _, logVT = load_model(vt_path)

    def m1m2_trimmed_logVT(x: Float[Array, "..."]) -> Float[Array, "..."]:
        m1m2 = x[..., 0:2]
        return vmap(logVT)(m1m2)

    return m1m2_trimmed_logVT


def constraint(
    x: Float[Array, "..."],
    has_spin: Bool[bool, "True", "False"],
    has_tilt: Bool[bool, "True", "False"],
) -> Bool[Array, "..."]:
    m1 = x[..., 0]
    m2 = x[..., 1]

    mask = m1 > 0.0
    mask &= m2 > 0.0
    mask &= m1 >= m2

    if has_spin:
        chi1 = x[..., 2]
        chi2 = x[..., 3]

        mask &= chi1 >= 0.0
        mask &= chi1 <= 1.0

        mask &= chi2 >= 0.0
        mask &= chi2 <= 1.0

    if has_tilt:
        cos_tilt_1 = x[..., 4]
        cos_tilt_2 = x[..., 5]

        mask &= cos_tilt_1 >= -1.0
        mask &= cos_tilt_1 <= 1.0

        mask &= cos_tilt_2 >= -1.0
        mask &= cos_tilt_2 <= 1.0

    return mask
