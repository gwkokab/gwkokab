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

from jax import numpy as jnp, vmap
from jaxtyping import Array, Bool

from gwkokab.utils.transformations import m1_q_to_m2
from gwkokab.vts import load_model


def get_logVT(vt_path):
    _, logVT = load_model(vt_path)

    def m1q_logVT(x: Array) -> Array:
        m1 = x[..., 0]
        q = x[..., 1]
        m2 = m1_q_to_m2(m1=m1, q=q)
        m1m2 = jnp.column_stack([m1, m2])
        return vmap(logVT)(m1m2)

    return m1q_logVT


def constraint(x: Array) -> Bool:
    m1 = x[..., 0]
    q = x[..., 1]
    mask = m1 > 0.0
    mask &= q >= 0.0
    mask &= q <= 1.0
    return mask
