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

from jaxtyping import Array, Bool


def constraint(x: Array) -> Bool:
    m1 = x[..., 0]
    m2 = x[..., 1]
    ecc = x[..., 2]
    mask = m2 <= m1
    mask &= m2 > 0.0
    mask &= m1 > 0.0
    mask &= ecc >= 0.0
    mask &= ecc <= 1.0
    return mask
