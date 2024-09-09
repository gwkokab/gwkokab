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

from typing_extensions import List


def expand_arguments(arg: str, n: int) -> List[str]:
    r"""Extend the argument with a number of strings.

    .. code:: python

        expand_arguments("physics", 3)
        # ["physics_0", "physics_1", "physics_2"]

    :param arg: argument to extend
    :param n: number of strings to extend
    :return: list of extended arguments
    """
    return list(map(lambda i: arg + f"_{i}", range(n)))
