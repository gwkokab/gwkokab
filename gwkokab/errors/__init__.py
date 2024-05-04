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

from typing_extensions import Optional

from jaxtyping import Array

from ..utils import get_key
from .errors import (
    banana_error_m1_m2 as banana_error_m1_m2,
    banana_error_m1_q as banana_error_m1_q,
    normal_error as normal_error,
    truncated_normal_error as truncated_normal_error,
    uniform_error as uniform_error,
)


def error_factory(
    error_type: str,
    key: Optional[int | Array] = None,
    **kwargs,
) -> Array:
    """Factory function to create different types of errors.

    :param error_type: name of the error
    :raises ValueError: if the error type is unknown
    :return: error values for the given error type
    """
    if key is None or isinstance(key, int):
        key = get_key(key)

    if error_type == "normal":
        return normal_error(key=key, **kwargs)
    elif error_type == "truncated_normal":
        return truncated_normal_error(key=key, **kwargs)
    elif error_type == "uniform":
        return uniform_error(key=key, **kwargs)
    elif error_type == "banana_m1_m2":
        return banana_error_m1_m2(key=key, **kwargs)
    elif error_type == "banana_m1_q":
        return banana_error_m1_q(key=key, **kwargs)
    else:
        raise ValueError(f"Unknown error type: {error_type}")
