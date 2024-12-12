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


from typing import TypeVar
from typing_extensions import Dict


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def fetch_first_matching_value(dictionary: Dict[_KT, _VT], *keys: _KT) -> _VT | None:
    """Get the first value in the dictionary that matches one of the keys.

    :param dictionary: The dictionary to search.
    :param keys: The keys to search for in order.
    :return: The value of the first key that is found in the dictionary, or None if
        no key is found.
    """
    for key in keys:
        if key in dictionary:
            return dictionary[key]
    return None
