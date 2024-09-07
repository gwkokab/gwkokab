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

import re
from typing_extensions import Dict, List


def matches_regex(pattern: str, string: str) -> bool:
    r"""Check if a string matches a regex pattern.

    :param pattern: regex pattern to match
    :param string: string to match
    :return: :code:`True` if the string matches the pattern, `code`:False: otherwise
    """
    return bool(re.fullmatch(pattern, string))


def match_all(
    strings: List[str], pattern_dict_with_val: Dict[str, float]
) -> Dict[str, float]:
    r"""Match all strings in a list with a dictionary of regex patterns.

    :param strings: list of strings to match
    :param pattern_dict_with_val: dictionary of regex patterns with values
    :return: dictionary of matched patterns with values
    """
    matches = {}
    for string in strings:
        for pattern, value in pattern_dict_with_val.items():
            if matches_regex(pattern, string):
                matches[string] = value
                break
    return matches
