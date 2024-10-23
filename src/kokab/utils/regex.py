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
from typing import TypeVar
from typing_extensions import Dict, List


_VT = TypeVar("_VT")


def matches_regex(pattern: str, string: str) -> bool:
    r"""Check if a string matches a regex pattern.

    :param pattern: regex pattern to match
    :param string: string to match
    :return: :code:`True` if the string matches the pattern, `code`:False: otherwise
    """
    return bool(re.fullmatch(pattern, string))


def match_all(
    strings: List[str], pattern_dict_with_val: Dict[str, str | _VT]
) -> Dict[str, str | _VT | None]:
    r"""Match all strings in a list with a dictionary of regex patterns.

    :param strings: list of strings to match
    :param pattern_dict_with_val: dictionary of regex patterns with values
    :return: dictionary of matched patterns with values
    """
    matches: Dict[str, str | _VT | None] = {}
    duplicates = []
    for string in strings:
        if pattern_dict_with_val.get(string):  # Exact match
            matched_string = pattern_dict_with_val[string]
            # Check for duplicates and if the value they are duplicated with is not parsed yet
            # then add them to duplicates list to be parsed later, else add them to matches
            if isinstance(matched_string, str) and matches.get(matched_string) is None:
                duplicates.append(string)
            else:
                matches[string] = matched_string
            continue
        pattern_found = False
        for pattern, value in pattern_dict_with_val.items():
            if matches_regex(pattern, string):
                matches[string] = value
                pattern_found = True
                break
        if not pattern_found:
            matches[string] = None
    for duplicate in duplicates:
        matches[duplicate] = matches[pattern_dict_with_val[duplicate]]
    return matches
