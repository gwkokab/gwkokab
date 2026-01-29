# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import re
import warnings
from typing import Dict, List


def matches_regex(pattern: str, string: str) -> bool:
    """Check if a string matches a regex pattern.

    Parameters
    ----------
    pattern : str
        regex pattern to match
    string : str
        string to match

    Returns
    -------
    bool
        :code:`True` if the string matches the pattern, `code`:False: otherwise
    """
    return bool(re.fullmatch(pattern, string))


def match_all(
    strings: List[str], pattern_dict_with_val: Dict[str, str | int | float | None]
) -> Dict[str, int | float | None]:
    """Match all strings in a list with a dictionary of regex patterns.

    Parameters
    ----------
    strings : List[str]
        list of strings to match
    pattern_dict_with_val : Dict[str, str  |  int  |  float  |  None]
        dictionary of regex patterns with values

    Returns
    -------
    Dict[str, int | float | None]
        dictionary of matched patterns with values
    """
    # TODO(Qazalbash): Simplify the logic
    matches: Dict[str, int | float | None] = {}
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
        pattern_found = False
        for pattern, value in pattern_dict_with_val.items():
            matched_duplicate = matches_regex(pattern, pattern_dict_with_val[duplicate])
            if not isinstance(value, dict) and matched_duplicate:  # do not match dict
                matches[duplicate] = value
                pattern_found = True
                break
            elif isinstance(value, dict):
                pattern_found = True
                matches[duplicate] = pattern_dict_with_val[duplicate]
        if not pattern_found:
            matches[duplicate] = None

    for string in strings:
        if matches.get(string) is None:
            warnings.warn(f"'{string}' does not match any pattern.")

    return matches
