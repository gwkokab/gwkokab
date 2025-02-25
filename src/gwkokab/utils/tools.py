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


import warnings
from typing import Dict, Optional, TypeVar


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def fetch_first_matching_value(dictionary: Dict[_KT, _VT], *keys: _KT) -> Optional[_VT]:
    """Get the first value in the dictionary that matches one of the keys.

    Parameters
    ----------
    dictionary : Dict[_KT, _VT]
        The dictionary to search.
    keys : _KT
        The keys to search for.

    Returns
    -------
    Optional[_VT]
        The value of the first key that is found in the dictionary, or None if no key is
        found.
    """
    for key in keys:
        if key in dictionary:
            return dictionary[key]
    return None


def error_if(cond: bool, err: Exception = ValueError, msg: str = "") -> None:
    """Raise an error if condition is met.

    Reference: utils of `interpax <https://github.com/f0uriest/interpax>`_.

    Parameters
    ----------
    cond : bool
        The condition to check.
    err : Exception, optional
        The error to raise, by default ValueError
    msg : str, optional
        The message to include with the error, by default ""

    Raises
    ------
    err
        The error raised if the condition is met.
    """
    if cond:
        raise err(msg)


def warn_if(cond: bool, err: Warning = UserWarning, msg: str = "") -> None:
    """Raise a warning if condition is met.

    Reference: utils of `interpax <https://github.com/f0uriest/interpax>`_.

    Parameters
    ----------
    cond : bool
        The condition to check.
    err : Warning, optional
        The warning to raise, by default UserWarning
    msg : str, optional
        The message to include with the warning, by default ""
    """
    if cond:
        warnings.warn(msg, err)
