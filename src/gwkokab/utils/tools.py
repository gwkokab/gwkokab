# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import warnings
from typing import Dict, Optional, TypeVar

from loguru import logger


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
        logger.error(msg)
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
        logger.warning(msg)
        warnings.warn(msg, err)
