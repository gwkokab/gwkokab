# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0
"""Custom Logged Exceptions and Warnings for gwkokab.

This module provides a registry of standard Python exceptions and warnings
extended with automatic logging capabilities via `loguru <https://github.com/Delgan/loguru>`_.

The mixin classes :class:`LoggedMixinException` and :class:`LoggedMixinWarning` provide
the functionality to format and log messages for exceptions and warnings, respectively.
The custom exception and warning classes inherit from these mixins and the corresponding
built-in exception or warning classes. When an instance of these custom classes is
created, the message is formatted using the provided arguments and keyword arguments,
and then logged at the appropriate level (error for exceptions and warning for
warnings). If formatting fails, the original message is used without formatting. This
allows for consistent logging of error and warning messages throughout the application.

You can use these custom exceptions and warnings in your code to automatically log
messages when they are raised or issued. For example:

.. code:: python

    import warnings

    from gwkokab.utils.exceptions import LoggedMixinException, LoggedMixinWarning


    class LoggedValueError(LoggedMixinException, ValueError):
        pass


    class LoggedUserWarning(LoggedMixinWarning, UserWarning):
        pass


    warnings.warn("This is a warning message", LoggedUserWarning)

    raise LoggedValueError("Invalid value: {value}", value=42)
"""

from typing import Any

from loguru import logger


__all__ = [
    "LoggedAssertionError",
    "LoggedDeprecationWarning",
    "LoggedImportError",
    "LoggedIndexError",
    "LoggedKeyError",
    "LoggedMixinException",
    "LoggedMixinWarning",
    "LoggedNotImplementedError",
    "LoggedRuntimeWarning",
    "LoggedTypeError",
    "LoggedUserWarning",
    "LoggedValueError",
]


def _format(msg: str, args: tuple, kwargs: dict) -> str:
    """Formats the message string using the provided arguments and keyword arguments. If
    formatting fails, it returns the original message string without formatting.

    Parameters
    ----------
    msg : str
        The message string to be formatted.
    args : tuple
        Positional arguments to be used for formatting the message string.
    kwargs : dict
        Keyword arguments to be used for formatting the message string.

    Returns
    -------
    str
        The formatted message string. If formatting fails, returns the original message string.
    """
    if args or kwargs:
        try:
            return msg.format(*args, **kwargs)
        except Exception:
            return msg
    return msg


class LoggedMixinException:
    r"""A mixin class for exceptions that logs the error message using loguru when the
    exception is instantiated.

    The message is formatted using the provided arguments and keyword arguments. If
    formatting fails, the original message is used without formatting. The formatted
    message is logged at the error level.
    """

    def __init__(
        self,
        msg: str,
        *args,
        loguru_opt: dict[str, Any] = {"depth": 1},
        **kwargs,
    ):

        formatted = _format(msg, args, kwargs)
        logger.opt(**loguru_opt).error(msg, *args, **kwargs)
        super().__init__(formatted)  # type: ignore[call-arg]


class LoggedMixinWarning:
    r"""A mixin class for warnings that logs the warning message using loguru when the
    warning is instantiated.

    The message is formatted using the provided arguments and keyword arguments. If
    formatting fails, the original message is used without formatting. The formatted
    message is logged at the warning level.
    """

    def __init__(
        self, msg: str, *args, loguru_opt: dict[str, Any] = {"depth": 1}, **kwargs
    ):
        formatted = _format(msg, args, kwargs)
        logger.opt(**loguru_opt).warning(msg, *args, **kwargs)
        super().__init__(formatted)  # type: ignore[call-arg]


# fmt: off
class LoggedAssertionError(LoggedMixinException, AssertionError): ...
class LoggedImportError(LoggedMixinException, ImportError): ...
class LoggedIndexError(LoggedMixinException, IndexError): ...
class LoggedKeyError(LoggedMixinException, KeyError): ...
class LoggedNotImplementedError(LoggedMixinException, NotImplementedError): ...
class LoggedTypeError(LoggedMixinException, TypeError): ...
class LoggedValueError(LoggedMixinException, ValueError): ...

class LoggedUserWarning(LoggedMixinWarning, UserWarning): ...
class LoggedRuntimeWarning(LoggedMixinWarning, RuntimeWarning): ...
class LoggedDeprecationWarning(LoggedMixinWarning, DeprecationWarning): ...
# fmt: on
