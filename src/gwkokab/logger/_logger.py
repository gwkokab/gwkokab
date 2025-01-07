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


import atexit as _atexit
import sys

import jax
from loguru._logger import Core as _Core, Logger as _Logger


DEBUG: bool = False


class Logger(_Logger):
    def trace(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'TRACE'``."""
        if DEBUG:
            jax.debug.callback(
                __self._log,
                level="TRACE",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def debug(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'DEBUG'``."""
        if DEBUG:
            jax.debug.callback(
                __self._log,
                level="DEBUG",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def info(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'INFO'``."""
        if DEBUG:
            jax.debug.callback(
                __self._log,
                level="INFO",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def success(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'SUCCESS'``."""
        if DEBUG:
            jax.debug.callback(
                __self._log,
                level="SUCCESS",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def warning(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'WARNING'``."""
        if DEBUG:
            jax.debug.callback(
                __self._log,
                level="WARNING",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def error(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'ERROR'``."""
        if DEBUG:
            jax.debug.callback(
                __self._log,
                level="ERROR",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def critical(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'CRITICAL'``."""
        if DEBUG:
            jax.debug.callback(
                __self._log,
                level="CRITICAL",
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def exception(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log an ``'ERROR'```` message while also capturing the currently handled
        exception.
        """
        if DEBUG:
            options = (True,) + __self._options[1:]
            jax.debug.callback(
                __self._log,
                level="ERROR",
                from_decorator=False,
                options=options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )

    def log(__self, __level, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``level``."""
        if DEBUG:
            jax.debug.callback(
                __self._log,
                level=__level,
                from_decorator=False,
                options=__self._options,
                message=__message,
                args=args,
                kwargs=kwargs,
            )


logger = Logger(
    core=_Core(),
    exception=None,
    depth=0,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)


_atexit.register(logger.remove)


def set_log_level(log_level: str = "DEBUG") -> None:
    """Set the log level.

    Parameters
    ----------
    log_level : str, optional
        The log level, by default "DEBUG".
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>",
    )


def enable_logging() -> None:
    """Enable logging."""
    set_log_level()
    global DEBUG
    DEBUG = True
