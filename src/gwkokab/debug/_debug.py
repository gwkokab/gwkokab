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


from typing_extensions import Callable

import jax


DEBUG: bool = False


def enable_debugging() -> None:
    r"""Enables debugging mode."""
    global DEBUG
    DEBUG = True


def debug_mode() -> bool:
    r"""Returns the current debugging mode."""
    return DEBUG


def debug_flush(fmt_str: str, **kwargs) -> None:
    """Flushes the debug message to the console.

    .. note::
        This function will only print message in debug mode.

    >>> debug_flush("Hello, {name}!", name="world")
    Hello, world!

    Parameters
    ----------
    fmt_str : str
        The format string for the debug message.
    """
    if debug_mode():
        jax.debug.print("\033[32mGWKokab " + fmt_str, **kwargs)


def debug(func: Callable) -> Callable:
    """A decorator that prints the arguments and return value of a function when
    :code:`DEBUG` is :code:`True`.

    Parameters
    ----------
    func : Callable
        The function to be decorated

    Returns
    -------
    Callable
        A wrapper function that prints the arguments and return value of the
        decorated function.
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if debug_mode():
            jax.debug.print(
                "\033[32mGWKokab Calling {func_name} with arguments {args_fmt} "
                "and {kwargs_fmt} returned {result_fmt}",
                func_name=func.__name__,
                result_fmt=result,
            )
        return result

    return wrapper
