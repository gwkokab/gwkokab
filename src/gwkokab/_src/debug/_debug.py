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


from typing_extensions import Callable

import jax


DEBUG: bool = False


def enable_debugging() -> None:
    r"""Enables debugging mode."""
    global DEBUG
    DEBUG = True


def debug(func) -> Callable:
    r"""A decorator that prints the arguments and return value of a function when
    :code:`DEBUG` is :code:`True`.

    :param func: The function to be decorated.
    :return: A wrapper function that prints the arguments and return value of the
    decorated function.
    """

    def wrapper(*args, **kwargs):
        if DEBUG:
            jax.debug.print(
                "Calling {func_name} with arguments {args_fmt} and {kwargs_fmt}",
                func_name=func.__name__,
                args_fmt=args,
                kwargs_fmt=kwargs,
            )
        result = func(*args, **kwargs)
        if DEBUG:
            jax.debug.print(
                "{func_name} returned {result_fmt}",
                func_name=func.__name__,
                result_fmt=result,
            )
        return result

    return wrapper
