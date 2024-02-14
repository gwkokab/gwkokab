#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing_extensions import Any, Callable, Optional

import jax
from jax import jit, numpy as jnp, vmap
from jaxtyping import Array

from ..typing import Numeric
from ..utils import get_key, gwk_shape_cast


class AbstractModel(object):
    def __init__(self, name: Optional[str] = None, shape: tuple[int, ...] = ()) -> None:
        self._name = name
        self._shape = shape
        self._stack = []

    @abstractmethod
    def samples(self, num_of_samples: int) -> Array:
        raise NotImplementedError

    @abstractmethod
    def add_error(self, x: Array, scale: float, size: int) -> Numeric:
        raise NotImplementedError

    def check_params(self) -> None:
        raise NotImplementedError

    # POINT VALUED

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _logppf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _pdf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logpdf_x(*x))

    @partial(jit, static_argnums=(0,))
    def _cdf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logcdf_x(*x))

    @partial(jit, static_argnums=(0,))
    def _ppf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logppf_x(*x))

    # VECTOR VALUED

    @partial(jit, static_argnums=(0,))
    def _logpdf_v(self, *x: Numeric) -> Numeric:
        return vmap(self._logpdf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def _logcdf_v(self, *x: Numeric) -> Numeric:
        return vmap(self._logcdf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def _logppf_v(self, *x: Numeric) -> Numeric:
        return vmap(self._logppf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def _cdf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logcdf_v(*x))

    @partial(jit, static_argnums=(0,))
    def _pdf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logpdf_v(*x))

    @partial(jit, static_argnums=(0,))
    def _ppf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logppf_v(*x))

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        raise NotImplementedError

    # XXF FACTORY METHODS

    def _pv_factory(
        self,
        func_p_repr: Callable[["AbstractModel"], Callable[[Numeric], Numeric]],
        func_v_repr: Callable[["AbstractModel"], Callable[[Numeric], Numeric]],
        shape: tuple[int, ...],
    ) -> Callable:
        if len(shape) < 2:
            fn = func_p_repr
        else:
            fn = func_v_repr

        if len(self._stack) == 0:
            return lambda *args: fn(self)(*args)
        return lambda *args: self._evaluate(fn, *args)

    @partial(jit, static_argnums=(0,))
    def pdf(self, *x: Numeric) -> Numeric:
        shape = gwk_shape_cast(*x)
        fn = self._pv_factory(lambda x: x._pdf_x, lambda x: x._pdf_v, shape)
        return fn(*x)

    @partial(jit, static_argnums=(0,))
    def cdf(self, *x: Numeric) -> Numeric:
        shape = gwk_shape_cast(*x)
        fn = self._pv_factory(lambda x: x._cdf_x, lambda x: x._cdf_v, shape)
        return fn(*x)

    @partial(jit, static_argnums=(0,))
    def ppf(self, *x: Numeric) -> Numeric:
        shape = gwk_shape_cast(*x)
        fn = self._pv_factory(lambda x: x._ppf_x, lambda x: x._ppf_v, shape)
        return fn(*x)

    @partial(jit, static_argnums=(0,))
    def logpdf(self, *x: Numeric) -> Numeric:
        shape = gwk_shape_cast(*x)
        fn = self._pv_factory(lambda x: x._logpdf_x, lambda x: x._logpdf_v, shape)
        return fn(*x)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, *x: Numeric) -> Numeric:
        shape = gwk_shape_cast(*x)
        fn = self._pv_factory(lambda x: x._logcdf_x, lambda x: x._logcdf_v, shape)
        return fn(*x)

    @partial(jit, static_argnums=(0,))
    def logppf(self, *x: Numeric) -> Numeric:
        shape = gwk_shape_cast(*x)
        fn = self._pv_factory(lambda x: x._logppf_x, lambda x: x._logppf_v, shape)
        return fn(*x)

    def rvs(self, shape: tuple[int, ...], seed: Optional[int] = None) -> Array:
        if seed is None:
            key = get_key()
        else:
            key = jax.random.PRNGKey(seed)
        new_shape = shape + self._shape
        return self._rvs(shape=new_shape, key=key)

    # postfix notation methods

    def _add_expression(self, op: Callable, *args: Any) -> None:
        stack: list[Any] = [op]
        for arg in args:
            if isinstance(arg, AbstractModel) and len(arg._stack) != 0:
                stack.extend(arg._stack)
            else:
                stack.append(arg)
        self._stack = stack

    def _evaluate(self, func: Callable, *args, **kwargs) -> Any:
        s = []
        s_ = []
        for i in range(len(self._stack)):
            if isinstance(self._stack[i], AbstractModel):
                s_.append(func(self._stack[i])(*args, **kwargs))
            else:
                s_.append(self._stack[i])
        for item in s_[::-1]:
            if isinstance(item, Callable):
                s.append(item(*[s.pop() for _ in range(item.__code__.co_argcount)]))
            else:
                s.append(item)
        return s[-1]

    # arithmetic operations

    def __add__(self, other) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({self.__repr__()} + {other.__repr__()})")
        new_variable._add_expression(lambda x, y: x + y, self, other)
        return new_variable

    def __sub__(self, other) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({self.__repr__()} - {other.__repr__()})")
        new_variable._add_expression(lambda x, y: x - y, self, other)
        return new_variable

    def __neg__(self):
        new_variable = AbstractModel(name=f"(-{self.__repr__()})")
        new_variable._add_expression(lambda x: -x, self)
        return new_variable

    def __mul__(self, other) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({self.__repr__()} * {other.__repr__()})")
        new_variable._add_expression(lambda x, y: x * y, self, other)
        return new_variable

    def __truediv__(self, other) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({self.__repr__()} / {other.__repr__()})")
        new_variable._add_expression(lambda x, y: x / y, self, other)
        return new_variable

    def __pow__(self, power, modulo=None) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({self.__repr__()}**{power})")
        new_variable._add_expression(lambda x, y: jnp.power(x, y), self, power)
        return new_variable

    # reverse arithmetic operations

    def __radd__(self, other) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({other.__repr__()} + {self.__repr__()})")
        new_variable._add_expression(lambda x, y: x + y, other, self)
        return new_variable

    def __rsub__(self, other) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({other.__repr__()} - {self.__repr__()})")
        new_variable._add_expression(lambda x, y: x - y, other, self)
        return new_variable

    def __rmul__(self, other) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({other.__repr__()} * {self.__repr__()})")
        new_variable._add_expression(lambda x, y: x * y, other, self)
        return new_variable

    def __rtruediv__(self, other) -> "AbstractModel":
        new_variable = AbstractModel(name=f"({other.__repr__()} / {self.__repr__()})")
        new_variable._add_expression(lambda x, y: x / y, other, self)
        return new_variable

    def __repr__(self) -> str:
        if self._name is None:
            return ""
        return self._name

    def __str__(self) -> str:
        return self.__repr__()
