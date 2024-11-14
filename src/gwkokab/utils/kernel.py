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

import jax
from jax import nn as jnn, numpy as jnp
from jaxtyping import Array


def log_planck_taper_window(x: Array, a: Array, b: Array) -> Array:
    r"""If :math:`x` is the point at which to evaluate the window, :math:`a` is the
    lower bound of the window, and :math:`b` is the window width, the Planck taper
    window is defined as,

    .. math::

        S(x\mid a,b)=\begin{cases}
            0                                                                             & \text{if } x < a,             \\
            \displaystyle\operatorname{expit}{\left(\frac{b}{b+a-x}+\frac{b}{a-x}\right)} & \text{if } a \leq x \leq a+b, \\
            1                                                                             & \text{if } x > a+b,           \\
        \end{cases}

    where :math:`\operatorname{expit}` is the logistic sigmoid function.

    .. math::

        \operatorname{expit}(x)=\frac{1}{1+e^{-x}}

    This function evaluates the log of the Planck taper window :math:`\ln{S(x\mid a,b)}`.

    :param value: point at which to evaluate the window
    :param low: lower bound of the window
    :param high: upper bound of the window
    :return: window value
    """
    return jax.jit(_log_planck_taper_window, inline=True)(x, a, b)


def _log_planck_taper_window(x: Array, a: Array, b: Array) -> Array:
    """Log Planck taper window.

    :param value: point at which to evaluate the window
    :param low: lower bound of the window
    :param high: upper bound of the window
    :return: window value
    """
    eps = 1e-6
    safe_b = jnp.where(b == 0, eps, b)
    x_norm = jnp.where(b == 0, eps, (x - a) / safe_b)
    safe_x_norm = jnp.clip(x_norm, eps, 1.0 - eps)
    condlist = [
        x_norm <= 0.0,
        (0.0 < x_norm) & (x_norm < 1.0),
        x_norm >= 1.0,
    ]
    choicelist = [
        -jnp.inf,
        -jnn.softplus(1.0 / (safe_x_norm - 1.0) + 1.0 / safe_x_norm),
        0.0,
    ]
    mask = jnp.select(condlist, choicelist)
    return mask
