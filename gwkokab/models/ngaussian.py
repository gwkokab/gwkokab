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


from __future__ import annotations

from functools import partial

from jax import jit, numpy as jnp
from numpyro import distributions as dist


@partial(jit, static_argnums=(0,))
def NGaussian(n: int, **params) -> dist.MixtureSameFamily:
    r"""Mixture of equally weighted Gaussian components.

    ```python
    >>> N = 3
    >>> ng = NGaussian(
    ...     n=N,
    ...     loc_0=0.0,
    ...     loc_1=1.0,
    ...     loc_2=2.0,
    ...     scale_0=0.1,
    ...     scale_1=0.2,
    ...     scale_2=0.3,
    ... )
    >>> samples = ng.sample(get_key(), (1000,))
    >>> samples.shape
    (1000,)
    ```

    :param n: Number of Gaussian components.
    :param loc_i: Mean of the i-th Gaussian component. Default is 0.0.
    :param scale_i: Standard deviation of the i-th Gaussian component. Default is 1.0.
    :return: Mixture of equally weighted Gaussian components.
    """
    loc = jnp.array([params.get(f"loc_{i}", 0.0) for i in range(n)])
    scale = jnp.array([params.get(f"scale_{i}", 1.0) for i in range(n)])
    mixing_dist = dist.Categorical(probs=jnp.ones(n) / n)
    component_dist = dist.Normal(loc=loc, scale=scale, validate_args=True)
    return dist.MixtureSameFamily(mixing_dist, component_dist, validate_args=True)
