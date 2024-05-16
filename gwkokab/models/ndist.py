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

import jax
from jax import jit
from jax import numpy as jnp
from numpyro import distributions as dist


def NDistribution(distribution: dist.Distribution, n: int, **params) -> dist.MixtureGeneral:
    """Mixture of any $n$ distributions.

    :param distribution: distribution to mix
    :param n: number of components
    :return: Mixture of $n$ distributions
    """
    arg_names = distribution.arg_constraints.keys()
    mixing_dist = dist.Categorical(probs=jnp.ones(n) / n, validate_args=True)
    args_per_component = [{arg: params.get(f"{arg}_{i}") for arg in arg_names} for i in range(n)]
    component_dists = jax.tree.map(
        lambda x: distribution(**x),
        args_per_component,
        is_leaf=lambda x: isinstance(x, dict),
    )
    return dist.MixtureGeneral(
        mixing_dist,
        component_dists,
        support=distribution.support,
        validate_args=True,
    )
