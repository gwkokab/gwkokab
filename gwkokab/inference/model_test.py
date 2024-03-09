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
from typing_extensions import Optional

import jax
from jax import jit, numpy as jnp

from gwkokab.models import *
from gwkokab.utils.misc import get_key
from gwkokab.vts.utils import interpolate_hdf5


raw_interpolator = interpolate_hdf5()


@partial(jit, static_argnums=(0, 1))
def exp_rate(rate, lambdas) -> float:
    m1 = lambdas[..., 0]
    m2 = lambdas[..., 1]
    value = raw_interpolator(m1, m2)
    I = jnp.mean(value)
    return rate * I


@partial(jit, static_argnums=(0,))
def log_inhomogeneous_poisson_likelihood(x, data: Optional[dict] = None):
    alpha = x[..., 0]
    mmin = x[..., 1]
    mmax = x[..., 2]
    rate = x[..., 3]

    mass_model = Wysocki2019MassModel(
        alpha_m=alpha,
        k=0,
        mmin=mmin,
        mmax=mmax,
    )
    log_rate = jnp.log(rate)
    integral_individual = jax.tree_map(
        lambda x: jax.nn.logsumexp(mass_model.log_prob(x)) + log_rate - jnp.log(x.shape[0]),
        data["data"],
    )

    log_likelihood = jnp.sum(jnp.asarray(jax.tree.leaves(integral_individual)))

    N = 1 << 14  # 2 ** 14
    lambdas = mass_model.sample(get_key(), sample_shape=(N,))

    expected_rate = exp_rate(
        rate,
        lambdas,
    )
    return log_likelihood - expected_rate
