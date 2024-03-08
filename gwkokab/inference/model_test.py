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

from typing_extensions import Optional

import jax
from jax import jit, numpy as jnp

from gwkokab.vts.utils import interpolate_hdf5

from ..models import *
from ..utils.misc import get_key


@jit
def exp_rate(rate, *, pop_params) -> float:
    N = 1 << 13  # 2 ** 13
    lambdas = Wysocki2019MassModel(
        alpha_m=pop_params["alpha_m"],
        k=0,
        mmin=pop_params["mmin"],
        mmax=pop_params["mmax"],
    ).sample(get_key(), sample_shape=(N,))
    I = 0
    m1 = lambdas[..., 0]
    m2 = lambdas[..., 1]
    value = jnp.exp(interpolate_hdf5(m1, m2))
    F = jnp.sum(value)
    I_current = F / N
    I += rate * I_current
    return I


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

    log_priors = data["log_priors"]

    ll = jax.tree_map(
        lambda x: jnp.sum(jax.nn.logsumexp(mass_model.log_prob(x) - log_priors) - jnp.log(x.shape[0]) + jnp.log(rate)),
        data["data"],
    )

    ll = jnp.sum(jnp.asarray(jax.tree.leaves(ll)))

    expval = exp_rate(
        rate,
        pop_params={
            "alpha_m": alpha,
            "mmin": mmin,
            "mmax": mmax,
        },
    )
    return ll - expval + log_priors
