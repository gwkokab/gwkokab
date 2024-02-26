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

import numpyro
from jax import numpy as jnp
from numpyro import distributions as dist

from ..models import *
from ..utils.misc import get_key
from gwkokab.vts.utils import interpolate_hdf5


def integrate_adaptive(
    p,
    f,
    iter_max=2**16,
    iter_start=2**10,
    err_abs=None,
    err_rel=None,
):
    converged = lambda err_abs_current, err_rel_current: (err_rel_current < err_rel) and (err_abs_current < err_abs)

    samples = 0
    new_samples = iter_start
    F = 0.0
    F2 = 0.0

    while samples < iter_max:
        X = p(new_samples)
        value = f(X)
        F += jnp.sum(value)
        F2 += jnp.sum(value * value)

        samples += new_samples
        new_samples = samples

        I_current = F / samples
        err_abs_current = jnp.sqrt((F2 - F * F / samples) / (samples * (samples - 1.0)))
        err_rel_current = err_abs_current / I_current

        if converged(err_abs_current, err_rel_current):
            break

    return I_current


def intensity(m1, m2, alpha, m_min, m_max, rate):
    """
    Calculate the intensity: last two terms of equation 6 in Wysocki et al. (2019).

    Parameters:
    m1 (float): Mass 1.
    m2 (float): Mass 2.
    alpha (float): Alpha value.
    m_min (float): Minimum mass value.
    m_max (float): Maximum mass value.
    rate (float): Rate value.

    Returns:
    float: The calculated intensity. Product of the rate and population model.
    """
    return rate * jnp.exp(
        Wysocki2019MassModel(
            alpha_m=alpha,
            k=0,
            mmin=m_min,
            mmax=m_max,
        ).log_prob(jnp.stack([m1, m2]))
    )

#following is the equation 6 from Wysocki et al. (2019)
def expval_mc(alpha, m_min, m_max, rate):
    model = Wysocki2019MassModel(
        alpha_m=alpha,
        k=0,
        mmin=m_min,
        mmax=m_max,
    )

    def p(n):
        return model.sample(get_key(), sample_shape=(n,))

    def f(X): #effective volume-time
        m1 = X[:, 0]
        m2 = X[:, 1]
        print("m1",m1)
        print("m2",m2)
        return jnp.exp(interpolate_hdf5(m1, m2))
    
    I = integrate_adaptive(p, f, 
                        err_abs=1e-5, err_rel=1e-3)
    
    return rate*I
    


def model(lambda_n):

    event = len(lambda_n)
    alpha_init = dist.Uniform(-5.0, 5.0)
    mmin_init = dist.Uniform(5.0, 15.0)
    mmax_init = dist.Uniform(30.0, 100.0)
    rate_init = dist.Uniform(1, 500)
    alpha = numpyro.sample("alpha", alpha_init)
    mmin = numpyro.sample("mmin", mmin_init)
    mmax = numpyro.sample("mmax", mmax_init)
    rate = numpyro.sample("rate", rate_init)

    ll_i = 0.0

    for n in range(event):
        mass_model = Wysocki2019MassModel(
            alpha_m=alpha,
            k=0,
            mmin=mmin,
            mmax=mmax,
        )
        p = jnp.exp(mass_model.log_prob(lambda_n[n]))*rate #probability
        ll_i += jnp.log(jnp.mean(p))

    mean = expval_mc(alpha, mmin, mmax, rate)
    log_prior = 0.0
    log_pos = log_prior+ll_i-mean #log posterior

    return log_pos