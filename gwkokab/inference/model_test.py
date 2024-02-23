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
from ..vts import mass_grid_coords


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
    return rate * jnp.exp(
        Wysocki2019MassModel(
            alpha_m=alpha,
            k=0,
            mmin=m_min,
            mmax=m_max,
        ).log_prob(jnp.stack([m1, m2]))
    )


def expval_mc(alpha, m_min, m_max, rate, raw_interpolator):
    model = Wysocki2019MassModel(
        alpha_m=alpha,
        k=0,
        mmin=m_min,
        mmax=m_max,
    )

    def p(n):
        return model.sample(get_key(), sample_shape=(n,))

    def f(X):
        m1 = X[:, 0]
        m2 = X[:, 1]
        logM, qtilde = mass_grid_coords(m1, m2, m_min)
        return jnp.exp(raw_interpolator((logM, qtilde)))

    return rate * integrate_adaptive(p, f, err_abs=1e-3, err_rel=1e-3)


def model(lambda_n):  # , raw_interpolator):
    r"""
    .. math::
        \mathcal{L}(\mathcal{R},\Lambda)\propto\prod_{n=1}^{N}\frac{\mathcal{R}}{N_i}\sum_{i=1}^{N_i}\frac{p(\lambda_i/\Lambda)}{\pi(\lambda_i)}
    """
    event, _, _ = lambda_n.shape
    alpha_prior = dist.Uniform(-5.0, 5.0)
    mmin_prior = dist.Uniform(5.0, 15.0)
    mmax_prior = dist.Uniform(30.0, 190.0)
    rate_prior = dist.LogUniform(10**-1, 10**6)
    alpha = numpyro.sample("alpha", alpha_prior)
    mmin = numpyro.sample("mmin", mmin_prior)
    mmax = numpyro.sample("mmax", mmax_prior)
    rate = numpyro.sample("rate", rate_prior)

    ll = 0.0

    ll = 0.0

    with numpyro.plate("data", event) as n:
        mass_model = Wysocki2019MassModel(
            alpha_m=alpha,
            k=0,
            mmin=mmin,
            mmax=mmax,
        )
        ll_i = 0.0
        p = jnp.exp(mass_model.log_prob(lambda_n[n, :, :]))
        ll_i = jnp.mean(p, axis=1)
        ll_i *= rate
        ll += jnp.log(ll_i)

    # mean = expval_mc(alpha, mmin, mmax, rate, raw_interpolator)
    # ll -= mean

    return jnp.exp(ll)
