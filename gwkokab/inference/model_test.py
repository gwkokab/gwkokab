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

from jax import numpy as jnp

from ..models.wysocki2019massmodel import Wysocki2019MassModel
from ..utils.misc import get_key


def integrate_adaptive(
    p,
    f,
    iter_max=2**16,
    iter_start=2**10,
    err_abs=None,
    err_rel=None,
):
    if err_abs is None and err_rel is None:

        def converged(err_abs_current, err_rel_current):
            return False

    elif err_abs is None:

        def converged(err_abs_current, err_rel_current):
            return err_rel_current < err_rel

    elif err_rel is None:

        def converged(err_abs_current, err_rel_current):
            return err_abs_current < err_abs

    else:

        def converged(err_abs_current, err_rel_current):
            return (err_rel_current < err_rel) and (err_abs_current < err_abs)

    # Initialize samples and integral accumulators.
    samples = 0
    new_samples = iter_start
    F = 0.0
    F2 = 0.0

    # Iteratively improve MC integral, until either the convergence criteria
    # set by ``err_abs`` and ``err_rel`` are reached, or the maximum number of
    # iterations is reached.
    while samples < iter_max:
        # Draw samples from the probability distribution function.
        X = p(new_samples)

        # Evaluate the function at these samples.
        # Accumulate the sum of the result in F and the squared result in F2,
        # which will be used to calculate the integral and its uncertainty.
        value = f(X)
        F += jnp.sum(value)
        F2 += jnp.sum(value * value)

        # Update the number of samples appropriately.
        # If more samples are needed to converge, the number will be doubled.
        samples += new_samples
        new_samples = samples

        # Estimate the integral as the sample average of f(X) over all
        # iterations.
        I_current = F / samples
        # Estimate the absolute error (standard error) and the relative error
        # (standard error divided by the integral).
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
            Mmax=2 * m_max,
        ).log_prob(jnp.stack([m1, m2]))
    )


def expval_mc(alpha, m_min, m_max, rate, raw_interpolator):
    model = Wysocki2019MassModel(
        alpha_m=alpha,
        k=0,
        mmin=m_min,
        mmax=m_max,
        Mmax=2 * m_max,
    )

    def p(n):
        return model.sample(get_key(), sample_shape=(n,))

    def f(X):
        return jnp.exp(raw_interpolator(X))

    return rate * integrate_adaptive(p, f, err_abs=1e-3, err_rel=1e-3)
