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
from typing_extensions import Callable, Tuple

import jax
from jax import jit, lax, numpy as jnp, random as jrd, tree as jtr
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, Int, PRNGKeyArray, Real
from numpyro.distributions import Beta, Distribution, TruncatedNormal
from numpyro.util import is_prng_key

from ..utils.math import beta_dist_mean_variance_to_concentrations


__all__ = [
    "JointDistribution",
    "numerical_inverse_transform_sampling",
    "smoothing_kernel",
    "get_spin_magnitude_and_misalignment_dist",
    "get_default_spin_magnitude_dists",
    "get_spin_misalignment_dist",
]


class JointDistribution(Distribution):
    r"""Joint distribution of multiple marginal distributions."""

    pytree_aux_fields = ("marginal_distributions",)

    def __init__(self, *marginal_distributions: Distribution) -> None:
        r"""
        :param marginal_distributions: A sequence of marginal distributions.
        """
        self.marginal_distributions = marginal_distributions
        self.shaped_values = tuple()
        batch_shape = lax.broadcast_shapes(
            *tuple(d.batch_shape for d in self.marginal_distributions)
        )
        k = 0
        for d in self.marginal_distributions:
            if d.event_shape:
                self.shaped_values += (slice(k, k + d.event_shape[0]),)
                k += d.event_shape[0]
            else:
                self.shaped_values += (k,)
                k += 1
        super(JointDistribution, self).__init__(
            batch_shape=batch_shape,
            event_shape=(k,),
            validate_args=True,
        )

    def log_prob(self, value):
        log_probs = jtr.map(
            lambda d, v: d.log_prob(value[..., v]),
            self.marginal_distributions,
            self.shaped_values,
            is_leaf=lambda x: isinstance(x, Distribution),
        )
        log_probs = jtr.reduce(
            lambda x, y: x + y,
            log_probs,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )
        return log_probs

    def sample(self, key: PRNGKeyArray, sample_shape: tuple[int, ...] = ()):
        assert is_prng_key(key)
        keys = tuple(jrd.split(key, len(self.marginal_distributions)))
        samples = jtr.map(
            lambda d, k: d.sample(k, sample_shape).reshape(*sample_shape, -1),
            self.marginal_distributions,
            keys,
            is_leaf=lambda x: isinstance(x, Distribution),
        )
        samples = jnp.concatenate(samples, axis=-1)
        return samples


def numerical_inverse_transform_sampling(
    logpdf: Callable[[Array], Array],
    limits: Array,
    sample_shape: tuple,
    *,
    key: PRNGKeyArray,
    batch_shape: tuple = (),
    n_grid_points: Int = 1000,
) -> Array:
    """Numerical inverse transform sampling.

    :param logpdf: log of the probability density function
    :param limits: limits of the domain
    :param n_samples: number of samples
    :param seed: random seed. defaults to None
    :param n_grid_points: number of points on grid, defaults to 1000
    :param points: length-N sequence of arrays specifying the grid coordinates.
    :param values: N-dimensional array specifying the grid values.
    :return: samples from the distribution
    """
    assert is_prng_key(key)
    grid = jnp.linspace(
        jnp.full(batch_shape, limits[0]),
        jnp.full(batch_shape, limits[1]),
        n_grid_points,
    )  # 1000 grid points
    pdf = jnp.exp(logpdf(grid))  # pdf
    pdf = pdf / trapezoid(y=pdf, x=grid, axis=0)  # normalize
    cdf = jnp.cumsum(pdf, axis=0)  # cdf
    cdf = cdf / cdf[-1]  # normalize

    u = jax.random.uniform(key, sample_shape)  # uniform samples

    interp = lambda _xp, _fp: jnp.interp(x=u, xp=_xp, fp=_fp)
    if batch_shape:
        interp = jax.vmap(interp, in_axes=(1, 1))
    samples = interp(cdf, grid)  # interpolate
    return samples  # inverse transform sampling


@partial(jit, inline=True)
def smoothing_kernel(
    mass: Array | Real, mass_min: Array | Real, delta: Array | Real
) -> Array | Real:
    r"""See equation B4 in `Population Properties of Compact Objects from the
    Second LIGO-Virgo Gravitational-Wave Transient Catalog 
    <https://arxiv.org/abs/2010.14533>`_.
    
    .. math::
        S(m\mid m_{\min}, \delta) = \begin{cases}
            0 & \text{if } m < m_{\min}, \\
            \left[\displaystyle 1 + \exp\left(\frac{\delta}{m}
            +\frac{\delta}{m-\delta}\right)\right]^{-1}
            & \text{if } m_{\min} \leq m < m_{\min} + \delta, \\
            1 & \text{if } m \geq m_{\min} + \delta
        \end{cases}

    :param mass: mass of the primary black hole
    :param mass_min: minimum mass of the primary black hole
    :param delta: small mass difference
    :return: smoothing kernel value
    """
    conditions = [
        mass < mass_min,
        (mass_min <= mass) & (mass < mass_min + delta),
    ]
    choices = [
        jnp.zeros_like(mass),
        jnp.reciprocal(
            1
            + jnp.exp((delta / (mass - mass_min)) + (delta / (mass - mass_min - delta)))
        ),
    ]
    return jnp.select(conditions, choices, default=jnp.ones_like(mass))


def get_default_spin_magnitude_dists(
    mean_chi1,
    variance_chi1,
    mean_chi2,
    variance_chi2,
):
    concentrations_chi1 = beta_dist_mean_variance_to_concentrations(
        mean_chi1, variance_chi1
    )
    concentrations_chi2 = beta_dist_mean_variance_to_concentrations(
        mean_chi2, variance_chi2
    )
    chi1_dist = Beta(
        *concentrations_chi1,
        validate_args=True,
    )
    chi2_dist = Beta(
        *concentrations_chi2,
        validate_args=True,
    )
    return chi1_dist, chi2_dist


def get_spin_misalignment_dist(
    mean_tilt_1, std_dev_tilt_1, mean_tilt_2, std_dev_tilt_2
) -> Tuple[TruncatedNormal]:
    cos_tilt1_dist = TruncatedNormal(
        loc=mean_tilt_1,
        scale=std_dev_tilt_1,
        low=-1,
        high=1,
        validate_args=True,
    )
    cos_tilt2_dist = TruncatedNormal(
        loc=mean_tilt_2,
        scale=std_dev_tilt_2,
        low=-1,
        high=1,
        validate_args=True,
    )
    return cos_tilt1_dist, cos_tilt2_dist


def get_spin_magnitude_and_misalignment_dist(
    mean_chi1,
    variance_chi1,
    mean_chi2,
    variance_chi2,
    mean_tilt_1,
    std_dev_tilt_1,
    mean_tilt_2,
    std_dev_tilt_2,
) -> Tuple[Distribution]:
    r"""This is a helper function to reduce the lines of code in the
    :func:`MultiSpinModel` and :func:`MultiSourceModel`.
    """
    concentrations_chi1 = beta_dist_mean_variance_to_concentrations(
        mean_chi1, variance_chi1
    )
    concentrations_chi2 = beta_dist_mean_variance_to_concentrations(
        mean_chi2, variance_chi2
    )
    chi1_dist = Beta(
        *concentrations_chi1,
        validate_args=True,
    )
    chi2_dist = Beta(
        *concentrations_chi2,
        validate_args=True,
    )
    cos_tilt1_dist = TruncatedNormal(
        loc=mean_tilt_1,
        scale=std_dev_tilt_1,
        low=-1,
        high=1,
        validate_args=True,
    )
    cos_tilt2_dist = TruncatedNormal(
        loc=mean_tilt_2,
        scale=std_dev_tilt_2,
        low=-1,
        high=1,
        validate_args=True,
    )
    return chi1_dist, chi2_dist, cos_tilt1_dist, cos_tilt2_dist
