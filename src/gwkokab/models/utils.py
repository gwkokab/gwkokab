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

from typing_extensions import Callable, List, Tuple

import jax
from jax import lax, numpy as jnp, random as jrd, tree as jtr
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, Float, Int, PRNGKeyArray
from numpyro.distributions import (
    CategoricalProbs,
    Distribution,
)
from numpyro.distributions.mixtures import MixtureGeneral
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key


__all__ = [
    "doubly_truncated_power_law_cdf",
    "doubly_truncated_power_law_icdf",
    "doubly_truncated_power_law_log_prob",
    "JointDistribution",
    "numerical_inverse_transform_sampling",
    "ScaledMixture",
]


class JointDistribution(Distribution):
    r"""Joint distribution of multiple marginal distributions."""

    pytree_aux_fields = ("marginal_distributions", "shaped_values")

    def __init__(self, *marginal_distributions: Distribution) -> None:
        r"""
        :param marginal_distributions: A sequence of marginal distributions.
        """
        self.marginal_distributions = marginal_distributions
        self.shaped_values: Tuple[int | slice, ...] = tuple()
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
    n_grid_points: Int[int, ""] = 1000,
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


@jax.custom_jvp
def doubly_truncated_power_law_log_prob(x, alpha, low, high):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L427-L444
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def neq_neg1_fn():
        one_more_alpha = 1.0 + neq_neg1_alpha
        return jnp.log(
            jnp.power(x, neq_neg1_alpha)
            * (one_more_alpha)
            / (jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha))
        )

    def eq_neg1_fn():
        return -jnp.log(x) - jnp.log(jnp.log(high) - jnp.log(low))

    return jnp.where(neq_neg1_mask, neq_neg1_fn(), eq_neg1_fn())


@doubly_truncated_power_law_log_prob.defjvp
def doubly_truncated_power_law_log_prob_jvp(primals, tangents):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L446-L524
    x, alpha, low, high = primals
    x_t, alpha_t, low_t, high_t = tangents

    log_low = jnp.log(low)
    log_high = jnp.log(high)
    log_x = jnp.log(x)

    # Mask and alpha values
    delta_eq_neg1 = 10e-4
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)
    eq_neg1_alpha = jnp.where(jnp.not_equal(alpha, 0.0), alpha, -1.0)

    primal_out = doubly_truncated_power_law_log_prob(*primals)

    # Alpha tangent with approximation
    # Variable part for all values alpha unequal -1
    def alpha_tangent_variable(alpha):
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return jnp.reciprocal(one_more_alpha) + (
            low_pow_one_more_alpha * log_low - high_pow_one_more_alpha * log_high
        ) / (high_pow_one_more_alpha - low_pow_one_more_alpha)

    # Alpha tangent
    alpha_tangent = jnp.where(
        neq_neg1_mask,
        log_x + alpha_tangent_variable(neq_neg1_alpha),
        # Approximate derivate with right an lefthand approximation
        log_x
        + (
            alpha_tangent_variable(alpha - delta_eq_neg1)
            + alpha_tangent_variable(alpha + delta_eq_neg1)
        )
        * 0.5,
    )

    # High and low tangents for alpha unequal -1
    one_more_alpha = 1.0 + neq_neg1_alpha
    low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
    high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
    change_sq = jnp.square(high_pow_one_more_alpha - low_pow_one_more_alpha)
    low_tangent_neq_neg1_common = (
        jnp.square(one_more_alpha) * jnp.power(x, neq_neg1_alpha) / change_sq
    )
    low_tangent_neq_neg1 = low_tangent_neq_neg1_common * jnp.power(low, neq_neg1_alpha)
    high_tangent_neq_neg1 = low_tangent_neq_neg1_common * jnp.power(
        high, neq_neg1_alpha
    )

    # High and low tangents for alpha equal -1
    low_tangent_eq_neg1_common = jnp.power(x, eq_neg1_alpha) / jnp.square(
        log_high - log_low
    )
    low_tangent_eq_neg1 = low_tangent_eq_neg1_common / low
    high_tangent_eq_neg1 = -low_tangent_eq_neg1_common / high

    # High and low tangents
    low_tangent = jnp.where(neq_neg1_mask, low_tangent_neq_neg1, low_tangent_eq_neg1)
    high_tangent = jnp.where(neq_neg1_mask, high_tangent_neq_neg1, high_tangent_eq_neg1)

    # Final tangents
    tangent_out = (
        alpha / x * x_t
        + alpha_tangent * alpha_t
        + low_tangent * low_t
        + high_tangent * high_t
    )
    return primal_out, tangent_out


@jax.custom_jvp
def doubly_truncated_power_law_cdf(x, alpha, low, high):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L545-L565
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def cdf_when_alpha_neq_neg1():
        one_more_alpha = 1.0 + neq_neg1_alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        return (jnp.power(x, one_more_alpha) - low_pow_one_more_alpha) / (
            jnp.power(high, one_more_alpha) - low_pow_one_more_alpha
        )

    def cdf_when_alpha_eq_neg1():
        return jnp.log(x / low) / jnp.log(high / low)

    cdf_val = jnp.where(
        neq_neg1_mask,
        cdf_when_alpha_neq_neg1(),
        cdf_when_alpha_eq_neg1(),
    )
    return jnp.clip(cdf_val, 0.0, 1.0)


@doubly_truncated_power_law_cdf.defjvp
def doubly_truncated_power_law_cdf_jvp(primals, tangents):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L567-L661
    x, alpha, low, high = primals
    x_t, alpha_t, low_t, high_t = tangents

    log_low = jnp.log(low)
    log_high = jnp.log(high)
    log_x = jnp.log(x)

    delta_eq_neg1 = 10e-4
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    # Calculate primal
    primal_out = doubly_truncated_power_law_cdf(*primals)

    # Tangents for alpha not equals -1
    def x_neq_neg1(alpha):
        one_more_alpha = 1.0 + alpha
        return (one_more_alpha * jnp.power(x, alpha)) / (
            jnp.power(high, one_more_alpha) - jnp.power(low, one_more_alpha)
        )

    def alpha_neq_neg1(alpha):
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
        term1 = (x_pow_one_more_alpha * log_x - low_pow_one_more_alpha * log_low) / (
            high_pow_one_more_alpha - low_pow_one_more_alpha
        )
        term2 = (
            (x_pow_one_more_alpha - low_pow_one_more_alpha)
            * (high_pow_one_more_alpha * log_high - low_pow_one_more_alpha * log_low)
        ) / jnp.square(high_pow_one_more_alpha - low_pow_one_more_alpha)
        return term1 - term2

    def low_neq_neg1(alpha):
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
        change = high_pow_one_more_alpha - low_pow_one_more_alpha
        term2 = one_more_alpha * jnp.power(low, alpha) / change
        term1 = term2 * (x_pow_one_more_alpha - low_pow_one_more_alpha) / change
        return term1 - term2

    def high_neq_neg1(alpha):
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        x_pow_one_more_alpha = jnp.power(x, one_more_alpha)
        return -(
            one_more_alpha
            * jnp.power(high, alpha)
            * (x_pow_one_more_alpha - low_pow_one_more_alpha)
        ) / jnp.square(high_pow_one_more_alpha - low_pow_one_more_alpha)

    # Tangents for alpha equals -1
    def x_eq_neg1():
        return jnp.reciprocal(x * (log_high - log_low))

    def low_eq_neg1():
        return (log_x - log_low) / (
            jnp.square(log_high - log_low) * low
        ) - jnp.reciprocal((log_high - log_low) * low)

    def high_eq_neg1():
        return (log_x - log_low) / (jnp.square(log_high - log_low) * high)

    # Including approximation for alpha = -1
    tangent_out = (
        jnp.where(neq_neg1_mask, x_neq_neg1(neq_neg1_alpha), x_eq_neg1()) * x_t
        + jnp.where(
            neq_neg1_mask,
            alpha_neq_neg1(neq_neg1_alpha),
            (
                alpha_neq_neg1(alpha - delta_eq_neg1)
                + alpha_neq_neg1(alpha + delta_eq_neg1)
            )
            * 0.5,
        )
        * alpha_t
        + jnp.where(neq_neg1_mask, low_neq_neg1(neq_neg1_alpha), low_eq_neg1()) * low_t
        + jnp.where(neq_neg1_mask, high_neq_neg1(neq_neg1_alpha), high_eq_neg1())
        * high_t
    )

    return primal_out, tangent_out


@jax.custom_jvp
def doubly_truncated_power_law_icdf(q, alpha, low, high):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L680-L703
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    def icdf_alpha_neq_neg1():
        one_more_alpha = 1.0 + neq_neg1_alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return jnp.power(
            low_pow_one_more_alpha
            + q * (high_pow_one_more_alpha - low_pow_one_more_alpha),
            jnp.reciprocal(one_more_alpha),
        )

    def icdf_alpha_eq_neg1():
        return jnp.power(high / low, q) * low

    icdf_val = jnp.where(
        neq_neg1_mask,
        icdf_alpha_neq_neg1(),
        icdf_alpha_eq_neg1(),
    )
    return icdf_val


@doubly_truncated_power_law_icdf.defjvp
def doubly_truncated_power_law_icdf_jvp(primals, tangents):
    # source https://github.com/pyro-ppl/numpyro/blob/94f4b99710d855bea456210cf91e6e55eeac3926/numpyro/distributions/truncated.py#L705-L815
    x, alpha, low, high = primals
    x_t, alpha_t, low_t, high_t = tangents

    log_low = jnp.log(low)
    log_high = jnp.log(high)
    high_over_low = jnp.divide(high, low)

    delta_eq_neg1 = 10e-4
    neq_neg1_mask = jnp.not_equal(alpha, -1.0)
    neq_neg1_alpha = jnp.where(neq_neg1_mask, alpha, 0.0)

    primal_out = doubly_truncated_power_law_icdf(*primals)

    # Tangents for alpha not equal -1
    def x_neq_neg1(alpha):
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        change = high_pow_one_more_alpha - low_pow_one_more_alpha
        return (
            change
            * jnp.power(
                low_pow_one_more_alpha + x * change,
                jnp.reciprocal(one_more_alpha) - 1,
            )
        ) / one_more_alpha

    def alpha_neq_neg1(alpha):
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        factor0 = low_pow_one_more_alpha + x * (
            high_pow_one_more_alpha - low_pow_one_more_alpha
        )
        term1 = jnp.power(factor0, jnp.reciprocal(one_more_alpha))
        term2 = (
            low_pow_one_more_alpha * log_low
            + x
            * (high_pow_one_more_alpha * log_high - low_pow_one_more_alpha * log_low)
        ) / (one_more_alpha * factor0)
        term3 = jnp.log(factor0) / jnp.square(one_more_alpha)
        return term1 * (term2 - term3)

    def low_neq_neg1(alpha):
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return (
            (1.0 - x)
            * jnp.power(low, alpha)
            * jnp.power(
                low_pow_one_more_alpha
                + x * (high_pow_one_more_alpha - low_pow_one_more_alpha),
                jnp.reciprocal(one_more_alpha) - 1,
            )
        )

    def high_neq_neg1(alpha):
        one_more_alpha = 1.0 + alpha
        low_pow_one_more_alpha = jnp.power(low, one_more_alpha)
        high_pow_one_more_alpha = jnp.power(high, one_more_alpha)
        return (
            x
            * jnp.power(high, alpha)
            * jnp.power(
                low_pow_one_more_alpha
                + x * (high_pow_one_more_alpha - low_pow_one_more_alpha),
                jnp.reciprocal(one_more_alpha) - 1,
            )
        )

    # Tangents for alpha equals -1
    def dx_eq_neg1():
        return low * jnp.power(high_over_low, x) * (log_high - log_low)

    def low_eq_neg1():
        return (
            jnp.power(high_over_low, x)
            - (high * x * jnp.power(high_over_low, x - 1)) / low
        )

    def high_eq_neg1():
        return x * jnp.power(high_over_low, x - 1)

    # Including approximation for alpha = -1 \
    tangent_out = (
        jnp.where(neq_neg1_mask, x_neq_neg1(neq_neg1_alpha), dx_eq_neg1()) * x_t
        + jnp.where(
            neq_neg1_mask,
            alpha_neq_neg1(neq_neg1_alpha),
            (
                alpha_neq_neg1(alpha - delta_eq_neg1)
                + alpha_neq_neg1(alpha + delta_eq_neg1)
            )
            * 0.5,
        )
        * alpha_t
        + jnp.where(neq_neg1_mask, low_neq_neg1(neq_neg1_alpha), low_eq_neg1()) * low_t
        + jnp.where(neq_neg1_mask, high_neq_neg1(neq_neg1_alpha), high_eq_neg1())
        * high_t
    )

    return primal_out, tangent_out


class ScaledMixture(MixtureGeneral):
    r"""A finite mixture of component distributions from different families. This is
    a generalization of :class:`~numpyro.distributions.Mixture` where the component
    distributions are scaled by a set of rates.

    **Example**

    .. doctest::

       >>> import jax
       >>> import jax.random as jrd
       >>> import numpyro.distributions as dist
       >>> from gwkokab.models.utils import ScaledMixture
       >>> log_scales = jrd.uniform(jrd.PRNGKey(42), (3,), minval=0, maxval=5)
       >>> component_dists = [
       ...     dist.Normal(loc=0.0, scale=1.0),
       ...     dist.Normal(loc=-0.5, scale=0.3),
       ...     dist.Normal(loc=0.6, scale=1.2),
       ... ]
       >>> mixture = ScaledMixture(log_scales, component_dists)
       >>> mixture.sample(jax.random.PRNGKey(42)).shape
       ()
    """

    pytree_data_fields = ("_log_scales",)

    def __init__(
        self,
        log_scales: Float[Array, "..."],
        component_distributions: List[Distribution],
        *,
        support=None,
        validate_args=None,
    ):
        self._log_scales = log_scales
        normed_scales = jax.nn.softmax(log_scales)
        mixing_distribution = CategoricalProbs(
            probs=normed_scales, validate_args=validate_args
        )
        super(ScaledMixture, self).__init__(
            mixing_distribution=mixing_distribution,
            component_distributions=component_distributions,
            support=support,
            validate_args=validate_args,
        )

    @validate_sample
    def log_prob(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        # Numpyro's MixtureGeneral does not support weights that do not sum to 1.
        # To bypass this constraint we have used this mathematical jugaad to remove
        # the weights that are normalized and add back the log of rates.
        sum_log_probs = (
            self._log_scales  # Adding back the log of rates to scale each component to their rate
            + self.component_log_probs(value)  # log of the component probabilities
            - jax.nn.log_softmax(
                self._mixing_distribution.logits
            )  # removing the normalized weights
        )
        safe_sum_log_probs = jnp.where(
            jnp.isneginf(sum_log_probs), -jnp.inf, sum_log_probs
        )
        mixture_log_prob = jax.nn.logsumexp(safe_sum_log_probs, axis=-1)
        return mixture_log_prob
