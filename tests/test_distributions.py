# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# source: https://github.com/pyro-ppl/numpyro/blob/3cde93d0f25490b9b90c1c423816c6cfd9ea23ed/test/test_distributions.py

import inspect
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import numpyro.distributions as dist
import pytest
import scipy.stats as osp
from jax import lax, vmap
from numpy.testing import assert_allclose
from numpyro.distributions import constraints
from numpyro.distributions.batch_util import vmap_over
from numpyro.distributions.gof import auto_goodness_of_fit, InvalidTest
from numpyro.distributions.transforms import biject_to
from numpyro.distributions.util import multinomial, signed_stick_breaking_tril
from scipy.sparse import csr_matrix

from gwkokab.models import (
    BetaFromMeanVar,
    NPowerlawMGaussian,
    PowerlawPrimaryMassRatio,
    PowerlawRedshift,
    Wysocki2019MassModel,
)
from gwkokab.models.constraints import (
    decreasing_vector,
    increasing_vector,
    mass_ratio_mass_sandwich,
    mass_sandwich,
    positive_decreasing_vector,
    positive_increasing_vector,
    strictly_decreasing_vector,
    strictly_increasing_vector,
)
from gwkokab.models.utils import ScaledMixture


TEST_FAILURE_RATE = 2e-5  # For all goodness-of-fit tests.


def my_kron(A, B):
    D = A[..., :, None, :, None] * B[..., None, :, None, :]
    ds = D.shape
    newshape = (*ds[:-4], ds[-4] * ds[-3], ds[-2] * ds[-1])
    return D.reshape(newshape)


generic_nspmsg = {
    ## rates
    "log_rate_0": -2.0,
    "log_rate_1": -1.0,
    "log_rate_2": 1.5,
    "log_rate_3": 1.0,
    ## powerlaw 0
    "alpha_pl_0": 2.0,
    "beta_pl_0": 3.0,
    "mmin_pl_0": 50.0,
    "mmax_pl_0": 70.0,
    "delta_pl_0": 5,
    ## powerlaw 1
    "alpha_pl_1": -1.5,
    "beta_pl_1": -1.0,
    "mmin_pl_1": 20.0,
    "mmax_pl_1": 100.0,
    "delta_pl_1": 20,
    ## gaussian 0
    "loc_g_0": 70.0,
    "scale_g_0": 2.1,
    "beta_g_0": 3.2,
    "mmin_g_0": 50.0,
    "mmax_g_0": 180.0,
    "delta_g_0": 5,
    ## gaussian 1
    "loc_g_1": 80.0,
    "scale_g_1": 1.1,
    "beta_g_1": 2.2,
    "mmin_g_1": 20.0,
    "mmax_g_1": 180.0,
    "delta_g_1": 5,
    # "use_spin": True,
    "chi1_mean_g": 0.5,
    "chi1_mean_pl": 0.7,
    "chi2_mean_g": 0.2,
    "chi2_mean_pl": 0.6,
    "chi1_variance_g": 0.1,
    "chi1_variance_pl": 0.2,
    "chi2_variance_g": 0.14,
    "chi2_variance_pl": 0.1,
    # "use_tilt": True,
    "cos_tilt_zeta_g_0": 0.5,
    "cos_tilt_zeta_g_1": 0.5,
    "cos_tilt_zeta_pl_0": 0.5,
    "cos_tilt_zeta_pl_1": 0.5,
    "cos_tilt_1_scale_g_0": 0.1,
    "cos_tilt_1_scale_g_1": 0.3,
    "cos_tilt_1_scale_pl_0": 0.1,
    "cos_tilt_1_scale_pl_1": 0.3,
    "cos_tilt_2_scale_g_0": 0.1,
    "cos_tilt_2_scale_g_1": 0.3,
    "cos_tilt_2_scale_pl_0": 0.1,
    "cos_tilt_2_scale_pl_1": 0.3,
    # "use_eccentricity": True,
    "eccentricity_loc_pl_0": 0.1,
    "eccentricity_scale_pl_0": 0.2,
    "eccentricity_loc_pl_1": 0.1,
    "eccentricity_scale_pl_1": 0.4,
    "eccentricity_loc_g_0": 0.1,
    "eccentricity_scale_g_0": 0.7,
    "eccentricity_loc_g_1": 0.1,
    "eccentricity_scale_g_1": 0.6,
    "eccentricity_low_pl_0": 0.0,
    "eccentricity_low_g_0": 0.0,
    "eccentricity_low_pl_1": 0.0,
    "eccentricity_low_g_1": 0.0,
    "eccentricity_high_pl_0": 1.0,
    "eccentricity_high_g_0": 1.0,
    "eccentricity_high_pl_1": 1.0,
    "eccentricity_high_g_1": 1.0,
}


generic_npmg = {
    ## rates
    "log_rate_0": -2.0,
    "log_rate_1": -1.0,
    "log_rate_2": 1.5,
    "log_rate_3": 1.0,
    ## powerlaw 0
    "alpha_pl_0": 2.0,
    "beta_pl_0": 3.0,
    "mmin_pl_0": 50.0,
    "mmax_pl_0": 70.0,
    "delta_pl_0": 5,
    ## powerlaw 1
    "alpha_pl_1": -1.5,
    "beta_pl_1": -1.0,
    "mmin_pl_1": 20.0,
    "mmax_pl_1": 100.0,
    "delta_pl_1": 20,
    ## gaussian 0
    "m1_loc_g_0": 70.0,
    "m2_loc_g_0": 30.0,
    "m1_scale_g_0": 2.1,
    "m2_scale_g_0": 3.2,
    "m1_low_g_0": 10.0,
    "m1_high_g_0": 180.0,
    "m2_low_g_0": 10.0,
    "m2_high_g_0": 180.0,
    ## gaussian 1
    "m1_loc_g_1": 80.0,
    "m2_loc_g_1": 20.0,
    "m1_scale_g_1": 1.1,
    "m2_scale_g_1": 2.2,
    "m1_low_g_1": 10.0,
    "m1_high_g_1": 180.0,
    "m2_low_g_1": 10.0,
    "m2_high_g_1": 180.0,
    # "use_spin": True,
    "chi1_mean_g": 0.5,
    "chi1_mean_pl": 0.7,
    "chi2_mean_g": 0.2,
    "chi2_mean_pl": 0.6,
    "chi1_variance_g": 0.1,
    "chi1_variance_pl": 0.2,
    "chi2_variance_g": 0.14,
    "chi2_variance_pl": 0.1,
    # "use_tilt": True,
    "cos_tilt_zeta_g_0": 0.5,
    "cos_tilt_zeta_g_1": 0.5,
    "cos_tilt_zeta_pl_0": 0.5,
    "cos_tilt_zeta_pl_1": 0.5,
    "cos_tilt_1_scale_g_0": 0.1,
    "cos_tilt_1_scale_g_1": 0.3,
    "cos_tilt_1_scale_pl_0": 0.1,
    "cos_tilt_1_scale_pl_1": 0.3,
    "cos_tilt_2_scale_g_0": 0.1,
    "cos_tilt_2_scale_g_1": 0.3,
    "cos_tilt_2_scale_pl_0": 0.1,
    "cos_tilt_2_scale_pl_1": 0.3,
    # "use_eccentricity": True,
    "eccentricity_loc_pl_0": 0.1,
    "eccentricity_scale_pl_0": 0.2,
    "eccentricity_loc_pl_1": 0.1,
    "eccentricity_scale_pl_1": 0.4,
    "eccentricity_loc_g_0": 0.1,
    "eccentricity_scale_g_0": 0.7,
    "eccentricity_loc_g_1": 0.1,
    "eccentricity_scale_g_1": 0.6,
    "eccentricity_low_pl_0": 0.0,
    "eccentricity_low_g_0": 0.0,
    "eccentricity_low_pl_1": 0.0,
    "eccentricity_low_g_1": 0.0,
    "eccentricity_high_pl_0": 1.0,
    "eccentricity_high_g_0": 1.0,
    "eccentricity_high_pl_1": 1.0,
    "eccentricity_high_g_1": 1.0,
}


CONTINUOUS = [
    (
        PowerlawPrimaryMassRatio,
        {"alpha": -1.0, "beta": 1.0, "mmin": 10.0, "mmax": 50.0},
    ),
    (
        PowerlawPrimaryMassRatio,
        {"alpha": -0.1, "beta": -8.0, "mmin": 70.0, "mmax": 100.0},
    ),
    (
        PowerlawPrimaryMassRatio,
        {"alpha": -1.4, "beta": 9.0, "mmin": 5.0, "mmax": 100.0},
    ),
    (PowerlawPrimaryMassRatio, {"alpha": 2.0, "beta": 3.0, "mmin": 50.0, "mmax": 70.0}),
    (Wysocki2019MassModel, {"alpha_m": -1.0, "mmin": 10.0, "mmax": 50.0}),
    (Wysocki2019MassModel, {"alpha_m": -2.3, "mmin": 5.0, "mmax": 100.0}),
    (Wysocki2019MassModel, {"alpha_m": 0.7, "mmin": 50.0, "mmax": 100.0}),
    (Wysocki2019MassModel, {"alpha_m": 3.1, "mmin": 70.0, "mmax": 100.0}),
    ######### NPowerlawMGaussian (m1, m2) #########
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 0, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 0, "N_g": 1, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 1, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 2, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 2, "N_g": 2, **generic_npmg}),
    ######### NPowerlawMGaussian (m1, m2, chi1, chi2) #########
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 0, "use_spin": True, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 0, "N_g": 1, "use_spin": True, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 1, "use_spin": True, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 2, "use_spin": True, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 2, "N_g": 2, "use_spin": True, **generic_npmg}),
    ######### NPowerlawMGaussian (m1, m2, cos_tilt_1, cos_tilt_2) #########
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 0, "use_tilt": True, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 0, "N_g": 1, "use_tilt": True, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 1, "use_tilt": True, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 1, "N_g": 2, "use_tilt": True, **generic_npmg}),
    (NPowerlawMGaussian, {"N_pl": 2, "N_g": 2, "use_tilt": True, **generic_npmg}),
    ######### NPowerlawMGaussian (m1, m2, ecc) #########
    (
        NPowerlawMGaussian,
        {"N_pl": 1, "N_g": 0, "use_eccentricity": True, **generic_npmg},
    ),
    (
        NPowerlawMGaussian,
        {"N_pl": 0, "N_g": 1, "use_eccentricity": True, **generic_npmg},
    ),
    (
        NPowerlawMGaussian,
        {"N_pl": 1, "N_g": 1, "use_eccentricity": True, **generic_npmg},
    ),
    (
        NPowerlawMGaussian,
        {"N_pl": 1, "N_g": 2, "use_eccentricity": True, **generic_npmg},
    ),
    (
        NPowerlawMGaussian,
        {"N_pl": 2, "N_g": 2, "use_eccentricity": True, **generic_npmg},
    ),
    ######### NPowerlawMGaussian (m1, m2, chi1, chi2, cos_tilt_1, cos_tilt_2, ecc) #########
    (
        NPowerlawMGaussian,
        {
            "N_pl": 1,
            "N_g": 0,
            "use_spin": True,
            "use_tilt": True,
            "use_eccentricity": True,
            **generic_npmg,
        },
    ),
    (
        NPowerlawMGaussian,
        {
            "N_pl": 0,
            "N_g": 1,
            "use_spin": True,
            "use_tilt": True,
            "use_eccentricity": True,
            **generic_npmg,
        },
    ),
    (
        NPowerlawMGaussian,
        {
            "N_pl": 1,
            "N_g": 1,
            "use_spin": True,
            "use_tilt": True,
            "use_eccentricity": True,
            **generic_npmg,
        },
    ),
    (
        NPowerlawMGaussian,
        {
            "N_pl": 1,
            "N_g": 2,
            "use_spin": True,
            "use_tilt": True,
            "use_eccentricity": True,
            **generic_npmg,
        },
    ),
    (
        NPowerlawMGaussian,
        {
            "N_pl": 2,
            "N_g": 2,
            "use_spin": True,
            "use_tilt": True,
            "use_eccentricity": True,
            **generic_npmg,
        },
    ),
    (PowerlawRedshift, {"kappa": 0.0, "z_max": 1.0}),
    (PowerlawRedshift, {"kappa": 1.0, "z_max": 2.3}),
    (PowerlawRedshift, {"kappa": 2.7, "z_max": 1.0}),
    (PowerlawRedshift, {"kappa": 0.0, "z_max": 2.3}),
    (BetaFromMeanVar, {"mean": 0.4, "variance": 0.02}),
    (BetaFromMeanVar, {"mean": 0.5, "variance": 0.05}),
]


def gen_values_within_bounds(constraint, size, key=jrd.PRNGKey(11)):
    eps = 1e-6

    if constraint is constraints.boolean:
        return jrd.bernoulli(key, shape=size)
    elif isinstance(constraint, constraints.greater_than):
        return jnp.exp(jrd.normal(key, size)) + constraint.lower_bound + eps
    elif isinstance(constraint, constraints.less_than):
        return constraint.upper_bound - jnp.exp(jrd.normal(key, size)) - eps
    elif constraint == constraints.positive:
        return jnp.exp(jrd.normal(key, size))
    elif isinstance(constraint, constraints.integer_interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return jrd.randint(key, size, lower_bound, upper_bound + 1)
    elif isinstance(constraint, constraints.integer_greater_than):
        return constraint.lower_bound + jrd.poisson(key, np.array(5), shape=size)
    elif isinstance(constraint, constraints.interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return jrd.uniform(key, size, minval=lower_bound, maxval=upper_bound)
    elif constraint in (constraints.real, constraints.real_vector):
        return jrd.normal(key, size)
    elif constraint is constraints.simplex:
        return osp.dirichlet.rvs(alpha=jnp.ones((size[-1],)), size=size[:-1])
    elif isinstance(constraint, constraints.multinomial):
        n = size[-1]
        return multinomial(
            key, p=jnp.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1]
        )
    elif constraint is constraints.corr_cholesky:
        return signed_stick_breaking_tril(
            jrd.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
    elif constraint is constraints.corr_matrix:
        cholesky = signed_stick_breaking_tril(
            jrd.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
        return jnp.matmul(cholesky, jnp.swapaxes(cholesky, -2, -1))
    elif constraint is constraints.lower_cholesky:
        return jnp.tril(jrd.uniform(key, size))
    elif constraint is constraints.positive_definite:
        x = jrd.normal(key, size)
        return jnp.matmul(x, jnp.swapaxes(x, -2, -1))
    elif isinstance(
        constraint,
        (
            constraints.ordered_vector,
            increasing_vector,
            positive_increasing_vector,
            strictly_increasing_vector,
        ),
    ):
        x = jnp.cumsum(jrd.exponential(key, size), -1)
        return x - jrd.normal(key, size[:-1] + (1,))
    elif isinstance(
        constraint,
        (
            decreasing_vector,
            positive_decreasing_vector,
            strictly_decreasing_vector,
        ),
    ):
        x = jnp.cumsum(jrd.exponential(key, size), -1)
        return x[..., ::-1]
    elif isinstance(constraint, constraints.independent):
        return gen_values_within_bounds(constraint.base_constraint, size, key)
    elif constraint is constraints.sphere:
        x = jrd.normal(key, size)
        return x / jnp.linalg.norm(x, axis=-1)
    elif constraint is constraints.l1_ball:
        key1, key2 = jrd.split(key)
        sign = jrd.bernoulli(key1)
        bounds = [0, (-1) ** sign * 0.5]
        return jrd.uniform(key, size, float, *sorted(bounds))
    elif isinstance(constraint, constraints.zero_sum):
        x = jrd.normal(key, size)
        zero_sum_axes = tuple(i for i in range(-constraint.event_dim, 0))
        for axis in zero_sum_axes:
            x -= x.mean(axis)
        return x
    elif isinstance(constraint, mass_sandwich):
        x = jrd.uniform(
            key, size + (2,), minval=constraint.mmin, maxval=constraint.mmax
        )
        x = jnp.sort(x, axis=-1)
        return x
    elif isinstance(constraint, mass_ratio_mass_sandwich):
        x = jrd.normal(key, size)
        x = jnp.abs(x)
        x = jnp.cumsum(x, -1)
        x = x - jrd.normal(key, size[:-1] + (1,))
        x = jnp.reciprocal(x)
        x = jax.nn.sigmoid(x)
        x = x * jnp.array([constraint.mmax - constraint.mmin, 1.0]) + jnp.array(
            [constraint.mmin, 0.0]
        )
        return x
    else:
        raise NotImplementedError("{} not implemented.".format(constraint))


def gen_values_outside_bounds(constraint, size, key=jrd.PRNGKey(11)):
    if constraint is constraints.boolean:
        return jrd.bernoulli(key, shape=size) - 2
    elif isinstance(constraint, constraints.greater_than):
        return constraint.lower_bound - jnp.exp(jrd.normal(key, size))
    elif isinstance(constraint, constraints.less_than):
        return constraint.upper_bound + jnp.exp(jrd.normal(key, size))
    elif constraint == constraints.positive:
        return -jnp.exp(jrd.normal(key, size))
    elif isinstance(constraint, constraints.integer_interval):
        lower_bound = jnp.broadcast_to(constraint.lower_bound, size)
        return jrd.randint(key, size, lower_bound - 1, lower_bound)
    elif isinstance(constraint, constraints.integer_greater_than):
        return constraint.lower_bound - jrd.poisson(key, np.array(5), shape=size)
    elif isinstance(constraint, constraints.interval):
        upper_bound = jnp.broadcast_to(constraint.upper_bound, size)
        return jrd.uniform(key, size, minval=upper_bound, maxval=upper_bound + 1.0)
    elif constraint in [constraints.real, constraints.real_vector]:
        return lax.full(size, np.nan)
    elif constraint is constraints.simplex:
        return osp.dirichlet.rvs(alpha=jnp.ones((size[-1],)), size=size[:-1]) + 1e-2
    elif isinstance(constraint, constraints.multinomial):
        n = size[-1]
        return (
            multinomial(
                key, p=jnp.ones((n,)) / n, n=constraint.upper_bound, shape=size[:-1]
            )
            + 1
        )
    elif constraint is constraints.corr_cholesky:
        return (
            signed_stick_breaking_tril(
                jrd.uniform(
                    key,
                    size[:-2] + (size[-1] * (size[-1] - 1) // 2,),
                    minval=-1,
                    maxval=1,
                )
            )
            + 1e-2
        )
    elif constraint is constraints.corr_matrix:
        cholesky = 1e-2 + signed_stick_breaking_tril(
            jrd.uniform(
                key, size[:-2] + (size[-1] * (size[-1] - 1) // 2,), minval=-1, maxval=1
            )
        )
        return jnp.matmul(cholesky, jnp.swapaxes(cholesky, -2, -1))
    elif constraint is constraints.lower_cholesky:
        return jrd.uniform(key, size)
    elif constraint is constraints.positive_definite:
        return jrd.normal(key, size)
    elif (
        constraint is constraints.ordered_vector
        or constraint is increasing_vector
        or constraint is positive_increasing_vector
    ):
        x = jnp.cumsum(jrd.exponential(key, size), -1)
        return x[..., ::-1]
    elif isinstance(constraint, constraints.independent):
        return gen_values_outside_bounds(constraint.base_constraint, size, key)
    elif constraint is constraints.sphere:
        x = jrd.normal(key, size)
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return 2 * x
    elif constraint is constraints.l1_ball:
        key1, key2 = jrd.split(key)
        sign = jrd.bernoulli(key1)
        bounds = [(-1) ** sign * 1.1, (-1) ** sign * 2]
        return jrd.uniform(key, size, float, *sorted(bounds))
    elif isinstance(constraint, constraints.zero_sum):
        x = jrd.normal(key, size)
        return x
    elif isinstance(constraint, (mass_ratio_mass_sandwich, mass_sandwich)):
        x = jrd.normal(key, size + (2,))
        x = -jnp.abs(x)
        return x
    elif isinstance(
        constraint, (strictly_increasing_vector, strictly_decreasing_vector)
    ):
        x = jnp.full(size, -1.0)
        return x
    elif isinstance(constraint, (decreasing_vector, positive_decreasing_vector)):
        x = jnp.cumsum(jrd.exponential(key, size), -1)
        return x
    else:
        raise NotImplementedError("{} not implemented.".format(constraint))


@pytest.mark.parametrize("jax_dist_cls, params", CONTINUOUS)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_dist_shape(jax_dist_cls, params, prepend_shape):
    if jax_dist_cls.__name__ in ("PowerlawPeak",):
        pytest.skip(reason=f"{jax_dist_cls.__name__} does not provide sample method")
    if isinstance(jax_dist_cls, types.FunctionType):
        if jax_dist_cls.__name__ in ("NSmoothedPowerlawMSmoothedGaussian",):
            pytest.skip(
                reason=f"{jax_dist_cls.__name__} does not provide sample method"
            )
    jax_dist = jax_dist_cls(**params)
    rng_key = jrd.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape + jax_dist.event_shape
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    assert jnp.shape(samples) == expected_shape


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
def test_has_rsample(jax_dist, params):
    if isinstance(jax_dist, types.FunctionType):
        pytest.skip("skip testing for non-distribution")
    jax_dist = jax_dist(**params)
    masked_dist = jax_dist.mask(False)
    indept_dist = jax_dist.expand_by([2]).to_event(1)
    transf_dist = dist.TransformedDistribution(jax_dist, biject_to(constraints.real))
    assert masked_dist.has_rsample == jax_dist.has_rsample
    assert indept_dist.has_rsample == jax_dist.has_rsample
    assert transf_dist.has_rsample == jax_dist.has_rsample

    if jax_dist.has_rsample:
        assert not jax_dist.is_discrete
        if isinstance(jax_dist, dist.TransformedDistribution):
            assert jax_dist.base_dist.has_rsample
        else:
            assert set(jax_dist.arg_constraints) == set(jax_dist.reparametrized_params)
        jax_dist.rsample(jrd.PRNGKey(0))
    else:
        with pytest.raises(NotImplementedError):
            jax_dist.rsample(jrd.PRNGKey(0))


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
def test_sample_gradient(jax_dist, params):
    if jax_dist.__name__ in ("PowerlawPeak",):
        pytest.skip(reason=f"{jax_dist.__name__} does not provide sample method")
    if jax_dist.__name__ in ("PowerlawRedshift",):
        pytest.xfail(
            reason=f"{jax_dist.__name__} uses interpolation and is not differentiable"
        )
    if isinstance(jax_dist, types.FunctionType):
        if jax_dist.__name__ in ("NSmoothedPowerlawMSmoothedGaussian",):
            pytest.skip(reason=f"{jax_dist.__name__} does not provide sample method")
    jax_class = type(jax_dist(**params))
    reparametrized_params = [p for p in jax_class.reparametrized_params]
    if not reparametrized_params:
        pytest.skip("{} not reparametrized.".format(jax_class.__name__))

    nonrepara_params_dict = {
        k: v for k, v in params.items() if k not in reparametrized_params
    }
    repara_params = tuple(v for k, v in params.items() if k in reparametrized_params)

    rng_key = jrd.PRNGKey(0)

    def fn(args):
        args_dict = dict(zip(reparametrized_params, args))
        return jnp.sum(
            jax_dist(**args_dict, **nonrepara_params_dict).sample(key=rng_key)
        )

    actual_grad = jax.grad(fn)(repara_params)
    assert len(actual_grad) == len(repara_params)

    eps = 1e-3
    for i in range(len(repara_params)):
        if repara_params[i] is None:
            continue
        args_lhs = [p if j != i else p - eps for j, p in enumerate(repara_params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(repara_params)]
        fn_lhs = fn(args_lhs)
        fn_rhs = fn(args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2.0 * eps)
        assert jnp.shape(actual_grad[i]) == jnp.shape(repara_params[i])
        assert_allclose(jnp.sum(actual_grad[i]), expected_grad, rtol=0.02, atol=0.03)


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
def test_jit_log_likelihood(jax_dist, params):
    if jax_dist.__name__ in (
        "EulerMaruyama",
        "GaussianRandomWalk",
        "_ImproperWrapper",
        "LKJ",
        "LKJCholesky",
        "_SparseCAR",
        "ZeroSumNormal",
        # "NPowerlawMGaussian",
    ):
        pytest.xfail(reason="non-jittable params")

    if jax_dist.__name__ in ("PowerlawPeak",):
        pytest.skip(reason=f"{jax_dist.__name__} does not provide sample method")
    if isinstance(jax_dist, types.FunctionType):
        if jax_dist.__name__ in ("NSmoothedPowerlawMSmoothedGaussian",):
            pytest.skip(reason=f"{jax_dist.__name__} does not provide sample method")
        if jax_dist.__name__ in ("NPowerlawMGaussian",):
            pytest.xfail(reason=f"{jax_dist.__name__} have shape broadcasting issues")

    rng_key = jrd.PRNGKey(0)
    samples = jax_dist(**params).sample(key=rng_key, sample_shape=(5,))

    def log_likelihood(**params):
        return jax_dist(**params).log_prob(samples)

    expected = log_likelihood(**params)
    actual = eqx.filter_jit(log_likelihood)(**params)
    assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
def test_entropy_samples(jax_dist, params):
    jax_dist = jax_dist(**params)

    try:
        actual = jax_dist.entropy()
    except NotImplementedError:
        pytest.skip(reason=f"distribution {jax_dist} does not implement `entropy`")

    samples = jax_dist.sample(jrd.PRNGKey(8), (1000,))
    neg_log_probs = -jax_dist.log_prob(samples)
    mean = neg_log_probs.mean(axis=0)
    stderr = neg_log_probs.std(axis=0) / jnp.sqrt(neg_log_probs.shape[-1] - 1)
    z = (actual - mean) / stderr

    # Check the z-score is small or that all values are close. This happens, for
    # example, for uniform distributions with constant log prob and hence zero stderr.
    assert (jnp.abs(z) < 5).all() or jnp.allclose(actual, neg_log_probs, atol=1e-5)


@pytest.mark.parametrize(
    "jax_dist, params",
    # TODO: add more complete pattern for Discrete.cdf
    CONTINUOUS,
)
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_cdf_and_icdf(jax_dist, params):
    d = jax_dist(**params)
    if d.event_dim > 0:
        pytest.skip("skip testing cdf/icdf methods of multivariate distributions")
    key1, key2 = jrd.split(jrd.PRNGKey(0))
    samples = d.sample(key=key1, sample_shape=(100,))
    quantiles = jrd.uniform(key2, (100,) + d.shape())
    try:
        atol = 1e-5
        rtol = 1e-5
        if jax_dist.__name__ in ["PowerlawRedshift"]:
            atol = 4e-3
            rtol = 0.02
        if d.shape() == () and not d.is_discrete:
            assert_allclose(
                jax.vmap(jax.grad(d.cdf))(samples),
                jnp.exp(d.log_prob(samples)),
                atol=atol,
                rtol=rtol,
            )
            assert_allclose(
                jax.vmap(jax.grad(d.icdf))(quantiles),
                jnp.exp(-d.log_prob(d.icdf(quantiles))),
                atol=atol,
                rtol=rtol,
            )
        assert_allclose(d.cdf(d.icdf(quantiles)), quantiles, atol=atol, rtol=rtol)
        assert_allclose(d.icdf(d.cdf(samples)), samples, atol=atol, rtol=rtol)
    except NotImplementedError:
        pytest.skip("cdf/icdf not implemented")


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
def test_gof(jax_dist, params):
    if isinstance(jax_dist, types.FunctionType):
        pytest.skip("skip testing for non-distribution")
    if jax_dist.__name__ in ("PowerlawPrimaryMassRatio",):
        pytest.skip("Failure rate is lower than expected.")
    if isinstance(jax_dist, ScaledMixture):
        pytest.skip("skip testing for ScaledMixture")
    if jax_dist.__name__ in ("PowerlawRedshift",):
        pytest.skip(f"{jax_dist.__name__} is not a valid probability distribution")
    num_samples = 10000
    rng_key = jrd.PRNGKey(0)
    d = jax_dist(**params)
    try:
        samples = d.sample(key=rng_key, sample_shape=(num_samples,))
    except NotImplementedError:
        pytest.skip("sample method not implemented")
    probs = np.exp(d.log_prob(samples))

    dim = None

    # Test each batch independently.
    probs = probs.reshape(num_samples, -1)
    samples = samples.reshape(probs.shape + d.event_shape)
    for b in range(probs.shape[1]):
        try:
            gof = auto_goodness_of_fit(samples[:, b], probs[:, b], dim=dim)
        except InvalidTest:
            pytest.skip("expensive test")
        else:
            assert gof > TEST_FAILURE_RATE


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
def test_log_prob_gradient(jax_dist, params):
    if jax_dist.__name__ in ("PowerlawPeak",):
        pytest.skip(reason=f"{jax_dist.__name__} does not provide sample method")
    if isinstance(jax_dist, types.FunctionType):
        if jax_dist.__name__ in ("NSmoothedPowerlawMSmoothedGaussian",):
            pytest.skip(reason=f"{jax_dist.__name__} does not provide sample method")
    rng_key = jrd.PRNGKey(0)
    value = jax_dist(**params).sample(rng_key)

    if isinstance(jax_dist, dist.Distribution):

        def fn(*args):
            return jnp.sum(jax_dist(*args).log_prob(value))
    else:
        params_mapping = {name: i for i, name in enumerate(params.keys())}

        def fn(*args):
            param = {k: args[params_mapping[k]] for k in params_mapping}
            return jnp.sum(jax_dist(**param).log_prob(value))

    eps = 1e-3
    for i, k in enumerate(params.keys()):
        if jax_dist is PowerlawPrimaryMassRatio and i > 1:
            continue
        if jax_dist is Wysocki2019MassModel and i != 0:
            continue
        if (jax_dist is NPowerlawMGaussian) and any(
            [k.startswith("mmin"), k.startswith("mmax"), "low" in k, "high" in k]
        ):
            continue
        if params[k] is None or jnp.result_type(params[k]) in (jnp.int32, jnp.int64):
            continue
        if isinstance(params[k], bool):
            continue
        actual_grad = jax.grad(fn, i)(*params.values())
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params.values())]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params.values())]
        fn_lhs = fn(*args_lhs)
        fn_rhs = fn(*args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2.0 * eps)
        assert jnp.shape(actual_grad) == jnp.shape(params[k])
        assert_allclose(jnp.sum(actual_grad), expected_grad, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_distribution_constraints(jax_dist, params, prepend_shape):
    if isinstance(jax_dist, types.FunctionType):
        pytest.skip("skip testing for non-distribution")
    if isinstance(jax_dist, ScaledMixture):
        pytest.skip("skip testing for ScaledMixture")
    if jax_dist.__name__ in ("PowerlawPrimaryMassRatio",):
        pytest.skip(f"skipping test for {jax_dist.__name__}")
    valid_params = {}
    oob_params = {}
    key = jrd.PRNGKey(1)
    dependent_constraint = False
    for name, value in params.items():
        if value is None:
            oob_params[name] = None
            valid_params[name] = None
            continue
        constraint = jax_dist.arg_constraints[name]
        if isinstance(constraint, constraints._Dependent):
            dependent_constraint = True
            break
        key, key_gen = jrd.split(key)
        oob_params[name] = gen_values_outside_bounds(
            constraint, jnp.shape(value), key_gen
        )
        valid_params[name] = gen_values_within_bounds(
            constraint, jnp.shape(value), key_gen
        )

    assert jax_dist(**oob_params)

    # Invalid parameter values throw ValueError
    if not dependent_constraint:
        with pytest.raises(ValueError):
            jax_dist(**oob_params, validate_args=True)

        with pytest.raises(ValueError):
            # test error raised under jit omnistaging
            oob_params = jax.device_get(oob_params)

            def dist_gen_fn():
                d = jax_dist(**oob_params, validate_args=True)
                return d

            eqx.filter_jit(dist_gen_fn)()

    d = jax_dist(**valid_params, validate_args=True)

    # Out of support samples throw ValueError
    oob_samples = gen_values_outside_bounds(
        d.support, size=prepend_shape + d.batch_shape + d.event_shape
    )
    with pytest.warns(UserWarning, match="Out-of-support"):
        d.log_prob(oob_samples)

    # with pytest.warns(UserWarning, match="Out-of-support"):
    #     # test warning work under jit omnistaging
    #     oob_samples = jax.device_get(oob_samples)
    #     valid_params = jax.device_get(valid_params)

    #     def log_prob_fn():
    #         d = jax_dist(**valid_params, validate_args=True)
    #         return d.log_prob(oob_samples)

    #     jax.jit(log_prob_fn)()


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("sample_shape", [(), (4,)])
def test_expand(jax_dist, params, prepend_shape, sample_shape):
    if jax_dist.__name__ in ("PowerlawPeak",):
        pytest.skip(reason=f"{jax_dist.__name__} does not provide sample method")

    if isinstance(jax_dist, types.FunctionType):
        if jax_dist.__name__ in ("NSmoothedPowerlawMSmoothedGaussian",):
            pytest.skip(reason=f"{jax_dist.__name__} does not provide sample method")
        if jax_dist.__name__ in ("NPowerlawMGaussian",):
            pytest.xfail(
                reason=f"{jax_dist.__name__} failing test cases, needs to be investigated."
            )
    jax_dist = jax_dist(**params)
    new_batch_shape = prepend_shape + jax_dist.batch_shape
    expanded_dist = jax_dist.expand(new_batch_shape)
    rng_key = jrd.PRNGKey(0)
    samples = expanded_dist.sample(rng_key, sample_shape)
    assert expanded_dist.batch_shape == new_batch_shape
    assert jnp.shape(samples) == sample_shape + new_batch_shape + jax_dist.event_shape
    assert expanded_dist.log_prob(samples).shape == sample_shape + new_batch_shape
    # test expand of expand
    assert (
        expanded_dist.expand((3,) + new_batch_shape).batch_shape
        == (3,) + new_batch_shape
    )
    # test expand error
    if prepend_shape:
        with pytest.raises(ValueError, match="Cannot broadcast distribution of shape"):
            assert expanded_dist.expand((3,) + jax_dist.batch_shape)


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
def test_dist_pytree(jax_dist, params):
    if isinstance(jax_dist, types.FunctionType):
        pytest.skip("skip testing for non-distribution")

    def f(x):
        return jax_dist(**params)

    eqx.filter_jit(f)(0)  # this test for flatten/unflatten
    lax.map(f, np.ones(3))  # this test for compatibility w.r.t. scan
    # Test that parameters do not change after flattening.
    expected_dist = f(0)
    actual_dist = eqx.filter_jit(f)(0)
    for name in expected_dist.arg_constraints:
        expected_arg = getattr(expected_dist, name)
        actual_arg = getattr(actual_dist, name)
        assert actual_arg is not None, f"arg {name} is None"
        if np.issubdtype(np.asarray(expected_arg).dtype, np.number):
            assert_allclose(actual_arg, expected_arg)
        else:
            assert (
                actual_arg.shape == expected_arg.shape
                and actual_arg.dtype == expected_arg.dtype
            )
    try:
        expected_sample = expected_dist.sample(jrd.PRNGKey(0))
        actual_sample = actual_dist.sample(jrd.PRNGKey(0))
        expected_log_prob = expected_dist.log_prob(expected_sample)
        actual_log_prob = actual_dist.log_prob(actual_sample)
        assert_allclose(actual_sample, expected_sample, rtol=1e-6)
        assert_allclose(actual_log_prob, expected_log_prob, rtol=2e-6)
    except NotImplementedError:
        pass


def _get_vmappable_dist_init_params(jax_dist):
    if jax_dist.__name__ == ("_TruncatedCauchy"):
        return [2, 3]
    elif issubclass(jax_dist, dist.Distribution):
        init_parameters = list(inspect.signature(jax_dist.__init__).parameters.keys())[
            1:
        ]
        vmap_over_parameters = list(
            inspect.signature(vmap_over.dispatch(jax_dist)).parameters.keys()
        )[1:]
        return list(
            [
                i
                for i, name in enumerate(init_parameters)
                if name in vmap_over_parameters
            ]
        )
    else:
        raise ValueError


def _allclose_or_equal(a1, a2):
    if isinstance(a1, np.ndarray):
        return np.allclose(a2, a1)
    elif isinstance(a1, jnp.ndarray):
        return jnp.allclose(a2, a1)
    elif isinstance(a1, csr_matrix):
        return np.allclose(a2.todense(), a1.todense())
    else:
        return a2 == a1 or a2 is a1


def _tree_equal(t1, t2):
    t = jax.tree.map(_allclose_or_equal, t1, t2)
    return jnp.all(jax.flatten_util.ravel_pytree(t)[0])


@pytest.mark.parametrize("jax_dist, params", CONTINUOUS)
def test_vmap_dist(jax_dist, params):
    if isinstance(jax_dist, types.FunctionType):
        pytest.skip("skip testing for non-distribution")
    if jax_dist.__name__ in ("PowerlawRedshift",):
        pytest.xfail(f"{jax_dist.__name__} has some KeyError issues")
    param_names = list(inspect.signature(jax_dist).parameters.keys())
    vmappable_param_idxs = _get_vmappable_dist_init_params(jax_dist)
    vmappable_param_idxs = vmappable_param_idxs[: len(params)]

    if len(vmappable_param_idxs) == 0:
        return

    def make_jax_dist(**params):
        return jax_dist(**params)

    def sample(d: dist.Distribution):
        return d.sample(jrd.PRNGKey(0))

    d = make_jax_dist(**params)

    in_out_axes_cases = [
        # vmap over all args
        (
            tuple(0 if i in vmappable_param_idxs else None for i in range(len(params))),
            0,
        ),
        # vmap over a single arg, out over all attributes of a distribution
        *(
            ([0 if i == idx else None for i in range(len(params))], 0)
            for idx in vmappable_param_idxs
            if params[idx] is not None
        ),
        # vmap over a single arg, out over the associated attribute of the distribution
        *(
            (
                [0 if i == idx else None for i in range(len(params))],
                vmap_over(d, **{param_names[idx]: 0}),
            )
            for idx in vmappable_param_idxs
            if params[idx] is not None
        ),
        # vmap over a single arg, axis=1, (out single attribute, axis=1)
        *(
            (
                [1 if i == idx else None for i in range(len(params))],
                vmap_over(d, **{param_names[idx]: 1}),
            )
            for idx in vmappable_param_idxs
            if isinstance(params[idx], jnp.ndarray) and jnp.array(params[idx]).ndim > 0
        ),
    ]

    for in_axes, out_axes in in_out_axes_cases:
        batched_params = [
            (
                jax.jax.tree.map(lambda x: jnp.expand_dims(x, ax), arg)
                if isinstance(ax, int)
                else arg
            )
            for arg, ax in zip(params, in_axes)
        ]
        # Recreate the jax_dist to avoid side effects coming from `d.sample`
        # triggering lazy_property computations, which, in a few cases, break
        # vmap_over's expectations regarding existing attributes to be vmapped.
        d = make_jax_dist(*params)
        batched_d = jax.vmap(make_jax_dist, in_axes=in_axes, out_axes=out_axes)(
            *batched_params
        )
        eq = vmap(lambda x, y: _tree_equal(x, y), in_axes=(out_axes, None))(
            batched_d, d
        )
        assert eq == jnp.array([True])

        samples_dist = sample(d)
        samples_batched_dist = jax.vmap(sample, in_axes=(out_axes,))(batched_d)
        assert samples_batched_dist.shape == (1, *samples_dist.shape)
