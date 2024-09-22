# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# source: https://github.com/pyro-ppl/numpyro/blob/3cde93d0f25490b9b90c1c423816c6cfd9ea23ed/test/test_distributions.py

import inspect
from collections import namedtuple

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
    BrokenPowerLawMassModel,
    MultiPeakMassModel,
    PowerLawPeakMassModel,
    PowerLawPrimaryMassRatio,
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


TEST_FAILURE_RATE = 2e-5  # For all goodness-of-fit tests.


def my_kron(A, B):
    D = A[..., :, None, :, None] * B[..., None, :, None, :]
    ds = D.shape
    newshape = (*ds[:-4], ds[-4] * ds[-3], ds[-2] * ds[-1])
    return D.reshape(newshape)


def _identity(x):
    return x


class T(namedtuple("TestCase", ["jax_dist", "sp_dist", "params"])):
    def __new__(cls, jax_dist, *params):
        sp_dist = get_sp_dist(jax_dist)
        return super(cls, T).__new__(cls, jax_dist, sp_dist, params)


_DIST_MAP = {}


def get_sp_dist(jax_dist):
    classes = jax_dist.mro() if isinstance(jax_dist, type) else [jax_dist]
    for cls in classes:
        if cls in _DIST_MAP:
            return _DIST_MAP[cls]
    return None


CONTINUOUS = [
    T(PowerLawPrimaryMassRatio, -1.0, 1.0, 10.0, 50.0),
    T(PowerLawPrimaryMassRatio, -7.0, -8.0, 70.0, 100.0),
    T(PowerLawPrimaryMassRatio, -7.0, 9.0, 5.0, 100.0),
    T(PowerLawPrimaryMassRatio, 2.0, 3.0, 50.0, 100.0),
    T(Wysocki2019MassModel, -1.0, 10.0, 50.0),
    T(Wysocki2019MassModel, -2.3, 5.0, 100.0),
    T(Wysocki2019MassModel, 0.7, 50.0, 100.0),
    T(Wysocki2019MassModel, 3.1, 70.0, 100.0),
]


def _is_batched_multivariate(jax_dist):
    return len(jax_dist.event_shape) > 0 and len(jax_dist.batch_shape) > 0


def gen_values_within_bounds(constraint, size, key=jrd.PRNGKey(11)):
    eps = 1e-6

    if constraint is constraints.boolean:
        return jrd.bernoulli(key, shape=size)
    elif isinstance(constraint, constraints.greater_than):
        return jnp.exp(jrd.normal(key, size)) + constraint.lower_bound + eps
    elif isinstance(constraint, constraints.less_than):
        return constraint.upper_bound - jnp.exp(jrd.normal(key, size)) - eps
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
        x = jrd.normal(key, size)
        x = jnp.abs(x)
        x = jax.nn.sigmoid(x)
        x = jnp.sort(x, axis=-1, descending=True)
        x *= jnp.broadcast_to(constraint.mmax, size) - jnp.broadcast_to(
            constraint.mmin, size
        )
        x += jnp.broadcast_to(constraint.mmin, size)
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
        x = jrd.normal(key, size)
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


@pytest.mark.parametrize("jax_dist_cls, sp_dist, params", CONTINUOUS)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_dist_shape(jax_dist_cls, sp_dist, params, prepend_shape):
    jax_dist = jax_dist_cls(*params)
    rng_key = jrd.PRNGKey(0)
    expected_shape = prepend_shape + jax_dist.batch_shape + jax_dist.event_shape
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    assert jnp.shape(samples) == expected_shape


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_infer_shapes(jax_dist, sp_dist, params):
    shapes = []
    for param in params:
        if param is None:
            shapes.append(None)
            continue
        shape = getattr(param, "shape", ())
        if callable(shape):
            shape = shape()
        shapes.append(shape)
    jax_dist = jax_dist(*params)
    try:
        expected_batch_shape, expected_event_shape = type(jax_dist).infer_shapes(
            *shapes
        )
    except NotImplementedError:
        pytest.skip(f"{type(jax_dist).__name__}.infer_shapes() is not implemented")
    assert jax_dist.batch_shape == expected_batch_shape
    assert jax_dist.event_shape == expected_event_shape


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_has_rsample(jax_dist, sp_dist, params):
    jax_dist = jax_dist(*params)
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


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_sample_gradient(jax_dist, sp_dist, params):
    # we have pathwise gradient for gamma sampler
    gamma_derived_params = {
        "Gamma": ["concentration"],
        "Beta": ["concentration1", "concentration0"],
        "BetaProportion": ["mean", "concentration"],
        "Chi2": ["df"],
        "Dirichlet": ["concentration"],
        "InverseGamma": ["concentration"],
        "LKJ": ["concentration"],
        "LKJCholesky": ["concentration"],
        "StudentT": ["df"],
    }.get(jax_dist.__name__, [])

    dist_args = [
        p
        for p in (
            inspect.getfullargspec(jax_dist.__init__)[0][1:]
            if inspect.isclass(jax_dist)
            # account the the case jax_dist is a function
            else inspect.getfullargspec(jax_dist)[0]
        )
    ]
    params_dict = dict(zip(dist_args[: len(params)], params))

    jax_class = type(jax_dist(**params_dict))
    reparametrized_params = [
        p for p in jax_class.reparametrized_params if p not in gamma_derived_params
    ]
    if not reparametrized_params:
        pytest.skip("{} not reparametrized.".format(jax_class.__name__))

    nonrepara_params_dict = {
        k: v for k, v in params_dict.items() if k not in reparametrized_params
    }
    repara_params = tuple(
        v for k, v in params_dict.items() if k in reparametrized_params
    )

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


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_jit_log_likelihood(jax_dist, sp_dist, params):
    if jax_dist.__name__ in (
        "EulerMaruyama",
        "GaussianRandomWalk",
        "_ImproperWrapper",
        "LKJ",
        "LKJCholesky",
        "_SparseCAR",
        "ZeroSumNormal",
    ):
        pytest.xfail(reason="non-jittable params")

    rng_key = jrd.PRNGKey(0)
    samples = jax_dist(*params).sample(key=rng_key, sample_shape=(2, 3))

    def log_likelihood(*params):
        return jax_dist(*params).log_prob(samples)

    expected = log_likelihood(*params)
    actual = jax.jit(log_likelihood)(*params)
    assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("jit", [False, True])
def test_log_prob(jax_dist, sp_dist, params, prepend_shape, jit):
    jit_fn = _identity if not jit else jax.jit
    jax_dist = jax_dist(*params)

    rng_key = jrd.PRNGKey(0)
    samples = jax_dist.sample(key=rng_key, sample_shape=prepend_shape)
    assert jax_dist.log_prob(samples).shape == prepend_shape + jax_dist.batch_shape

    if sp_dist is None:
        pytest.skip("no corresponding scipy distn.")
    if _is_batched_multivariate(jax_dist):
        pytest.skip("batching not allowed in multivariate distns.")
    if jax_dist.event_shape and prepend_shape:
        # >>> d = sp.dirichlet([1.1, 1.1])
        # >>> samples = d.rvs(size=(2,))
        # >>> d.logpdf(samples)
        # ValueError: The input vector 'x' must lie within the normal simplex ...
        pytest.skip("batched samples cannot be scored by multivariate distributions.")
    sp_dist = sp_dist(*params)
    try:
        expected = sp_dist.logpdf(samples)
    except AttributeError:
        expected = sp_dist.logpmf(samples)
    except ValueError as e:
        # precision issue: jnp.sum(x / jnp.sum(x)) = 0.99999994 != 1
        if "The input vector 'x' must lie within the normal simplex." in str(e):
            samples = jax.device_get(samples).astype("float64")
            samples = samples / samples.sum(axis=-1, keepdims=True)
            expected = sp_dist.logpdf(samples)
        else:
            raise e
    assert_allclose(jit_fn(jax_dist.log_prob)(samples), expected, atol=1e-5)


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_entropy_scipy(jax_dist, sp_dist, params):
    jax_dist = jax_dist(*params)

    try:
        actual = jax_dist.entropy()
    except NotImplementedError:
        pytest.skip(reason=f"distribution {jax_dist} does not implement `entropy`")
    if _is_batched_multivariate(jax_dist):
        pytest.skip("batching not allowed in multivariate distns.")
    if sp_dist is None:
        pytest.skip(reason="no corresponding scipy distribution")

    sp_dist = sp_dist(*params)
    expected = sp_dist.entropy()
    assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_entropy_samples(jax_dist, sp_dist, params):
    jax_dist = jax_dist(*params)

    try:
        actual = jax_dist.entropy()
    except NotImplementedError:
        pytest.skip(reason=f"distribution {jax_dist} does not implement `entropy`")

    samples = jax_dist.sample(jax.jrd.key(8), (1000,))
    neg_log_probs = -jax_dist.log_prob(samples)
    mean = neg_log_probs.mean(axis=0)
    stderr = neg_log_probs.std(axis=0) / jnp.sqrt(neg_log_probs.shape[-1] - 1)
    z = (actual - mean) / stderr

    # Check the z-score is small or that all values are close. This happens, for
    # example, for uniform distributions with constant log prob and hence zero stderr.
    assert (jnp.abs(z) < 5).all() or jnp.allclose(actual, neg_log_probs, atol=1e-5)


@pytest.mark.parametrize(
    "jax_dist, sp_dist, params",
    # TODO: add more complete pattern for Discrete.cdf
    CONTINUOUS,
)
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_cdf_and_icdf(jax_dist, sp_dist, params):
    d = jax_dist(*params)
    if d.event_dim > 0:
        pytest.skip("skip testing cdf/icdf methods of multivariate distributions")
    samples = d.sample(key=jrd.PRNGKey(0), sample_shape=(100,))
    quantiles = jrd.uniform(jrd.PRNGKey(1), (100,) + d.shape())
    try:
        rtol = 1e-5
        if d.shape() == () and not d.is_discrete:
            assert_allclose(
                jax.vmap(jax.grad(d.cdf))(samples),
                jnp.exp(d.log_prob(samples)),
                atol=1e-5,
                rtol=rtol,
            )
            assert_allclose(
                jax.vmap(jax.grad(d.icdf))(quantiles),
                jnp.exp(-d.log_prob(d.icdf(quantiles))),
                atol=1e-5,
                rtol=rtol,
            )
        assert_allclose(d.cdf(d.icdf(quantiles)), quantiles, atol=1e-5, rtol=1e-5)
        assert_allclose(d.icdf(d.cdf(samples)), samples, atol=1e-5, rtol=rtol)
    except NotImplementedError:
        pytest.skip("cdf/icdf not implemented")

    # test against scipy
    if not sp_dist:
        pytest.skip("no corresponding scipy distn.")
    sp_dist = sp_dist(*params)
    try:
        actual_cdf = d.cdf(samples)
        expected_cdf = sp_dist.cdf(samples)
        assert_allclose(actual_cdf, expected_cdf, atol=1e-5, rtol=1e-5)
        actual_icdf = d.icdf(quantiles)
        expected_icdf = sp_dist.ppf(quantiles)
        assert_allclose(actual_icdf, expected_icdf, atol=1e-4, rtol=1e-4)
    except NotImplementedError:
        pytest.skip("cdf/icdf not implemented")


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_gof(jax_dist, sp_dist, params):
    num_samples = 10000
    rng_key = jrd.PRNGKey(0)
    d = jax_dist(*params)
    samples = d.sample(key=rng_key, sample_shape=(num_samples,))
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


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_log_prob_gradient(jax_dist, sp_dist, params):
    rng_key = jrd.PRNGKey(0)
    value = jax_dist(*params).sample(rng_key)

    def fn(*args):
        return jnp.sum(jax_dist(*args).log_prob(value))

    eps = 1e-3
    for i in range(len(params)):
        if isinstance(
            params[i], dist.Distribution
        ):  # skip taking grad w.r.t. base_dist
            continue
        if params[i] is None or jnp.result_type(params[i]) in (jnp.int32, jnp.int64):
            continue
        actual_grad = jax.grad(fn, i)(*params)
        args_lhs = [p if j != i else p - eps for j, p in enumerate(params)]
        args_rhs = [p if j != i else p + eps for j, p in enumerate(params)]
        fn_lhs = fn(*args_lhs)
        fn_rhs = fn(*args_rhs)
        # finite diff approximation
        expected_grad = (fn_rhs - fn_lhs) / (2.0 * eps)
        assert jnp.shape(actual_grad) == jnp.shape(params[i])
        assert_allclose(jnp.sum(actual_grad), expected_grad, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_mean_var(jax_dist, sp_dist, params):
    n = 200_000
    if isinstance(
        jax_dist, (BrokenPowerLawMassModel, MultiPeakMassModel, PowerLawPeakMassModel)
    ):
        n = 2000
    d_jax = jax_dist(*params)
    k = jrd.PRNGKey(0)
    samples = d_jax.sample(k, sample_shape=(n,)).astype(np.float32)
    # check with suitable scipy implementation if available
    # XXX: VonMises is already tested below
    if sp_dist and not _is_batched_multivariate(d_jax):
        d_sp = sp_dist(*params)
        try:
            sp_mean = d_sp.mean()
        except TypeError:  # mvn does not have .mean() method
            sp_mean = d_sp.mean
        # for multivariate distns try .cov first
        if d_jax.event_shape:
            try:
                sp_var = jnp.diag(d_sp.cov())
            except TypeError:  # mvn does not have .cov() method
                sp_var = jnp.diag(d_sp.cov)
            except (AttributeError, ValueError):
                sp_var = d_sp.var()
        else:
            sp_var = d_sp.var()
        assert_allclose(d_jax.mean, sp_mean, rtol=0.01, atol=1e-7)
        assert_allclose(d_jax.variance, sp_var, rtol=0.01, atol=1e-7)
        if jnp.all(jnp.isfinite(sp_mean)):
            assert_allclose(jnp.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
        if jnp.all(jnp.isfinite(sp_var)):
            assert_allclose(
                jnp.std(samples, 0), jnp.sqrt(d_jax.variance), rtol=0.05, atol=1e-2
            )
        else:  # unbatched
            sample_shape = (200_000,)
            # mean
            jnp.allclose(
                jnp.mean(samples, 0),
                jnp.squeeze(d_jax.mean),
                rtol=0.5,
                atol=1e-2,
            )
            # cov
            samples_mvn = jnp.squeeze(samples).reshape(sample_shape + (-1,), order="F")
            scale_tril = my_kron(
                jnp.squeeze(d_jax.scale_tril_column), jnp.squeeze(d_jax.scale_tril_row)
            )
            sample_scale_tril = jnp.linalg.cholesky(jnp.cov(samples_mvn.T))
            jnp.allclose(sample_scale_tril, scale_tril, atol=0.5, rtol=1e-2)
    else:
        try:
            if jnp.all(jnp.isfinite(d_jax.mean)):
                assert_allclose(jnp.mean(samples, 0), d_jax.mean, rtol=0.05, atol=1e-2)
            if jnp.all(jnp.isfinite(d_jax.variance)):
                assert jnp.allclose(
                    jnp.std(samples, 0), jnp.sqrt(d_jax.variance), rtol=0.05, atol=1e-2
                )
        except NotImplementedError:
            pytest.skip(
                f"mean/variance not implemented for {jax_dist.__class__.__name__}"
            )


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
@pytest.mark.parametrize("prepend_shape", [(), (2,), (2, 3)])
def test_distribution_constraints(jax_dist, sp_dist, params, prepend_shape):
    dist_args = [p for p in inspect.getfullargspec(jax_dist.__init__)[0][1:]]

    valid_params, oob_params = list(params), list(params)
    key = jrd.PRNGKey(1)
    dependent_constraint = False
    for i in range(len(params)):
        if params[i] is None:
            oob_params[i] = None
            valid_params[i] = None
            continue
        constraint = jax_dist.arg_constraints[dist_args[i]]
        if isinstance(constraint, constraints._Dependent):
            dependent_constraint = True
            break
        key, key_gen = jrd.split(key)
        oob_params[i] = gen_values_outside_bounds(
            constraint, jnp.shape(params[i]), key_gen
        )
        valid_params[i] = gen_values_within_bounds(
            constraint, jnp.shape(params[i]), key_gen
        )

    assert jax_dist(*oob_params)

    # Invalid parameter values throw ValueError
    if not dependent_constraint:
        with pytest.raises(ValueError):
            jax_dist(*oob_params, validate_args=True)

        with pytest.raises(ValueError):
            # test error raised under jit omnistaging
            oob_params = jax.device_get(oob_params)

            def dist_gen_fn():
                d = jax_dist(*oob_params, validate_args=True)
                return d

            jax.jit(dist_gen_fn)()

    d = jax_dist(*valid_params, validate_args=True)

    # Test agreement of log density evaluation on randomly generated samples
    # with scipy's implementation when available.
    if (
        sp_dist
        and not _is_batched_multivariate(d)
        and not (d.event_shape and prepend_shape)
    ):
        valid_samples = gen_values_within_bounds(
            d.support, size=prepend_shape + d.batch_shape + d.event_shape
        )
        try:
            expected = sp_dist(*valid_params).logpdf(valid_samples)
        except AttributeError:
            expected = sp_dist(*valid_params).logpmf(valid_samples)
        assert_allclose(d.log_prob(valid_samples), expected, atol=1e-5, rtol=1e-5)

    # Out of support samples throw ValueError
    oob_samples = gen_values_outside_bounds(
        d.support, size=prepend_shape + d.batch_shape + d.event_shape
    )
    with pytest.warns(UserWarning, match="Out-of-support"):
        d.log_prob(oob_samples)

    with pytest.warns(UserWarning, match="Out-of-support"):
        # test warning work under jit omnistaging
        oob_samples = jax.device_get(oob_samples)
        valid_params = jax.device_get(valid_params)

        def log_prob_fn():
            d = jax_dist(*valid_params, validate_args=True)
            return d.log_prob(oob_samples)

        jax.jit(log_prob_fn)()


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
@pytest.mark.parametrize("prepend_shape", [(), (2, 3)])
@pytest.mark.parametrize("sample_shape", [(), (4,)])
def test_expand(jax_dist, sp_dist, params, prepend_shape, sample_shape):
    jax_dist = jax_dist(*params)
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


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_dist_pytree(jax_dist, sp_dist, params):
    def f(x):
        return jax_dist(*params)

    jax.jit(f)(0)  # this test for flatten/unflatten
    lax.map(f, np.ones(3))  # this test for compatibility w.r.t. scan
    # Test that parameters do not change after flattening.
    expected_dist = f(0)
    actual_dist = jax.jit(f)(0)
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
    expected_sample = expected_dist.sample(jrd.PRNGKey(0))
    actual_sample = actual_dist.sample(jrd.PRNGKey(0))
    expected_log_prob = expected_dist.log_prob(expected_sample)
    actual_log_prob = actual_dist.log_prob(actual_sample)
    assert_allclose(actual_sample, expected_sample, rtol=1e-6)
    assert_allclose(actual_log_prob, expected_log_prob, rtol=2e-6)


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


@pytest.mark.parametrize("jax_dist, sp_dist, params", CONTINUOUS)
def test_vmap_dist(jax_dist, sp_dist, params):
    param_names = list(inspect.signature(jax_dist).parameters.keys())
    vmappable_param_idxs = _get_vmappable_dist_init_params(jax_dist)
    vmappable_param_idxs = vmappable_param_idxs[: len(params)]

    if len(vmappable_param_idxs) == 0:
        return

    def make_jax_dist(*params):
        return jax_dist(*params)

    def sample(d: dist.Distribution):
        return d.sample(jrd.PRNGKey(0))

    d = make_jax_dist(*params)

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
