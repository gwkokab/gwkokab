# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import chex
import jax
import jax.random as jrd
import pytest
from absl.testing import parameterized
from jax import numpy as jnp
from numpy.testing import assert_allclose
from numpyro.distributions import (
    constraints,
    Exponential,
    LogNormal,
    MultivariateNormal,
    Normal,
    Uniform,
)

from gwkokab.models.utils._joindistribution import (
    _flatten_marginal_distributions,
    JointDistribution,
)


normal = Normal(0.0, 1.0)
uniform = Uniform(0.0, 1.0)
exponential = Exponential(1.0)
lognormal = LogNormal(0.0, 1.0)

marginal_distributions_collection = [
    # 1. Single normal
    (normal,),
    # 2. Two base distributions
    (normal, uniform),
    # 3. Joint nested inside another
    (JointDistribution(normal, JointDistribution(uniform, exponential)),),
    # 4. Left-heavy deep nesting
    (
        JointDistribution(
            JointDistribution(JointDistribution(normal, uniform), exponential),
            lognormal,
        ),
    ),
    # 5. Right-heavy deep nesting
    (
        JointDistribution(
            normal,
            JointDistribution(uniform, JointDistribution(exponential, lognormal)),
        ),
    ),
    # 6. Multiple nested JointDists at same level
    (
        JointDistribution(normal, uniform),
        JointDistribution(exponential, lognormal),
    ),
    # 7. Mix of atomic and nested joints
    (
        normal,
        JointDistribution(uniform, exponential),
        lognormal,
        JointDistribution(exponential, normal),
    ),
    # 8. Three-layer symmetric tree
    (
        JointDistribution(
            JointDistribution(normal, uniform),
            JointDistribution(exponential, lognormal),
        ),
    ),
    # 9. Deeply nested tree of only JointDistributions
    (
        JointDistribution(
            JointDistribution(
                JointDistribution(normal, uniform),
                JointDistribution(exponential, lognormal),
            ),
            JointDistribution(lognormal, exponential),
        ),
    ),
    # 10. Atomic + nested tree
    (
        lognormal,
        JointDistribution(
            JointDistribution(normal, uniform),
            JointDistribution(exponential, lognormal),
        ),
    ),
]


def test_panic_on_empty_marginal_distributions():
    with pytest.raises(
        ValueError, match="At least one marginal distribution is required."
    ):
        JointDistribution()


class TestJointDistribution(parameterized.TestCase):
    @parameterized.product(
        marginal_distributions=marginal_distributions_collection,
        flatten_method=[None, "shallow", "deep"],
    )
    def test_flatten_marginal_distributions(
        self, marginal_distributions, flatten_method
    ):
        flattened = _flatten_marginal_distributions(
            marginal_distributions, flatten_method=flatten_method
        )

        if flatten_method is None:
            expected_len = len(marginal_distributions)
            for i, dist in enumerate(flattened):
                assert dist is marginal_distributions[i], (
                    f"Expected {dist} to be {marginal_distributions[i]}"
                )

        elif flatten_method == "shallow":
            expected_len = 0
            for dist in marginal_distributions:
                if isinstance(dist, JointDistribution):
                    expected_len += len(dist.marginal_distributions)
                else:
                    expected_len += 1

        elif flatten_method == "deep":
            # recursive flattening: count all underlying distributions
            def count_all(d):
                if isinstance(d, JointDistribution):
                    return sum(count_all(x) for x in d.marginal_distributions)
                return 1

            expected_len = sum(count_all(d) for d in marginal_distributions)
            for dist in flattened:
                assert not isinstance(dist, JointDistribution), (
                    "Expected no JointDistribution in deep flattening, got: "
                    + str(dist)
                )

        assert len(flattened) == expected_len, (
            "Flattened length mismatch for method: " + str(flatten_method)
        )

    @chex.variants(  # pyright: ignore
        with_jit=True,  # test case failing
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=False,  # pmap not supported in this test
    )
    @parameterized.product(
        marginal_distributions=marginal_distributions_collection,
        flatten_method=[None, "shallow", "deep"],
    )
    def test_creation_under_jax_transforms(
        self, marginal_distributions, flatten_method
    ):
        @self.variant
        def create_joint_distribution():
            jd = JointDistribution(
                *marginal_distributions,
                flatten_method=flatten_method,
                validate_args=True,
            )
            return jd.sample(jrd.key(0))

        create_joint_distribution()


###############################################################################
# semantics: layout, density, support and sampling
###############################################################################


def _mixed_joint():
    return JointDistribution(
        Normal(0.0, 1.0),
        Uniform(0.0, 1.0),
        MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
    )


def test_event_shape_is_the_sum_of_the_marginal_widths():
    jd = _mixed_joint()
    assert jd.batch_shape == ()
    assert jd.event_shape == (4,)
    assert jd.shaped_values == (0, 1, (2, 4))


def test_log_prob_is_the_sum_of_the_marginal_log_probs():
    jd = _mixed_joint()
    value = jnp.asarray([0.5, 0.25, 1.0, -1.0])
    expected = (
        Normal(0.0, 1.0).log_prob(0.5)
        + Uniform(0.0, 1.0).log_prob(0.25)
        + MultivariateNormal(jnp.zeros(2), jnp.eye(2)).log_prob(
            jnp.asarray([1.0, -1.0])
        )
    )
    assert_allclose(jd.log_prob(value), expected, rtol=1e-12)
    assert_allclose(jd.marginal_log_probs(value).sum(axis=-1), expected, rtol=1e-12)


def test_marginal_log_probs_are_sliced_per_marginal():
    jd = _mixed_joint()
    value = jnp.asarray([0.5, 0.25, 1.0, -1.0])
    marginals = jd.marginal_log_probs(value)
    assert marginals.shape == (3,)
    assert_allclose(marginals[0], Normal(0.0, 1.0).log_prob(0.5), rtol=1e-12)
    assert_allclose(marginals[1], Uniform(0.0, 1.0).log_prob(0.25), rtol=1e-12)


@pytest.mark.parametrize("sample_shape", [(), (3,), (2, 3)])
def test_log_prob_and_sample_shapes(sample_shape):
    jd = _mixed_joint()
    samples = jd.sample(jrd.key(0), sample_shape)
    assert samples.shape == sample_shape + (4,)
    assert jd.log_prob(samples).shape == sample_shape


def test_samples_lie_in_the_support():
    jd = _mixed_joint()
    samples = jd.sample(jrd.key(1), (2048,))
    assert jnp.all(jd.support.check(samples))


def test_support_is_the_conjunction_of_the_marginal_supports():
    jd = _mixed_joint()
    inside = jnp.asarray([0.5, 0.25, 1.0, -1.0])
    outside = jnp.asarray([0.5, 1.5, 1.0, -1.0])  # the uniform slot is out of range
    assert_allclose(jd.support.check(jnp.stack([inside, outside])), [True, False])


def test_an_explicit_support_is_used_verbatim():
    jd = JointDistribution(
        normal, uniform, support=constraints.independent(constraints.real, 1)
    )
    assert jd.support == constraints.independent(constraints.real, 1)


def test_a_single_marginal_is_still_a_joint():
    jd = JointDistribution(Normal(0.0, 1.0))
    assert jd.event_shape == (1,)
    assert_allclose(
        jd.log_prob(jnp.asarray([0.5])), Normal(0.0, 1.0).log_prob(0.5), rtol=1e-12
    )


def test_rejects_an_unknown_flatten_method():
    with pytest.raises(ValueError, match="Unknown flatten method"):
        JointDistribution(normal, uniform, flatten_method="sideways")


def test_flattening_does_not_change_the_density():
    marginals = (normal, JointDistribution(uniform, JointDistribution(exponential)))
    value = jnp.asarray([0.5, 0.25, 1.0])
    densities = [
        JointDistribution(*marginals, flatten_method=method).log_prob(value)
        for method in (None, "shallow", "deep")
    ]
    assert_allclose(densities, [densities[0]] * 3, rtol=1e-12)


def test_under_jit_and_grad():
    jd = _mixed_joint()
    value = jnp.asarray([0.5, 0.25, 1.0, -1.0])
    assert_allclose(jax.jit(jd.log_prob)(value), jd.log_prob(value), rtol=1e-12)
    gradient = jax.grad(jd.log_prob)(value)
    assert_allclose(gradient, [-0.5, 0.0, -1.0, 1.0], rtol=1e-12)


def test_pytree_roundtrip():
    jd = _mixed_joint()
    value = jnp.asarray([0.5, 0.25, 1.0, -1.0])
    leaves, treedef = jax.tree_util.tree_flatten(jd)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert_allclose(rebuilt.log_prob(value), jd.log_prob(value), rtol=1e-12)


def test_validate_args_warns_outside_the_support():
    jd = JointDistribution(normal, uniform, validate_args=True)
    with pytest.warns(UserWarning, match="Out-of-support values"):
        jd.log_prob(jnp.asarray([0.5, 1.5]))


def test_marginals_must_have_a_scalar_batch_shape():
    # the per-marginal log probabilities are stacked along a new trailing axis, which
    # requires every marginal to contribute the same shape
    jd = JointDistribution(Normal(jnp.zeros(3), 1.0), Uniform(0.0, 1.0))
    assert jd.batch_shape == (3,)
    with pytest.raises(ValueError):
        jd.log_prob(jnp.asarray([0.5, 0.25]))


def test_sample_rejects_a_plain_integer_key():
    with pytest.raises(AssertionError):
        _mixed_joint().sample(0)
