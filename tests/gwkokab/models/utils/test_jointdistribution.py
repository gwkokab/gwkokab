# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import chex
import jax.random as jrd
import pytest
from absl.testing import parameterized
from numpyro.distributions import Exponential, LogNormal, Normal, Uniform

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
            return jd.sample(jrd.PRNGKey(0))

        create_joint_distribution()
