# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable

import chex
import numpy as np
import pytest
from absl.testing import parameterized
from jax import numpy as jnp
from jaxtyping import Array
from numpy.testing import assert_allclose

from gwkokab.utils.math import (
    beta_dist_concentrations_to_mean_variance,
    beta_dist_mean_variance_to_concentrations,
    cumtrapz,
)


@pytest.mark.parametrize(
    "alpha, beta", [(10, 20), (30, 40), (60, 50), (70, 80), (100, 90)]
)
def test_beta_dist1(alpha, beta):
    mean, variance = beta_dist_concentrations_to_mean_variance(alpha, beta)
    alpha_, beta_ = beta_dist_mean_variance_to_concentrations(mean, variance)
    assert np.allclose(alpha, alpha_)
    assert np.allclose(beta, beta_)


@pytest.mark.parametrize(
    "mean, var", [(0.1, 0.02), (0.2, 0.05), (0.3, 0.07), (0.4, 0.1), (0.5, 0.12)]
)
def test_beta_dist2(mean, var):
    alpha, beta = beta_dist_mean_variance_to_concentrations(mean, var)
    mean_, var_ = beta_dist_concentrations_to_mean_variance(alpha, beta)
    assert np.allclose(mean, mean_)
    assert np.allclose(var, var_)


class TestVariants(parameterized.TestCase):
    @chex.variants(  # pyright: ignore
        with_jit=True,
        without_jit=True,
        with_device=True,
        without_device=True,
        with_pmap=True,
    )
    @parameterized.named_parameters(
        [
            (
                "0",
                jnp.stack(
                    jnp.meshgrid(
                        jnp.linspace(0, 1, 40), jnp.linspace(0, 1, 40), indexing="ij"
                    ),
                    axis=-1,
                ),
                lambda x: jnp.ones(x.shape[:-1]),
                lambda x: x[..., 0] * x[..., 1],
            ),
            (
                "1",
                jnp.stack(
                    jnp.meshgrid(
                        jnp.linspace(0, 2, 40), jnp.linspace(0, 2, 40), indexing="ij"
                    ),
                    axis=-1,
                ),
                lambda x: jnp.ones(x.shape[:-1]) * 0.25,
                lambda x: x[..., 0] * x[..., 1] * 0.25,
            ),
            (
                "2",
                jnp.stack(
                    jnp.meshgrid(
                        jnp.linspace(0, 2, 40), jnp.linspace(0, 2, 40), indexing="ij"
                    ),
                    axis=-1,
                ),
                lambda x: jnp.square((3.0 / 8.0) * x[..., 0] * x[..., 1]),
                lambda x: jnp.power(x[..., 0] * x[..., 1] * 0.25, 3.0),
            ),
            (
                "3",
                jnp.linspace(0, 2, 40).reshape(-1, 1),
                lambda x: (3.0 / 8.0) * x * x,
                lambda x: jnp.power(x * 0.5, 3.0),
            ),
            (
                "4",
                jnp.stack(
                    jnp.meshgrid(
                        jnp.linspace(0, 2, 40),
                        jnp.linspace(0, 2, 40),
                        jnp.linspace(0, 2, 40),
                        indexing="ij",
                    ),
                    axis=-1,
                ),
                lambda x: (3.0 / 8.0) ** 3
                * jnp.square(x[..., 0] * x[..., 1] * x[..., 2]),
                lambda x: jnp.power(x[..., 0] * x[..., 1] * x[..., 2], 3.0) / 8**3,
            ),
        ]
    )
    def test_cumtrapz(
        self,
        x: Array,
        pdf_fn: Callable[[Array], Array],
        cdf_fn: Callable[[Array], Array],
    ) -> None:
        @self.variant
        def _cumtrapz(y: Array, x: Array) -> Array:
            return cumtrapz(y, x)

        y = pdf_fn(x)
        result = _cumtrapz(y, x)
        assert_allclose(result, cdf_fn(x), atol=1e-3)
