# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the :mod:`gwkokab.inference` tests.

The fixtures here deliberately build models whose Poisson log-likelihood is available in
closed form, so the tests can assert exact numbers rather than merely self-consistent
ones. The workhorse is a :class:`~gwkokab.models.utils.ScaledMixture` of uniform
components on the unit hyper-cube: its log-density at *any* interior point is
``logsumexp(log_scales)``, independent of the point, which collapses every Monte-Carlo
sum in the likelihood to a closed-form expression.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import pytest
from jaxtyping import Array
from numpyro.distributions import Distribution, Normal, Uniform

from gwkokab.models.utils import JointDistribution, ScaledMixture


def _unit_square_mixture(
    log_scales: Sequence[float] | Array,
    dim: int = 2,
    validate_args: Optional[bool] = None,
) -> ScaledMixture:
    """Mixture of ``len(log_scales)`` uniform components on :math:`[0, 1]^{dim}`."""
    log_scales = jnp.atleast_1d(jnp.asarray(log_scales, dtype=float))
    components = [
        JointDistribution(*(Uniform(0.0, 1.0) for _ in range(dim)))
        for _ in range(log_scales.shape[-1])
    ]
    return ScaledMixture(log_scales, components, validate_args=validate_args)


class _RecordingEstimator:
    """Poisson-mean estimator returning fixed values and recording its calls.

    The real estimators return the pair ``(mean, variance)``; only that contract matters
    to the likelihood, so a constant stand-in keeps the expected rate out of the closed-
    form arithmetic while still exercising the call path.
    """

    def __init__(self, mean: float, variance: float) -> None:
        self.mean = jnp.asarray(mean, dtype=float)
        self.variance = jnp.asarray(variance, dtype=float)
        self.calls: List[Tuple[Distribution, Dict[str, Any]]] = []

    def __call__(
        self, model_instance: Distribution, **kwargs: Any
    ) -> Tuple[Array, Array]:
        self.calls.append((model_instance, kwargs))
        return self.mean, self.variance


class _RecordingModelFactory:
    """``dist_fn`` stand-in recording the keyword arguments it was built with.

    The wrappers under test are responsible for turning a flat parameter vector (or a
    set of NumPyro sample sites) into keyword arguments; recording them is how the tests
    check that mapping without reaching into the wrapper internals.
    """

    def __init__(self, log_scale_name: str = "log_scale", dim: int = 2) -> None:
        self.log_scale_name = log_scale_name
        self.dim = dim
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> ScaledMixture:
        self.calls.append(dict(kwargs))
        return _unit_square_mixture(
            jnp.atleast_1d(kwargs[self.log_scale_name]),
            dim=self.dim,
            validate_args=kwargs.get("validate_args"),
        )

    @property
    def last_call(self) -> Dict[str, Any]:
        assert self.calls, "dist_fn was never called"
        return self.calls[-1]


@pytest.fixture
def mixture() -> Callable[..., ScaledMixture]:
    """Factory for a uniform :class:`ScaledMixture` on the unit hyper-cube.

    Returns a callable ``(log_scales, dim=2, validate_args=None) -> ScaledMixture``.
    Because every component is uniform on :math:`[0, 1]^{dim}`, the mixture's log-
    density is the constant ``logsumexp(log_scales)`` inside the cube.
    """
    return _unit_square_mixture


@pytest.fixture
def normal_mixture() -> Callable[..., ScaledMixture]:
    """Factory for a mixture whose log-density actually varies across samples.

    Uniform components make the closed-form checks easy but hide any bug that only shows
    up when the per-sample log-probabilities differ (the variance terms, in particular).
    This factory builds a mixture of diagonal Gaussians for those tests.
    """

    def _make(log_scales: Sequence[float] | Array, locs, scales) -> ScaledMixture:
        log_scales = jnp.atleast_1d(jnp.asarray(log_scales, dtype=float))
        locs = jnp.asarray(locs, dtype=float)
        scales = jnp.asarray(scales, dtype=float)
        components = [
            JointDistribution(
                *(Normal(loc, scale) for loc, scale in zip(loc_row, scale_row))
            )
            for loc_row, scale_row in zip(locs, scales)
        ]
        return ScaledMixture(log_scales, components)

    return _make


@pytest.fixture
def estimator() -> Callable[..., _RecordingEstimator]:
    """Factory for a constant, call-recording Poisson-mean estimator."""

    def _make(mean: float = 0.0, variance: float = 0.0) -> _RecordingEstimator:
        return _RecordingEstimator(mean, variance)

    return _make


@pytest.fixture
def model_factory() -> Callable[..., _RecordingModelFactory]:
    """Factory for a call-recording ``dist_fn``."""

    def _make(
        log_scale_name: str = "log_scale", dim: int = 2
    ) -> _RecordingModelFactory:
        return _RecordingModelFactory(log_scale_name, dim)

    return _make


@pytest.fixture
def discrete_data() -> Callable[..., Dict[str, Any]]:
    """Factory for the ``data`` payload the discrete likelihood consumes.

    The default payload is a single bucket of ``n_events`` events, each with
    ``n_samples`` posterior samples sitting at the centre of the unit hyper-cube, unit
    reference priors and an all-true mask — the configuration for which the log-
    likelihood reduces to ``n_events * logsumexp(log_scales)``.
    """

    def _make(
        n_events: int = 3,
        n_samples: int = 5,
        dim: int = 2,
        value: float = 0.5,
        T_obs: float = 1.0,
    ) -> Dict[str, Any]:
        masks = jnp.ones((n_events, n_samples), dtype=bool)
        return {
            "data_group": (jnp.full((n_events, n_samples, dim), value),),
            "log_ref_priors_group": (jnp.zeros((n_events, n_samples)),),
            "masks_group": (masks,),
            "pmean_kwargs": {"T_obs": T_obs},
            "N_pes": (jnp.full((n_events,), n_samples),),
        }

    return _make


@pytest.fixture
def analytical_data() -> Callable[..., Dict[str, Any]]:
    """Factory for the ``data`` payload the analytical-GWalk likelihood consumes."""

    def _make(
        n_events: int = 3,
        n_samples: int = 5,
        dim: int = 2,
        value: float = 0.5,
        T_obs: float = 1.0,
    ) -> Dict[str, Any]:
        return {
            "samples_stack": jnp.full((n_events, n_samples, dim), value),
            "ln_offsets": jnp.zeros((n_events, n_samples)),
            "pmean_kwargs": {"T_obs": T_obs},
        }

    return _make
