# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.hybrids._npowerlawmgaussian`.

The hyper-parameter names this family consumes are enumerated by
:class:`~gwkokab.analysis.n_pls_m_gs.common.NPowerlawMGaussianCore`, so the tests below
drive the model through that list. That keeps the fixtures honest and turns any drift
between the model factory and its analysis front-end into a failure.
"""

import inspect

import jax
import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose

from gwkokab.analysis.n_pls_m_gs.common import NPowerlawMGaussianCore
from gwkokab.models import NPowerlawMGaussian
from gwkokab.models.utils import JointDistribution, ScaledMixture


# every ``use_*`` keyword the analysis front-end accepts for this family
SWITCHES = [
    name
    for name in inspect.signature(NPowerlawMGaussianCore.__init__).parameters
    if name.startswith("use_")
]

# GWTC4EffectiveSpinSkewNormalModel has no sampler, so any configuration that includes
# it can only be evaluated, not drawn from.
UNSAMPLEABLE = "use_skew_normal_chi_eff"

COUNTS = [
    dict(N_pl=1, N_g=0),
    dict(N_pl=0, N_g=1),
    dict(N_pl=1, N_g=1),
    dict(N_pl=2, N_g=1),
    dict(N_pl=2, N_g=2),
]


def _build(counts, hyper_parameters, **overrides):
    flags = {name: overrides.get(name, False) for name in SWITCHES}
    core = NPowerlawMGaussianCore(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    return core, NPowerlawMGaussian(**counts, **flags, **params)


@pytest.mark.parametrize("counts", COUNTS)
def test_npmg_is_a_scaled_mixture_of_the_requested_size(counts, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    assert isinstance(model, ScaledMixture)
    assert model.mixture_size == counts["N_pl"] + counts["N_g"]
    assert model.batch_shape == ()
    assert model.event_shape == (2,)


@pytest.mark.parametrize("counts", COUNTS)
def test_npmg_log_scales_are_the_log_rates(counts, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    rates = hyper_parameters([
        f"log_rate_{i}" for i in range(counts["N_pl"] + counts["N_g"])
    ])
    assert_allclose(model.log_scales, list(rates.values()), rtol=1e-12)


@pytest.mark.parametrize("counts", COUNTS)
def test_npmg_components_are_joint_distributions(counts, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    assert all(
        isinstance(component, JointDistribution)
        for component in model.component_distributions
    )


@pytest.mark.parametrize("counts", COUNTS)
@pytest.mark.parametrize("sample_shape", [(), (5,), (2, 3)])
def test_npmg_sample_and_log_prob_shapes(counts, sample_shape, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    samples = model.sample(jrd.key(0), sample_shape)
    assert samples.shape == model.shape(sample_shape)
    assert model.log_prob(samples).shape == sample_shape


@pytest.mark.parametrize("counts", COUNTS)
def test_npmg_samples_lie_in_the_support(counts, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    samples = model.sample(jrd.key(1), (2048,))
    assert jnp.all(model.support.check(samples))
    assert jnp.all(jnp.isfinite(model.log_prob(samples)))


@pytest.mark.parametrize("counts", COUNTS)
def test_npmg_log_prob_is_the_rate_weighted_logsumexp(counts, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    value = model.sample(jrd.key(2), (16,))
    # the mixture carries an explicit union support, which switches on per-component
    # masking: a power-law component contributes nothing to a Gaussian component's draw
    per_component = jnp.stack(
        [
            jnp.where(
                component.support.check(value), component.log_prob(value), -jnp.inf
            )
            for component in model.component_distributions
        ],
        axis=-1,
    )
    expected = jax.nn.logsumexp(model.log_scales + per_component, axis=-1)
    assert_allclose(model.log_prob(value), expected, rtol=1e-10)


@pytest.mark.parametrize("switch", SWITCHES)
def test_npmg_each_switch_adds_the_parameters_it_advertises(switch, hyper_parameters):
    core, model = _build(dict(N_pl=1, N_g=1), hyper_parameters, **{switch: True})
    assert model.event_shape == (len(core.parameters),)


@pytest.mark.parametrize("switch", [s for s in SWITCHES if s != UNSAMPLEABLE])
def test_npmg_each_switch_stays_sampleable(switch, hyper_parameters):
    _, model = _build(dict(N_pl=1, N_g=1), hyper_parameters, **{switch: True})
    samples = model.sample(jrd.key(3), (64,))
    assert samples.shape == (64,) + model.event_shape
    assert jnp.all(jnp.isfinite(model.log_prob(samples)))


def test_npmg_skew_normal_chi_eff_cannot_be_sampled(hyper_parameters):
    _, model = _build(dict(N_pl=1, N_g=1), hyper_parameters, **{UNSAMPLEABLE: True})
    assert jnp.isfinite(model.log_prob(jnp.asarray([35.0, 20.0, 0.1])))
    with pytest.raises(NotImplementedError):
        model.sample(jrd.key(0))


def test_npmg_with_every_switch_on(hyper_parameters):
    core, model = _build(
        dict(N_pl=1, N_g=1),
        hyper_parameters,
        **{name: True for name in SWITCHES},
    )
    # two switches each add a second copy of a coordinate the core lists only once
    # (eccentricity and redshift), so the event is wider than `parameters`
    assert model.event_shape == (len(core.parameters) + 2,)


def test_npmg_a_realistic_configuration(hyper_parameters):
    switches = dict(
        use_beta_spin_magnitude=True,
        use_tilt=True,
        use_eccentricity_mixture=True,
        use_powerlaw_redshift=True,
    )
    core, model = _build(dict(N_pl=2, N_g=1), hyper_parameters, **switches)
    assert model.event_shape == (len(core.parameters),)
    samples = model.sample(jrd.key(4), (256,))
    assert jnp.all(model.support.check(samples))
    assert jnp.all(jnp.isfinite(model.log_prob(samples)))


def test_npmg_with_no_components_at_all(hyper_parameters):
    # neither branch of the builder runs, so the component list is never bound
    with pytest.raises(UnboundLocalError):
        NPowerlawMGaussian(N_pl=0, N_g=0)


def test_npmg_reports_a_missing_hyper_parameter(hyper_parameters):
    core = NPowerlawMGaussianCore(N_pl=1, N_g=0, **{name: False for name in SWITCHES})
    params = hyper_parameters(core.model_parameters)
    params.pop("alpha_pl_0")
    with pytest.raises(ValueError, match="Missing parameter alpha_pl_0"):
        NPowerlawMGaussian(
            N_pl=1, N_g=0, **{name: False for name in SWITCHES}, **params
        )


def test_npmg_is_differentiable_in_its_hyper_parameters(hyper_parameters):
    core = NPowerlawMGaussianCore(N_pl=1, N_g=1, **{name: False for name in SWITCHES})
    params = hyper_parameters(core.model_parameters)
    value = jnp.asarray([35.0, 20.0])

    def log_prob(alpha):
        return NPowerlawMGaussian(
            N_pl=1,
            N_g=1,
            **{name: False for name in SWITCHES},
            **{**params, "alpha_pl_0": alpha},
        ).log_prob(value)

    alpha = jnp.asarray(params["alpha_pl_0"])
    step = 1e-5
    finite_difference = (log_prob(alpha + step) - log_prob(alpha - step)) / (2.0 * step)
    assert_allclose(jax.grad(log_prob)(alpha), finite_difference, rtol=1e-4, atol=1e-6)


def test_npmg_under_jit(hyper_parameters):
    core = NPowerlawMGaussianCore(N_pl=1, N_g=1, **{name: False for name in SWITCHES})
    params = hyper_parameters(core.model_parameters)
    value = jnp.asarray([35.0, 20.0])

    def log_prob(params):
        return NPowerlawMGaussian(
            N_pl=1, N_g=1, **{name: False for name in SWITCHES}, **params
        ).log_prob(value)

    assert_allclose(jax.jit(log_prob)(params), log_prob(params), rtol=1e-10)
