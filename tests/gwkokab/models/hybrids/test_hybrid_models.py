# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the composite population models in :mod:`gwkokab.models.hybrids`.

The hyper-parameter names each family consumes are enumerated by the matching ``*Core``
class in :mod:`gwkokab.analysis`, so the tests below drive the models through those
lists. That keeps the fixtures honest and turns any drift between a model factory and
its analysis front-end into a failure.
"""

import inspect

import jax
import pytest
from jax import numpy as jnp, random as jrd
from numpy.testing import assert_allclose

from gwkokab.analysis.multisource.common import MultiSourceModelCore
from gwkokab.analysis.n_pls_m_gs.common import NPowerlawMGaussianCore
from gwkokab.analysis.subpopulation.common import SubPopulationModelCore
from gwkokab.models import MultiSourceModel, NPowerlawMGaussian, SubPopulationModel
from gwkokab.models.utils import JointDistribution, ScaledMixture


def _switches(core_cls) -> list[str]:
    return [
        name
        for name in inspect.signature(core_cls.__init__).parameters
        if name.startswith("use_")
    ]


NPMG_SWITCHES = _switches(NPowerlawMGaussianCore)
MULTISOURCE_SWITCHES = _switches(MultiSourceModelCore)
SUBPOPULATION_SWITCHES = _switches(SubPopulationModelCore)

# GWTC4EffectiveSpinSkewNormalModel has no sampler, so any configuration that includes
# it can only be evaluated, not drawn from.
UNSAMPLEABLE = "use_skew_normal_chi_eff"


def _build(model_fn, core_cls, counts, hyper_parameters, **overrides):
    flags = {name: overrides.get(name, False) for name in _switches(core_cls)}
    core = core_cls(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    return core, model_fn(**counts, **flags, **params)


###############################################################################
# NPowerlawMGaussian
###############################################################################


NPMG_COUNTS = [
    dict(N_pl=1, N_g=0),
    dict(N_pl=0, N_g=1),
    dict(N_pl=1, N_g=1),
    dict(N_pl=2, N_g=1),
    dict(N_pl=2, N_g=2),
]


@pytest.mark.parametrize("counts", NPMG_COUNTS)
def test_npmg_is_a_scaled_mixture_of_the_requested_size(counts, hyper_parameters):
    _, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        counts,
        hyper_parameters,
    )
    assert isinstance(model, ScaledMixture)
    assert model.mixture_size == counts["N_pl"] + counts["N_g"]
    assert model.batch_shape == ()
    assert model.event_shape == (2,)


@pytest.mark.parametrize("counts", NPMG_COUNTS)
def test_npmg_log_scales_are_the_log_rates(counts, hyper_parameters):
    core, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        counts,
        hyper_parameters,
    )
    rates = hyper_parameters([
        f"log_rate_{i}" for i in range(counts["N_pl"] + counts["N_g"])
    ])
    assert_allclose(model.log_scales, list(rates.values()), rtol=1e-12)


@pytest.mark.parametrize("counts", NPMG_COUNTS)
def test_npmg_components_are_joint_distributions(counts, hyper_parameters):
    _, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        counts,
        hyper_parameters,
    )
    assert all(
        isinstance(component, JointDistribution)
        for component in model.component_distributions
    )


@pytest.mark.parametrize("counts", NPMG_COUNTS)
@pytest.mark.parametrize("sample_shape", [(), (5,), (2, 3)])
def test_npmg_sample_and_log_prob_shapes(counts, sample_shape, hyper_parameters):
    _, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        counts,
        hyper_parameters,
    )
    samples = model.sample(jrd.key(0), sample_shape)
    assert samples.shape == model.shape(sample_shape)
    assert model.log_prob(samples).shape == sample_shape


@pytest.mark.parametrize("counts", NPMG_COUNTS)
def test_npmg_samples_lie_in_the_support(counts, hyper_parameters):
    _, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        counts,
        hyper_parameters,
    )
    samples = model.sample(jrd.key(1), (2048,))
    assert jnp.all(model.support.check(samples))
    assert jnp.all(jnp.isfinite(model.log_prob(samples)))


@pytest.mark.parametrize("counts", NPMG_COUNTS)
def test_npmg_log_prob_is_the_rate_weighted_logsumexp(counts, hyper_parameters):
    _, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        counts,
        hyper_parameters,
    )
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


@pytest.mark.parametrize("switch", NPMG_SWITCHES)
def test_npmg_each_switch_adds_the_parameters_it_advertises(switch, hyper_parameters):
    core, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        dict(N_pl=1, N_g=1),
        hyper_parameters,
        **{switch: True},
    )
    assert model.event_shape == (len(core.parameters),)


@pytest.mark.parametrize("switch", [s for s in NPMG_SWITCHES if s != UNSAMPLEABLE])
def test_npmg_each_switch_stays_sampleable(switch, hyper_parameters):
    _, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        dict(N_pl=1, N_g=1),
        hyper_parameters,
        **{switch: True},
    )
    samples = model.sample(jrd.key(3), (64,))
    assert samples.shape == (64,) + model.event_shape
    assert jnp.all(jnp.isfinite(model.log_prob(samples)))


def test_npmg_skew_normal_chi_eff_cannot_be_sampled(hyper_parameters):
    _, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        dict(N_pl=1, N_g=1),
        hyper_parameters,
        **{UNSAMPLEABLE: True},
    )
    assert jnp.isfinite(model.log_prob(jnp.asarray([35.0, 20.0, 0.1])))
    with pytest.raises(NotImplementedError):
        model.sample(jrd.key(0))


def test_npmg_with_every_switch_on(hyper_parameters):
    core, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        dict(N_pl=1, N_g=1),
        hyper_parameters,
        **{name: True for name in NPMG_SWITCHES},
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
    core, model = _build(
        NPowerlawMGaussian,
        NPowerlawMGaussianCore,
        dict(N_pl=2, N_g=1),
        hyper_parameters,
        **switches,
    )
    assert model.event_shape == (len(core.parameters),)
    samples = model.sample(jrd.key(4), (256,))
    assert jnp.all(model.support.check(samples))
    assert jnp.all(jnp.isfinite(model.log_prob(samples)))


def test_npmg_with_no_components_at_all(hyper_parameters):
    # neither branch of the builder runs, so the component list is never bound
    with pytest.raises(UnboundLocalError):
        NPowerlawMGaussian(N_pl=0, N_g=0)


def test_npmg_reports_a_missing_hyper_parameter(hyper_parameters):
    core = NPowerlawMGaussianCore(
        N_pl=1, N_g=0, **{name: False for name in NPMG_SWITCHES}
    )
    params = hyper_parameters(core.model_parameters)
    params.pop("alpha_pl_0")
    with pytest.raises(ValueError, match="Missing parameter alpha_pl_0"):
        NPowerlawMGaussian(
            N_pl=1, N_g=0, **{name: False for name in NPMG_SWITCHES}, **params
        )


def test_npmg_is_differentiable_in_its_hyper_parameters(hyper_parameters):
    core = NPowerlawMGaussianCore(
        N_pl=1, N_g=1, **{name: False for name in NPMG_SWITCHES}
    )
    params = hyper_parameters(core.model_parameters)
    value = jnp.asarray([35.0, 20.0])

    def log_prob(alpha):
        return NPowerlawMGaussian(
            N_pl=1,
            N_g=1,
            **{name: False for name in NPMG_SWITCHES},
            **{**params, "alpha_pl_0": alpha},
        ).log_prob(value)

    alpha = jnp.asarray(params["alpha_pl_0"])
    step = 1e-5
    finite_difference = (log_prob(alpha + step) - log_prob(alpha - step)) / (2.0 * step)
    assert_allclose(jax.grad(log_prob)(alpha), finite_difference, rtol=1e-4, atol=1e-6)


def test_npmg_under_jit(hyper_parameters):
    core = NPowerlawMGaussianCore(
        N_pl=1, N_g=1, **{name: False for name in NPMG_SWITCHES}
    )
    params = hyper_parameters(core.model_parameters)
    value = jnp.asarray([35.0, 20.0])

    def log_prob(params):
        return NPowerlawMGaussian(
            N_pl=1, N_g=1, **{name: False for name in NPMG_SWITCHES}, **params
        ).log_prob(value)

    assert_allclose(jax.jit(log_prob)(params), log_prob(params), rtol=1e-10)


###############################################################################
# MultiSourceModel
###############################################################################


MULTISOURCE_COUNTS = [
    dict(N_spl=1, N_bpl=0, N_gpl=0, N_gg=0),
    dict(N_spl=0, N_bpl=1, N_gpl=0, N_gg=0),
    dict(N_spl=0, N_bpl=0, N_gpl=1, N_gg=0),
    dict(N_spl=0, N_bpl=0, N_gpl=0, N_gg=1),
    dict(N_spl=1, N_bpl=1, N_gpl=1, N_gg=1),
]


@pytest.mark.parametrize("counts", MULTISOURCE_COUNTS)
def test_multisource_is_a_scaled_mixture_of_the_requested_size(
    counts, hyper_parameters
):
    _, model = _build(
        MultiSourceModel,
        MultiSourceModelCore,
        counts,
        hyper_parameters,
    )
    assert isinstance(model, ScaledMixture)
    assert model.mixture_size == sum(counts.values())
    assert model.batch_shape == ()
    assert model.event_shape == (2,)


@pytest.mark.parametrize("counts", MULTISOURCE_COUNTS)
def test_multisource_log_scales_are_the_log_rates(counts, hyper_parameters):
    _, model = _build(
        MultiSourceModel,
        MultiSourceModelCore,
        counts,
        hyper_parameters,
    )
    rates = hyper_parameters([f"log_rate_{i}" for i in range(sum(counts.values()))])
    assert_allclose(model.log_scales, list(rates.values()), rtol=1e-12)


@pytest.mark.parametrize("counts", MULTISOURCE_COUNTS)
def test_multisource_log_prob_shape(counts, hyper_parameters):
    _, model = _build(
        MultiSourceModel,
        MultiSourceModelCore,
        counts,
        hyper_parameters,
    )
    value = jnp.asarray([[35.0, 20.0], [50.0, 45.0]])
    assert model.log_prob(value).shape == (2,)


@pytest.mark.parametrize("switch", MULTISOURCE_SWITCHES)
def test_multisource_each_switch_adds_the_parameters_it_advertises(
    switch, hyper_parameters
):
    core, model = _build(
        MultiSourceModel,
        MultiSourceModelCore,
        dict(N_spl=1, N_bpl=1, N_gpl=1, N_gg=1),
        hyper_parameters,
        **{switch: True},
    )
    assert model.event_shape == (len(core.parameters),)


def test_multisource_components_are_ordered_by_family(hyper_parameters):
    counts = dict(N_spl=1, N_bpl=2, N_gpl=1, N_gg=1)
    _, model = _build(
        MultiSourceModel,
        MultiSourceModelCore,
        counts,
        hyper_parameters,
    )
    assert model.mixture_size == 5
    assert len(model.component_distributions) == 5


def test_multisource_with_no_components_at_all(hyper_parameters):
    # every family contributes nothing, so the mixture has no components to stack
    with pytest.raises((ValueError, IndexError)):
        MultiSourceModel(N_spl=0, N_bpl=0, N_gpl=0, N_gg=0, log_rate_0=0.0)


def test_multisource_reports_a_missing_hyper_parameter(hyper_parameters):
    counts = dict(N_spl=0, N_bpl=1, N_gpl=0, N_gg=0)
    flags = {name: False for name in MULTISOURCE_SWITCHES}
    core = MultiSourceModelCore(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    params.pop("m1_alpha1_bpl_0")
    with pytest.raises(ValueError, match="Missing parameter m1_alpha1_bpl_0"):
        MultiSourceModel(**counts, **flags, **params)


###############################################################################
# SubPopulationModel
###############################################################################


# N_spl > 0 routes through create_powerlaws, which is broken; see the xfail below.
SUBPOPULATION_COUNTS = [
    dict(N_spl=0, N_bpl=1, N_gpl=1),
    dict(N_spl=0, N_bpl=2, N_gpl=0),
    dict(N_spl=0, N_bpl=0, N_gpl=2),
    dict(N_spl=0, N_bpl=2, N_gpl=2),
]


@pytest.mark.parametrize("counts", SUBPOPULATION_COUNTS)
def test_subpopulation_is_a_scaled_mixture_of_the_requested_size(
    counts, hyper_parameters
):
    _, model = _build(
        SubPopulationModel,
        SubPopulationModelCore,
        counts,
        hyper_parameters,
    )
    assert isinstance(model, ScaledMixture)
    assert model.mixture_size == sum(counts.values())
    assert model.batch_shape == ()
    assert model.event_shape == (2,)


@pytest.mark.parametrize("counts", SUBPOPULATION_COUNTS)
def test_subpopulation_log_scales_combine_rate_weights_and_normalisation(
    counts, hyper_parameters
):
    core, model = _build(
        SubPopulationModel,
        SubPopulationModelCore,
        counts,
        hyper_parameters,
    )
    n = sum(counts.values())
    params = hyper_parameters(core.model_parameters)
    lambdas = [params[f"lambda_{i}"] for i in range(n - 1)]
    lambdas.append(1.0 - sum(lambdas))
    # every log scale is log_rate - logZ + log(lambda_i), so the differences between
    # them are exactly the differences between the log mixing weights
    log_lambdas = jnp.log(jnp.asarray(lambdas))
    assert_allclose(
        model.log_scales - model.log_scales[0],
        log_lambdas - log_lambdas[0],
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("counts", SUBPOPULATION_COUNTS)
def test_subpopulation_log_prob_is_finite_on_its_support(counts, hyper_parameters):
    _, model = _build(
        SubPopulationModel,
        SubPopulationModelCore,
        counts,
        hyper_parameters,
    )
    # the event of this family is (m1, q), not (m1, m2)
    value = jnp.asarray([[30.0, 0.7], [45.0, 0.9]])
    assert jnp.all(model.support.check(value))
    assert jnp.all(jnp.isfinite(model.log_prob(value)))


@pytest.mark.parametrize("switch", SUBPOPULATION_SWITCHES)
def test_subpopulation_each_switch_adds_the_parameters_it_advertises(
    switch, hyper_parameters
):
    core, model = _build(
        SubPopulationModel,
        SubPopulationModelCore,
        dict(N_spl=0, N_bpl=1, N_gpl=1),
        hyper_parameters,
        **{switch: True},
    )
    assert model.event_shape == (len(core.parameters),)


def test_subpopulation_reports_a_missing_hyper_parameter(hyper_parameters):
    counts = dict(N_spl=0, N_bpl=1, N_gpl=1)
    flags = {name: False for name in SUBPOPULATION_SWITCHES}
    core = SubPopulationModelCore(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    params.pop("m1_break_bpl_0")
    with pytest.raises(ValueError, match="Missing parameter m1_break_bpl_0"):
        SubPopulationModel(**counts, **flags, **params)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the smoothed-powerlaw sub-population routes through create_powerlaws, which "
        "passes mmin/mmax to DoublyTruncatedPowerLaw instead of low/high"
    ),
)
def test_subpopulation_with_a_smoothed_powerlaw_component(hyper_parameters):
    counts = dict(N_spl=1, N_bpl=1, N_gpl=1)
    flags = {name: False for name in SUBPOPULATION_SWITCHES}
    core = SubPopulationModelCore(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    SubPopulationModel(**counts, **flags, **params)
