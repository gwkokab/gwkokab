# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.hybrids._subpopulation`.

The hyper-parameter names this family consumes are enumerated by
:class:`~gwkokab.analysis.subpopulation.common.SubPopulationModelCore`, so the tests
below drive the model through that list. That keeps the fixtures honest and turns any
drift between the model factory and its analysis front-end into a failure.
"""

import inspect

import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from gwkokab.analysis.subpopulation.common import SubPopulationModelCore
from gwkokab.models import SubPopulationModel
from gwkokab.models.utils import ScaledMixture


# every ``use_*`` keyword the analysis front-end accepts for this family
SWITCHES = [
    name
    for name in inspect.signature(SubPopulationModelCore.__init__).parameters
    if name.startswith("use_")
]

# N_spl > 0 routes through create_powerlaws, which is broken; see the xfail below.
COUNTS = [
    dict(N_spl=0, N_bpl=1, N_gpl=1),
    dict(N_spl=0, N_bpl=2, N_gpl=0),
    dict(N_spl=0, N_bpl=0, N_gpl=2),
    dict(N_spl=0, N_bpl=2, N_gpl=2),
]


def _build(counts, hyper_parameters, **overrides):
    flags = {name: overrides.get(name, False) for name in SWITCHES}
    core = SubPopulationModelCore(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    return core, SubPopulationModel(**counts, **flags, **params)


@pytest.mark.parametrize("counts", COUNTS)
def test_subpopulation_is_a_scaled_mixture_of_the_requested_size(
    counts, hyper_parameters
):
    _, model = _build(counts, hyper_parameters)
    assert isinstance(model, ScaledMixture)
    assert model.mixture_size == sum(counts.values())
    assert model.batch_shape == ()
    assert model.event_shape == (2,)


@pytest.mark.parametrize("counts", COUNTS)
def test_subpopulation_log_scales_combine_rate_weights_and_normalisation(
    counts, hyper_parameters
):
    core, model = _build(counts, hyper_parameters)
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


@pytest.mark.parametrize("counts", COUNTS)
def test_subpopulation_log_prob_is_finite_on_its_support(counts, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    # the event of this family is (m1, q), not (m1, m2)
    value = jnp.asarray([[30.0, 0.7], [45.0, 0.9]])
    assert jnp.all(model.support.check(value))
    assert jnp.all(jnp.isfinite(model.log_prob(value)))


@pytest.mark.parametrize("switch", SWITCHES)
def test_subpopulation_each_switch_adds_the_parameters_it_advertises(
    switch, hyper_parameters
):
    core, model = _build(
        dict(N_spl=0, N_bpl=1, N_gpl=1),
        hyper_parameters,
        **{switch: True},
    )
    assert model.event_shape == (len(core.parameters),)


def test_subpopulation_reports_a_missing_hyper_parameter(hyper_parameters):
    counts = dict(N_spl=0, N_bpl=1, N_gpl=1)
    flags = {name: False for name in SWITCHES}
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
    flags = {name: False for name in SWITCHES}
    core = SubPopulationModelCore(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    SubPopulationModel(**counts, **flags, **params)
