# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`gwkokab.models.hybrids._multisource`.

The hyper-parameter names this family consumes are enumerated by
:class:`~gwkokab.analysis.multisource.common.MultiSourceModelCore`, so the tests below
drive the model through that list. That keeps the fixtures honest and turns any drift
between the model factory and its analysis front-end into a failure.
"""

import inspect

import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from gwkokab.analysis.multisource.common import MultiSourceModelCore
from gwkokab.models import MultiSourceModel
from gwkokab.models.utils import ScaledMixture


# every ``use_*`` keyword the analysis front-end accepts for this family
SWITCHES = [
    name
    for name in inspect.signature(MultiSourceModelCore.__init__).parameters
    if name.startswith("use_")
]

COUNTS = [
    dict(N_spl=1, N_bpl=0, N_gpl=0, N_gg=0),
    dict(N_spl=0, N_bpl=1, N_gpl=0, N_gg=0),
    dict(N_spl=0, N_bpl=0, N_gpl=1, N_gg=0),
    dict(N_spl=0, N_bpl=0, N_gpl=0, N_gg=1),
    dict(N_spl=1, N_bpl=1, N_gpl=1, N_gg=1),
]


def _build(counts, hyper_parameters, **overrides):
    flags = {name: overrides.get(name, False) for name in SWITCHES}
    core = MultiSourceModelCore(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    return core, MultiSourceModel(**counts, **flags, **params)


@pytest.mark.parametrize("counts", COUNTS)
def test_multisource_is_a_scaled_mixture_of_the_requested_size(
    counts, hyper_parameters
):
    _, model = _build(counts, hyper_parameters)
    assert isinstance(model, ScaledMixture)
    assert model.mixture_size == sum(counts.values())
    assert model.batch_shape == ()
    assert model.event_shape == (2,)


@pytest.mark.parametrize("counts", COUNTS)
def test_multisource_log_scales_are_the_log_rates(counts, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    rates = hyper_parameters([f"log_rate_{i}" for i in range(sum(counts.values()))])
    assert_allclose(model.log_scales, list(rates.values()), rtol=1e-12)


@pytest.mark.parametrize("counts", COUNTS)
def test_multisource_log_prob_shape(counts, hyper_parameters):
    _, model = _build(counts, hyper_parameters)
    value = jnp.asarray([[35.0, 20.0], [50.0, 45.0]])
    assert model.log_prob(value).shape == (2,)


@pytest.mark.parametrize("switch", SWITCHES)
def test_multisource_each_switch_adds_the_parameters_it_advertises(
    switch, hyper_parameters
):
    core, model = _build(
        dict(N_spl=1, N_bpl=1, N_gpl=1, N_gg=1),
        hyper_parameters,
        **{switch: True},
    )
    assert model.event_shape == (len(core.parameters),)


def test_multisource_components_are_ordered_by_family(hyper_parameters):
    counts = dict(N_spl=1, N_bpl=2, N_gpl=1, N_gg=1)
    _, model = _build(counts, hyper_parameters)
    assert model.mixture_size == 5
    assert len(model.component_distributions) == 5


def test_multisource_with_no_components_at_all(hyper_parameters):
    # every family contributes nothing, so the mixture has no components to stack
    with pytest.raises((ValueError, IndexError)):
        MultiSourceModel(N_spl=0, N_bpl=0, N_gpl=0, N_gg=0, log_rate_0=0.0)


def test_multisource_reports_a_missing_hyper_parameter(hyper_parameters):
    counts = dict(N_spl=0, N_bpl=1, N_gpl=0, N_gg=0)
    flags = {name: False for name in SWITCHES}
    core = MultiSourceModelCore(**counts, **flags)
    params = hyper_parameters(core.model_parameters)
    params.pop("m1_alpha1_bpl_0")
    with pytest.raises(ValueError, match="Missing parameter m1_alpha1_bpl_0"):
        MultiSourceModel(**counts, **flags, **params)
