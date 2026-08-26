# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`gwkokab.inference.factory.get_likelihood_fn`.

The factory is the single place where the (sampler backend x data representation) matrix
is resolved, and it is called with strings read straight out of user JSON, so both the
happy mapping and the failure mode matter.
"""

import pytest

from gwkokab.inference import (
    flowMC_analytical_gwalk_poisson_likelihood,
    flowMC_discrete_poisson_likelihood,
    numpyro_analytical_gwalk_poisson_likelihood,
    numpyro_discrete_poisson_likelihood,
)
from gwkokab.inference.factory import get_likelihood_fn
from gwkokab.utils.exceptions import LoggedValueError


@pytest.mark.parametrize(
    ("sampler_name", "analysis_type", "expected"),
    [
        ("flowMC", "discrete", flowMC_discrete_poisson_likelihood),
        ("flowMC", "analytical_gwalk", flowMC_analytical_gwalk_poisson_likelihood),
        ("numpyro", "discrete", numpyro_discrete_poisson_likelihood),
        ("numpyro", "analytical_gwalk", numpyro_analytical_gwalk_poisson_likelihood),
    ],
)
def test_get_likelihood_fn_resolves_the_matrix(sampler_name, analysis_type, expected):
    """Each of the four supported combinations maps to its own module."""
    assert get_likelihood_fn(sampler_name, analysis_type) is expected


def test_get_likelihood_fn_returns_four_distinct_functions():
    """A copy-paste slip in the factory would collapse two cells of the matrix."""
    resolved = {
        get_likelihood_fn(sampler, analysis)
        for sampler in ("flowMC", "numpyro")
        for analysis in ("discrete", "analytical_gwalk")
    }
    assert len(resolved) == 4


@pytest.mark.parametrize("sampler_name", ["flowMC", "numpyro"])
def test_unknown_analysis_type_falls_back_to_discrete(sampler_name):
    """Only ``analytical_gwalk`` is special-cased; anything else means discrete.

    This is the factory's current contract — the analysis type has already been
    validated by the Pydantic config layer by the time it gets here.
    """
    expected = {
        "flowMC": flowMC_discrete_poisson_likelihood,
        "numpyro": numpyro_discrete_poisson_likelihood,
    }[sampler_name]
    assert get_likelihood_fn(sampler_name, "something_else") is expected


@pytest.mark.parametrize(
    "sampler_name", ["", "FlowMC", "flowmc", "NumPyro", "emcee", "nessai"]
)
def test_unknown_sampler_raises(sampler_name):
    """Sampler names are matched exactly; a near-miss must fail loudly."""
    with pytest.raises(LoggedValueError, match="Unsupported sampler_name"):
        get_likelihood_fn(sampler_name, "discrete")


def test_unknown_sampler_error_names_both_arguments():
    with pytest.raises(LoggedValueError) as excinfo:
        get_likelihood_fn("mcmc", "analytical_gwalk")
    message = str(excinfo.value)
    assert "mcmc" in message
    assert "analytical_gwalk" in message


def test_public_api_is_re_exported():
    """``gwkokab.inference`` re-exports the four likelihoods, the two pure-math helpers
    and the factory module itself.
    """
    import gwkokab.inference as inference

    expected = {
        "factory",
        "flowMC_analytical_gwalk_poisson_likelihood",
        "flowMC_discrete_poisson_likelihood",
        "numpyro_analytical_gwalk_poisson_likelihood",
        "numpyro_discrete_poisson_likelihood",
        "analytical_gwalk_poisson_likelihood_fn",
        "discrete_poisson_likelihood_fn",
    }
    missing = {name for name in expected if not hasattr(inference, name)}
    assert not missing


def test_factory_module_is_reachable_from_the_package():
    import gwkokab.inference as inference

    assert inference.factory.get_likelihood_fn is get_likelihood_fn
