# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-parameter model factories in
:mod:`gwkokab.models.hybrids._ncombination`.
"""

import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose
from numpyro.distributions import (
    Beta,
    constraints,
    MixtureGeneral,
    Normal,
    Uniform,
)

from gwkokab.models.hybrids._ncombination import (
    _get_parameter,
    combine_distributions,
    create_beta_distributions,
    create_broken_powerlaws,
    create_gaussian_primary_mass_ratio,
    create_generic_powerlaws,
    create_generic_smoothed_powerlaw_mass_ratio,
    create_generic_tilt_model,
    create_gwtc4_effective_spin_skew_normal_models,
    create_madau_dickinson_redshift_model,
    create_powerlaw_primary_mass_ratios,
    create_powerlaw_redshift_model,
    create_powerlaws,
    create_smoothed_broken_powerlaws_mass_ratio_powerlaw,
    create_smoothed_gaussian_primary_mass_ratio,
    create_smoothed_powerlaw_primary_mass_ratio,
    create_spin_magnitude_mixture_models,
    create_truncated_normal_distributions,
    create_two_truncated_normal_mixture,
    create_uniform_distributions,
)
from gwkokab.models.mass import (
    BrokenPowerlaw,
    GaussianPrimaryMassRatio,
    GenericSmoothedPowerlawMassRatio,
    PowerlawPrimaryMassRatio,
    SmoothedBrokenPowerlawMassRatioPowerlaw,
    SmoothedGaussianPrimaryMassRatio,
    SmoothedPowerlawPrimaryMassRatio,
)
from gwkokab.models.redshift import (
    MadauDickinsonRedshiftModel,
    PowerlawRedshiftModel,
)
from gwkokab.models.spin import GWTC4EffectiveSpinSkewNormalModel
from gwkokab.models.utils import (
    DoublyTruncatedPowerLaw,
    ExtendedSupportTransformedDistribution,
)


###############################################################################
# _get_parameter
###############################################################################


def test_get_parameter_returns_a_present_value():
    assert _get_parameter({"alpha_pl_0": 1.5}, "alpha_pl_0") == 1.5


def test_get_parameter_falls_back_to_the_default():
    assert _get_parameter({}, "alpha_pl_0", default=2.5) == 2.5


def test_get_parameter_prefers_the_stored_value_over_the_default():
    assert _get_parameter({"alpha_pl_0": 1.5}, "alpha_pl_0", default=2.5) == 1.5


def test_get_parameter_returns_none_for_an_optional_missing_value():
    assert _get_parameter({}, "alpha_pl_0", is_necessary=False) is None


def test_get_parameter_raises_for_a_required_missing_value():
    with pytest.raises(ValueError, match="Missing parameter alpha_pl_0"):
        _get_parameter({}, "alpha_pl_0")


def test_get_parameter_treats_a_stored_none_as_missing():
    assert _get_parameter({"alpha_pl_0": None}, "alpha_pl_0", default=2.5) == 2.5


@pytest.mark.parametrize("default", [0.0, False])
def test_get_parameter_accepts_a_falsy_default(default):
    assert _get_parameter({}, "alpha_pl_0", default=default) == default


###############################################################################
# combine_distributions
###############################################################################


def test_combine_distributions_appends_one_distribution_per_component():
    base = [["a"], ["b"], ["c"]]
    combined = combine_distributions(base, ["x", "y", "z"])
    assert combined == [["a", "x"], ["b", "y"], ["c", "z"]]


def test_combine_distributions_stops_at_the_shorter_input():
    assert combine_distributions([["a"], ["b"]], ["x"]) == [["a", "x"]]


def test_combine_distributions_does_not_mutate_its_input():
    base = [["a"]]
    combine_distributions(base, ["x"])
    assert base == [["a"]]


###############################################################################
# the simple per-parameter factories
###############################################################################


@pytest.mark.parametrize("N", [0, 1, 3])
def test_create_beta_distributions(N):
    params = {}
    for i in range(N):
        params[f"a_1_mean_pl_{i}"] = 0.2 + 0.1 * i
        params[f"a_1_variance_pl_{i}"] = 0.01
    built = create_beta_distributions(
        N=N, parameter_name="a_1", component_type="pl", params=params
    )
    assert len(built) == N
    for i, distribution in enumerate(built):
        assert isinstance(distribution, Beta)
        assert_allclose(distribution.mean, 0.2 + 0.1 * i, rtol=1e-10)
        assert_allclose(distribution.variance, 0.01, rtol=1e-10)


def test_create_beta_distributions_requires_its_parameters():
    with pytest.raises(ValueError, match="Missing parameter a_1_mean_pl_0"):
        create_beta_distributions(
            N=1, parameter_name="a_1", component_type="pl", params={}
        )


@pytest.mark.parametrize("N", [1, 2])
def test_create_truncated_normal_distributions(N):
    params = {}
    for i in range(N):
        params[f"chi_p_loc_g_{i}"] = 0.2
        params[f"chi_p_scale_g_{i}"] = 0.3
        params[f"chi_p_low_g_{i}"] = 0.0
        params[f"chi_p_high_g_{i}"] = 1.0
    built = create_truncated_normal_distributions(
        N=N, parameter_name="chi_p", component_type="g", params=params
    )
    assert len(built) == N
    for distribution in built:
        assert_allclose(distribution.low, 0.0, rtol=1e-12)
        assert_allclose(distribution.high, 1.0, rtol=1e-12)
        assert_allclose(distribution.base_dist.loc, 0.2, rtol=1e-12)
        assert_allclose(distribution.base_dist.scale, 0.3, rtol=1e-12)


def test_create_truncated_normal_distributions_bounds_are_optional():
    # numpyro's TruncatedNormal is a factory: with no bounds it hands back a plain
    # Normal rather than a truncated distribution
    built = create_truncated_normal_distributions(
        N=1,
        parameter_name="chi_p",
        component_type="g",
        params={"chi_p_loc_g_0": 0.2, "chi_p_scale_g_0": 0.3},
    )
    assert isinstance(built[0], Normal)
    assert_allclose(built[0].loc, 0.2, rtol=1e-12)
    assert_allclose(built[0].scale, 0.3, rtol=1e-12)
    assert built[0].support == constraints.real


@pytest.mark.parametrize("N", [1, 2])
def test_create_uniform_distributions(N):
    params = {}
    for i in range(N):
        params[f"phi_1_low_pl_{i}"] = 0.0
        params[f"phi_1_high_pl_{i}"] = 6.0
    built = create_uniform_distributions(
        N=N, parameter_name="phi_1", component_type="pl", params=params
    )
    assert len(built) == N
    for distribution in built:
        assert isinstance(distribution, Uniform)
        assert_allclose(distribution.low, 0.0, rtol=1e-12)
        assert_allclose(distribution.high, 6.0, rtol=1e-12)


def test_create_generic_powerlaws():
    built = create_generic_powerlaws(
        N=1,
        parameter_name="eccentricity",
        component_type="pl",
        params={
            "eccentricity_alpha_pl_0": -1.5,
            "eccentricity_low_pl_0": 1e-3,
            "eccentricity_high_pl_0": 1.0,
        },
    )
    assert isinstance(built[0], DoublyTruncatedPowerLaw)
    assert_allclose(built[0].alpha, -1.5, rtol=1e-12)
    assert_allclose(built[0].low, 1e-3, rtol=1e-12)
    assert_allclose(built[0].high, 1.0, rtol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "create_powerlaws calls DoublyTruncatedPowerLaw with mmin/mmax, but the "
        "distribution takes low/high; this is what breaks SubPopulationModel for N_spl>0"
    ),
)
def test_create_powerlaws():
    built = create_powerlaws(
        N=1,
        parameter_name="m1",
        component_type="spl",
        params={"m1_alpha_spl_0": 1.5, "m1_low_spl_0": 10.0, "m1_high_spl_0": 60.0},
    )
    assert isinstance(built[0], DoublyTruncatedPowerLaw)


def test_create_broken_powerlaws():
    built = create_broken_powerlaws(
        N=1,
        parameter_name="m1",
        component_type="bpl",
        params={
            "m1_alpha1_bpl_0": 1.5,
            "m1_alpha2_bpl_0": 3.0,
            "m1_break_bpl_0": 30.0,
            "m1_low_bpl_0": 10.0,
            "m1_high_bpl_0": 60.0,
        },
    )
    assert isinstance(built[0], BrokenPowerlaw)
    assert_allclose(built[0].alpha1, 1.5, rtol=1e-12)
    assert_allclose(built[0].alpha2, 3.0, rtol=1e-12)
    assert_allclose(built[0].mbreak, 30.0, rtol=1e-12)
    assert built[0].support.lower_bound == 10.0
    assert built[0].support.upper_bound == 60.0


###############################################################################
# redshift
###############################################################################


def test_create_powerlaw_redshift_model():
    built = create_powerlaw_redshift_model(
        N=1,
        parameter_name="redshift",
        component_type="g",
        params={"redshift_kappa_g_0": 2.7, "redshift_z_max_g_0": 1.5},
    )
    assert isinstance(built[0], PowerlawRedshiftModel)
    assert_allclose(built[0].kappa, 2.7, rtol=1e-12)
    assert built[0].support.upper_bound == 1.5


def test_create_madau_dickinson_redshift_model():
    built = create_madau_dickinson_redshift_model(
        N=1,
        parameter_name="redshift",
        component_type="g",
        params={
            "redshift_kappa_g_0": 2.7,
            "redshift_z_max_g_0": 1.5,
            "redshift_gamma_g_0": 2.9,
            "redshift_z_peak_g_0": 0.5,
        },
    )
    assert isinstance(built[0], MadauDickinsonRedshiftModel)
    assert_allclose(built[0].gamma, 2.9, rtol=1e-12)
    assert_allclose(built[0].z_peak, 0.5, rtol=1e-12)


###############################################################################
# spin
###############################################################################


def test_create_generic_tilt_model():
    params = {
        "cos_tilt_zeta_pl_0": 0.4,
        "cos_tilt_1_loc_pl_0": 1.0,
        "cos_tilt_2_loc_pl_0": 1.0,
        "cos_tilt_1_scale_pl_0": 0.5,
        "cos_tilt_2_scale_pl_0": 0.6,
    }
    built = create_generic_tilt_model(
        N=1, parameter_name="cos_tilt", component_type="pl", params=params
    )
    assert isinstance(built[0], MixtureGeneral)
    assert built[0].event_shape == (2,)
    assert_allclose(built[0].mixing_distribution.probs, [0.6, 0.4], rtol=1e-12)


def test_create_generic_tilt_model_bounds_default_to_the_unit_interval():
    params = {
        "cos_tilt_zeta_pl_0": 0.0,
        "cos_tilt_1_loc_pl_0": 1.0,
        "cos_tilt_2_loc_pl_0": 1.0,
        "cos_tilt_1_scale_pl_0": 0.5,
        "cos_tilt_2_scale_pl_0": 0.6,
    }
    built = create_generic_tilt_model(
        N=1, parameter_name="cos_tilt", component_type="pl", params=params
    )
    # a zero mixing weight leaves the isotropic component, uniform on [-1, 1]^2
    assert_allclose(
        built[0].log_prob(jnp.asarray([0.0, 0.0])), jnp.log(0.25), rtol=1e-12
    )


def test_create_two_truncated_normal_mixture():
    params = {
        "eccentricity_comp1_loc_g_0": 0.1,
        "eccentricity_comp1_scale_g_0": 0.2,
        "eccentricity_comp2_loc_g_0": 0.6,
        "eccentricity_comp2_scale_g_0": 0.1,
        "eccentricity_zeta_g_0": 0.3,
    }
    built = create_two_truncated_normal_mixture(
        N=1, parameter_name="eccentricity", component_type="g", params=params
    )
    assert isinstance(built[0], MixtureGeneral)
    assert_allclose(built[0].mixing_distribution.probs, [0.7, 0.3], rtol=1e-12)
    # the truncation bounds are optional and default to the whole real line
    assert all(
        component.support == constraints.real
        for component in built[0].component_distributions
    )


def test_create_spin_magnitude_mixture_models():
    params = {"a_zeta_pl_0": 0.3}
    for spin in ("a_1", "a_2"):
        for component in ("comp1", "comp2"):
            params[f"{spin}_{component}_loc_pl_0"] = 0.3
            params[f"{spin}_{component}_scale_pl_0"] = 0.2
    built = create_spin_magnitude_mixture_models(
        N=1, parameter_name="a", component_type="pl", params=params
    )
    assert isinstance(built[0], MixtureGeneral)
    assert built[0].event_shape == (2,)
    # the bounds default to the unit interval for both spins
    inner = built[0].component_distributions[0].base_dist
    assert_allclose(inner.low, [0.0, 0.0], rtol=1e-12)
    assert_allclose(inner.high, [1.0, 1.0], rtol=1e-12)


def test_create_gwtc4_effective_spin_skew_normal_models():
    built = create_gwtc4_effective_spin_skew_normal_models(
        N=2,
        parameter_name="chi_eff",
        component_type="g",
        params={
            "chi_eff_loc_g_0": 0.0,
            "chi_eff_scale_g_0": 0.3,
            "chi_eff_epsilon_g_0": 0.2,
            "chi_eff_loc_g_1": 0.1,
            "chi_eff_scale_g_1": 0.4,
            "chi_eff_epsilon_g_1": -0.2,
        },
    )
    assert len(built) == 2
    assert all(isinstance(d, GWTC4EffectiveSpinSkewNormalModel) for d in built)
    assert_allclose(built[1].loc, 0.1, rtol=1e-12)
    assert_allclose(built[1].epsilon, -0.2, rtol=1e-12)


###############################################################################
# the mass factories
###############################################################################


def test_create_powerlaw_primary_mass_ratios():
    built = create_powerlaw_primary_mass_ratios(
        N=1,
        parameter_name=None,
        component_type="pl",
        params={
            "alpha_pl_0": 1.5,
            "beta_pl_0": 1.0,
            "mmin_pl_0": 10.0,
            "mmax_pl_0": 60.0,
        },
    )
    assert isinstance(built[0], ExtendedSupportTransformedDistribution)
    assert isinstance(built[0].base_dist, PowerlawPrimaryMassRatio)
    assert built[0].event_shape == (2,)


def test_create_gaussian_primary_mass_ratio():
    built = create_gaussian_primary_mass_ratio(
        N=1,
        parameter_name=None,
        component_type="gpl",
        params={
            "m1_loc_gpl_0": 35.0,
            "m1_scale_gpl_0": 5.0,
            "beta_gpl_0": 1.0,
            "m1_low_gpl_0": 10.0,
            "m1_high_gpl_0": 60.0,
        },
    )
    assert isinstance(built[0].base_dist, GaussianPrimaryMassRatio)
    assert_allclose(built[0].base_dist.loc, 35.0, rtol=1e-12)


def test_create_smoothed_powerlaw_primary_mass_ratio():
    built = create_smoothed_powerlaw_primary_mass_ratio(
        N=1,
        parameter_name=None,
        component_type="spl",
        params={
            "m1_alpha_spl_0": 1.5,
            "beta_spl_0": 1.0,
            "m1_delta_spl_0": 4.0,
            "m2_delta_spl_0": 4.0,
            "m1_low_spl_0": 10.0,
            "m2_low_spl_0": 10.0,
            "m1_high_spl_0": 60.0,
        },
    )
    assert isinstance(built[0].base_dist, SmoothedPowerlawPrimaryMassRatio)
    assert_allclose(built[0].base_dist.alpha, 1.5, rtol=1e-12)


def test_create_smoothed_gaussian_primary_mass_ratio():
    built = create_smoothed_gaussian_primary_mass_ratio(
        N=1,
        parameter_name=None,
        component_type="g",
        params={
            "loc_g_0": 35.0,
            "scale_g_0": 5.0,
            "beta_g_0": 1.0,
            "m1min_g_0": 10.0,
            "m2min_g_0": 10.0,
            "mmax_g_0": 60.0,
            "delta_m1_g_0": 4.0,
            "delta_m2_g_0": 4.0,
        },
    )
    assert isinstance(built[0].base_dist, SmoothedGaussianPrimaryMassRatio)
    assert_allclose(built[0].base_dist.loc, 35.0, rtol=1e-12)


def test_create_smoothed_broken_powerlaws_mass_ratio_powerlaw():
    built = create_smoothed_broken_powerlaws_mass_ratio_powerlaw(
        N=1,
        parameter_name=None,
        component_type="bpl",
        params={
            "m1_alpha1_bpl_0": 1.5,
            "m1_alpha2_bpl_0": 3.0,
            "beta_bpl_0": 1.0,
            "m1_delta_bpl_0": 4.0,
            "m2_delta_bpl_0": 4.0,
            "m1_low_bpl_0": 10.0,
            "m2_low_bpl_0": 10.0,
            "m1_break_bpl_0": 30.0,
            "m1_high_bpl_0": 60.0,
        },
    )
    assert isinstance(built[0].base_dist, SmoothedBrokenPowerlawMassRatioPowerlaw)
    assert_allclose(built[0].base_dist.mbreak, 30.0, rtol=1e-12)


def test_create_generic_smoothed_powerlaw_mass_ratio_shares_one_primary_smoothing():
    primary = [DoublyTruncatedPowerLaw(alpha=-1.5, low=10.0, high=60.0)]
    built = create_generic_smoothed_powerlaw_mass_ratio(
        N=1,
        primary_mass_distributions=primary,
        parameter_name=None,
        component_type="spl",
        params={
            "m1_delta": 4.0,  # shared across every component type
            "beta_spl_0": 1.0,
            "m2_delta_spl_0": 3.0,
            "m2_low_spl_0": 10.0,
        },
    )
    assert isinstance(built[0], GenericSmoothedPowerlawMassRatio)
    assert built[0].primary_mass_distribution is primary[0]
    assert_allclose(built[0].delta_m1, 4.0, rtol=1e-12)
    assert_allclose(built[0].delta_m2, 3.0, rtol=1e-12)


def test_create_generic_smoothed_powerlaw_mass_ratio_requires_the_shared_smoothing():
    with pytest.raises(ValueError, match="Missing parameter m1_delta"):
        create_generic_smoothed_powerlaw_mass_ratio(
            N=1,
            primary_mass_distributions=[
                DoublyTruncatedPowerLaw(alpha=-1.5, low=10.0, high=60.0)
            ],
            parameter_name=None,
            component_type="spl",
            params={"beta_spl_0": 1.0, "m2_delta_spl_0": 3.0, "m2_low_spl_0": 10.0},
        )


###############################################################################
# every factory agrees on the component-count contract
###############################################################################


@pytest.mark.parametrize(
    "factory, parameter_name, component_type, per_component",
    [
        (create_beta_distributions, "a_1", "pl", ("a_1_mean_pl", "a_1_variance_pl")),
        (
            create_truncated_normal_distributions,
            "chi_p",
            "pl",
            ("chi_p_loc_pl", "chi_p_scale_pl"),
        ),
        (
            create_uniform_distributions,
            "phi_1",
            "pl",
            ("phi_1_low_pl", "phi_1_high_pl"),
        ),
        (
            create_powerlaw_redshift_model,
            "redshift",
            "pl",
            ("redshift_kappa_pl", "redshift_z_max_pl"),
        ),
    ],
)
def test_factories_build_exactly_n_components(
    factory, parameter_name, component_type, per_component
):
    for N in (0, 1, 4):
        params = {f"{name}_{i}": 0.5 for name in per_component for i in range(N)}
        built = factory(
            N=N,
            parameter_name=parameter_name,
            component_type=component_type,
            params=params,
        )
        assert len(built) == N
