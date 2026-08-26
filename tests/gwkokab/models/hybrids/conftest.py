# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import math
from typing import Callable, Dict, Iterable

import pytest


_TWO_PI = 2.0 * math.pi

# Every hyper-parameter name follows ``<role>_<component type>_<index>``, so a table
# keyed by role is enough to build a complete, physically sensible parameter set for any
# combination of ``use_*`` switches. Keeping the table keyed by role (rather than by full
# name) means it stays valid as the number of components changes.
_ROLE_VALUES: Dict[str, float] = {
    # rates and mixing weights
    "log_rate": -1.0,
    "lambda": 0.25,
    # powerlaw primary mass and mass ratio
    "alpha": 1.5,
    "beta": 1.0,
    "mmin": 10.0,
    "mmax": 60.0,
    "m1min": 10.0,
    "m1max": 60.0,
    # gaussian masses
    "m1_loc": 35.0,
    "m2_loc": 30.0,
    "m1_scale": 5.0,
    "m2_scale": 5.0,
    "m1_low": 10.0,
    "m2_low": 10.0,
    "m1_high": 60.0,
    "m2_high": 60.0,
    # smoothed and broken powerlaw masses
    "m1_alpha": 1.5,
    "m1_alpha1": 1.5,
    "m1_alpha2": 3.0,
    "m1_break": 30.0,
    "m1_delta": 4.0,
    "m2_delta": 4.0,
    # spin magnitude, beta parameterisation
    "a_1_mean": 0.3,
    "a_2_mean": 0.3,
    "a_1_variance": 0.02,
    "a_2_variance": 0.02,
    # spin magnitude, mixture parameterisation
    "a_zeta": 0.4,
    "a_1_comp1_loc": 0.2,
    "a_1_comp1_scale": 0.2,
    "a_1_comp1_low": 0.0,
    "a_1_comp1_high": 1.0,
    "a_1_comp2_loc": 0.7,
    "a_1_comp2_scale": 0.1,
    "a_1_comp2_low": 0.0,
    "a_1_comp2_high": 1.0,
    "a_2_comp1_loc": 0.2,
    "a_2_comp1_scale": 0.2,
    "a_2_comp1_low": 0.0,
    "a_2_comp1_high": 1.0,
    "a_2_comp2_loc": 0.7,
    "a_2_comp2_scale": 0.1,
    "a_2_comp2_low": 0.0,
    "a_2_comp2_high": 1.0,
    # cartesian spin components
    **{
        f"spin_{i}{axis}_{key}": value
        for i in (1, 2)
        for axis in "xyz"
        for key, value in (("loc", 0.0), ("scale", 0.3), ("low", -1.0), ("high", 1.0))
    },
    # effective and precessing spin
    "chi_eff_comp1_loc": -0.1,
    "chi_eff_comp1_scale": 0.3,
    "chi_eff_comp1_low": -1.0,
    "chi_eff_comp1_high": 1.0,
    "chi_eff_comp2_loc": 0.2,
    "chi_eff_comp2_scale": 0.1,
    "chi_eff_comp2_low": -1.0,
    "chi_eff_comp2_high": 1.0,
    "chi_eff_zeta": 0.4,
    "chi_eff_loc": 0.05,
    "chi_eff_scale": 0.3,
    "chi_eff_epsilon": 0.2,
    "chi_p_loc": 0.2,
    "chi_p_scale": 0.2,
    "chi_p_low": 0.0,
    "chi_p_high": 1.0,
    # tilts
    "cos_tilt_zeta": 0.4,
    "cos_tilt_1_loc": 1.0,
    "cos_tilt_1_scale": 0.5,
    "cos_tilt_1_low": -1.0,
    "cos_tilt_1_high": 1.0,
    "cos_tilt_2_loc": 1.0,
    "cos_tilt_2_scale": 0.5,
    "cos_tilt_2_low": -1.0,
    "cos_tilt_2_high": 1.0,
    # angles and time
    "phi_1_low": 0.0,
    "phi_1_high": _TWO_PI,
    "phi_2_low": 0.0,
    "phi_2_high": _TWO_PI,
    "phi_12_low": 0.0,
    "phi_12_high": _TWO_PI,
    "phi_orb_low": 0.0,
    "phi_orb_high": _TWO_PI,
    "psi_low": 0.0,
    "psi_high": math.pi,
    "ra_low": 0.0,
    "ra_high": _TWO_PI,
    "dec_low": -1.0,
    "dec_high": 1.0,
    "cos_iota_low": -1.0,
    "cos_iota_high": 1.0,
    "mean_anomaly_low": 0.0,
    "mean_anomaly_high": _TWO_PI,
    "detection_time_low": 0.0,
    "detection_time_high": 1.0,
    # eccentricity
    "eccentricity_comp1_loc": 0.1,
    "eccentricity_comp1_scale": 0.2,
    "eccentricity_comp1_low": 0.0,
    "eccentricity_comp1_high": 1.0,
    "eccentricity_comp2_loc": 0.6,
    "eccentricity_comp2_scale": 0.1,
    "eccentricity_comp2_low": 0.0,
    "eccentricity_comp2_high": 1.0,
    "eccentricity_zeta": 0.4,
    "eccentricity_alpha": -1.5,
    "eccentricity_low": 1e-3,
    "eccentricity_high": 1.0,
    # redshift
    "redshift_kappa": 2.7,
    "redshift_z_max": 1.5,
    "redshift_gamma": 2.9,
    "redshift_z_peak": 0.5,
}

_COMPONENT_TYPES = ("pl", "g", "spl", "bpl", "gpl", "gg")


def _role_of(name: str) -> str:
    """Strip the trailing component index and component-type tag from `name`."""
    head, _, tail = name.rpartition("_")
    if tail.isdigit():
        name = head
    head, _, tail = name.rpartition("_")
    if tail in _COMPONENT_TYPES:
        name = head
    return name


@pytest.fixture(scope="session")
def hyper_parameters() -> Callable[[Iterable[str]], Dict[str, float]]:
    """Build a value for every hyper-parameter name a model family asks for.

    The analysis-layer ``*Core`` classes expose exactly the names their model factory
    consumes, so feeding this fixture ``core.model_parameters`` both builds a valid
    model and checks that the two lists have not drifted apart.
    """

    def _build(names: Iterable[str]) -> Dict[str, float]:
        params = {}
        for name in names:
            role = _role_of(name)
            if role not in _ROLE_VALUES:
                raise KeyError(
                    f"no test value known for hyper-parameter {name!r} (role {role!r})"
                )
            params[name] = _ROLE_VALUES[role]
        return params

    return _build
