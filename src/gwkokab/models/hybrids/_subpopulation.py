# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from jax import numpy as jnp
from jaxtyping import Array
from numpyro.distributions import (
    CategoricalProbs,
    constraints,
    Distribution,
    MixtureGeneral,
)

from ...parameters import Parameters as P
from ...utils.kernel import log_planck_taper_window
from ..constraints import any_constraint
from ..utils import JointDistribution, ScaledMixture
from ._ncombination import (
    combine_distributions,
    create_beta_distributions,
    create_broken_powerlaws,
    create_generic_powerlaws,
    create_generic_smoothed_powerlaw_mass_ratio,
    create_gwtc4_effective_spin_skew_normal_models,
    create_independent_spin_orientation_gaussian_isotropic,
    create_madau_dickinson_redshift_model,
    create_powerlaw_redshift_model,
    create_powerlaws,
    create_spin_magnitude_mixture_models,
    create_truncated_normal_distributions,
    create_two_truncated_normal_mixture,
    create_uniform_distributions,
)
from ._utils import _M1_GRID_SIZE


def _build_non_mass_distributions(
    N: int,
    component_type: Literal["spl", "bpl", "gpl"],
    mass_distributions: List[Distribution],
    use_beta_spin_magnitude: bool,
    use_spin_magnitude_mixture: bool,
    use_truncated_normal_spin_x: bool,
    use_truncated_normal_spin_y: bool,
    use_truncated_normal_spin_z: bool,
    use_chi_eff_mixture: bool,
    use_skew_normal_chi_eff: bool,
    use_truncated_normal_chi_p: bool,
    use_tilt: bool,
    use_eccentricity_mixture: bool,
    use_eccentricity_powerlaw: bool,
    use_mean_anomaly: bool,
    use_powerlaw_redshift: bool,
    use_madau_dickinson_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    build_distributions = mass_distributions
    # fmt: off
    _info_collection: Sequence[Tuple[bool, str, Callable[..., Sequence[Distribution]]]] = [
        (use_beta_spin_magnitude, P.PRIMARY_SPIN_MAGNITUDE, create_beta_distributions),
        (use_beta_spin_magnitude, P.SECONDARY_SPIN_MAGNITUDE, create_beta_distributions),
        # combined spin magnitude distribution
        (use_spin_magnitude_mixture, P.PRIMARY_SPIN_MAGNITUDE + "_" + P.SECONDARY_SPIN_MAGNITUDE, create_spin_magnitude_mixture_models),
        (use_truncated_normal_spin_x, P.PRIMARY_SPIN_X, create_truncated_normal_distributions),
        (use_truncated_normal_spin_x, P.SECONDARY_SPIN_X, create_truncated_normal_distributions),
        (use_truncated_normal_spin_y, P.PRIMARY_SPIN_Y, create_truncated_normal_distributions),
        (use_truncated_normal_spin_y, P.SECONDARY_SPIN_Y, create_truncated_normal_distributions),
        (use_truncated_normal_spin_z, P.PRIMARY_SPIN_Z, create_truncated_normal_distributions),
        (use_truncated_normal_spin_z, P.SECONDARY_SPIN_Z, create_truncated_normal_distributions),
        (use_chi_eff_mixture, P.EFFECTIVE_SPIN, create_two_truncated_normal_mixture),
        (use_skew_normal_chi_eff, P.EFFECTIVE_SPIN, create_gwtc4_effective_spin_skew_normal_models),
        (use_truncated_normal_chi_p, P.PRECESSING_SPIN, create_truncated_normal_distributions),
        # combined tilt distribution
        (use_tilt, P.COS_TILT_1 + "_" + P.COS_TILT_2, create_independent_spin_orientation_gaussian_isotropic),
        (use_eccentricity_mixture, P.ECCENTRICITY, create_two_truncated_normal_mixture),
        (use_eccentricity_powerlaw, P.ECCENTRICITY, create_generic_powerlaws),
        (use_mean_anomaly, P.MEAN_ANOMALY, create_uniform_distributions),
        (use_powerlaw_redshift, P.REDSHIFT, create_powerlaw_redshift_model),
        (use_madau_dickinson_redshift, P.REDSHIFT, create_madau_dickinson_redshift_model),
    ]
    # fmt: on

    # Iterate over the list of tuples and build distributions
    for use, param_name, build_func in _info_collection:
        if use:
            distributions = build_func(
                N=N,
                parameter_name=param_name,
                component_type=component_type,
                params=params,
                validate_args=validate_args,
            )
            build_distributions = combine_distributions(
                build_distributions, distributions
            )

    return build_distributions


def _build_component_distributions(
    N: int,
    component_type: Literal["spl", "bpl", "gpl"],
    use_beta_spin_magnitude: bool,
    use_spin_magnitude_mixture: bool,
    use_truncated_normal_spin_x: bool,
    use_truncated_normal_spin_y: bool,
    use_truncated_normal_spin_z: bool,
    use_chi_eff_mixture: bool,
    use_skew_normal_chi_eff: bool,
    use_truncated_normal_chi_p: bool,
    use_tilt: bool,
    use_eccentricity_mixture: bool,
    use_eccentricity_powerlaw: bool,
    use_mean_anomaly: bool,
    use_powerlaw_redshift: bool,
    use_madau_dickinson_redshift: bool,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> Tuple[List[Distribution], List[JointDistribution]]:
    if N == 0:
        return [], []

    if component_type == "spl":
        _mass_distributions = create_powerlaws(
            N=N,
            parameter_name=None,  # type: ignore # unused parameter
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )

    if component_type == "bpl":
        _mass_distributions = create_broken_powerlaws(
            N=N,
            parameter_name=None,  # type: ignore # unused parameter
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )

    if component_type == "gpl":
        _mass_distributions = create_truncated_normal_distributions(
            N=N,
            parameter_name="m1",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )

    _m1q_distributions = create_generic_smoothed_powerlaw_mass_ratio(
        N=N,
        primary_mass_distributions=_mass_distributions,
        parameter_name=None,  # type: ignore # unused parameter
        component_type=component_type,
        params=params,
        validate_args=validate_args,
    )

    mass_distributions = [[d] for d in _m1q_distributions]

    build_distributions = _build_non_mass_distributions(
        N=N,
        component_type=component_type,
        mass_distributions=mass_distributions,
        use_beta_spin_magnitude=use_beta_spin_magnitude,
        use_spin_magnitude_mixture=use_spin_magnitude_mixture,
        use_truncated_normal_spin_x=use_truncated_normal_spin_x,
        use_truncated_normal_spin_y=use_truncated_normal_spin_y,
        use_truncated_normal_spin_z=use_truncated_normal_spin_z,
        use_chi_eff_mixture=use_chi_eff_mixture,
        use_skew_normal_chi_eff=use_skew_normal_chi_eff,
        use_truncated_normal_chi_p=use_truncated_normal_chi_p,
        use_tilt=use_tilt,
        use_eccentricity_mixture=use_eccentricity_mixture,
        use_eccentricity_powerlaw=use_eccentricity_powerlaw,
        use_mean_anomaly=use_mean_anomaly,
        use_powerlaw_redshift=use_powerlaw_redshift,
        use_madau_dickinson_redshift=use_madau_dickinson_redshift,
        params=params,
        validate_args=validate_args,
    )

    return _mass_distributions, [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def SubPopulationModel(
    N_spl: int,
    N_bpl: int,
    N_gpl: int,
    use_beta_spin_magnitude: bool = False,
    use_spin_magnitude_mixture: bool = False,
    use_truncated_normal_spin_x: bool = False,
    use_truncated_normal_spin_y: bool = False,
    use_truncated_normal_spin_z: bool = False,
    use_chi_eff_mixture: bool = False,
    use_skew_normal_chi_eff: bool = False,
    use_truncated_normal_chi_p: bool = False,
    use_tilt: bool = False,
    use_eccentricity_mixture: bool = False,
    use_eccentricity_powerlaw: bool = False,
    use_mean_anomaly: bool = False,
    use_powerlaw_redshift: bool = False,
    use_madau_dickinson_redshift: bool = False,
    *,
    validate_args=None,
    **params,
) -> ScaledMixture:
    component_dists = []
    mass_dist = []
    for component_type, N in zip(("spl", "bpl", "gpl"), (N_spl, N_bpl, N_gpl)):
        m_dist, _component_dists = _build_component_distributions(
            N=N,
            component_type=component_type,
            use_beta_spin_magnitude=use_beta_spin_magnitude,
            use_spin_magnitude_mixture=use_spin_magnitude_mixture,
            use_truncated_normal_spin_x=use_truncated_normal_spin_x,
            use_truncated_normal_spin_y=use_truncated_normal_spin_y,
            use_truncated_normal_spin_z=use_truncated_normal_spin_z,
            use_chi_eff_mixture=use_chi_eff_mixture,
            use_skew_normal_chi_eff=use_skew_normal_chi_eff,
            use_truncated_normal_chi_p=use_truncated_normal_chi_p,
            use_tilt=use_tilt,
            use_eccentricity_mixture=use_eccentricity_mixture,
            use_eccentricity_powerlaw=use_eccentricity_powerlaw,
            use_mean_anomaly=use_mean_anomaly,
            use_powerlaw_redshift=use_powerlaw_redshift,
            use_madau_dickinson_redshift=use_madau_dickinson_redshift,
            params=params,
            validate_args=validate_args,
        )
        mass_dist.extend(m_dist)
        component_dists.extend(_component_dists)

    N = N_spl + N_bpl + N_gpl
    _lambdas = [params.pop(f"lambda_{i}") for i in range(N - 1)]
    _lambdas.append(1.0 - sum(_lambdas))
    lambdas = jnp.stack(_lambdas, axis=-1)

    delta_m1 = params.pop("delta_m1")
    log_rate = params.pop("log_rate")
    m1max = params.pop("m1max")
    m1min = params.pop("m1min")

    mixing_distribution = CategoricalProbs(probs=lambdas, validate_args=validate_args)
    mass_dist_mixture = MixtureGeneral(
        mixing_distribution,
        mass_dist,
        support=constraints.interval(m1min, m1max),
        validate_args=validate_args,
    )

    mm = jnp.linspace(m1min, m1max, _M1_GRID_SIZE)
    safe_delta_m = jnp.where(delta_m1 <= 0.0, 1.0, delta_m1)
    _log_prob_m1 = mass_dist_mixture.log_prob(mm) + log_planck_taper_window(
        (mm - m1min) / safe_delta_m
    )
    _prob_m1 = jnp.where(delta_m1 <= 0.0, 0.0, jnp.exp(_log_prob_m1))
    Z = jnp.trapezoid(_prob_m1, mm, axis=0)
    logZ = jnp.log(Z)

    safe_lambdas = jnp.where(lambdas <= 0.0, 1.0, lambdas)
    safe_log_lambdas = jnp.where(lambdas <= 0.0, -jnp.inf, jnp.log(safe_lambdas))
    log_scales = log_rate - logZ + safe_log_lambdas

    return ScaledMixture(
        log_scales,
        component_dists,
        support=any_constraint([
            component_dist._support for component_dist in component_dists
        ]),
        validate_args=validate_args,
    )
