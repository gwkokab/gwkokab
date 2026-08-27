# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The sub-population model.

Like :mod:`~gwkokab.models.hybrids._multisource`, this builds a
:class:`~gwkokab.models.utils.ScaledMixture` over three mass families -- a truncated
power law (``spl``), a broken power law (``bpl``) and a truncated normal (``gpl``) --
but it mixes them at the level of the *primary mass* rather than of the whole component.
The primary mass models are first combined into one
:class:`~numpyro.distributions.MixtureGeneral` with mixing weights ``lambda_<i>``, that
mixture is tapered by a shared Planck window of width ``m1_delta`` and normalised
numerically, and a single overall ``log_rate`` is then split across the components in
proportion to their weights.

So where the multi-source model has one rate per component, this one has one rate for
the population and a set of branching fractions -- which is the natural parameterisation
when the components are sub-populations of a single channel rather than separate
channels.
"""

from typing import Callable, Dict, Final, List, Literal, Optional, Sequence, Tuple

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
    create_generic_tilt_model,
    create_gwtc4_effective_spin_skew_normal_models,
    create_madau_dickinson_redshift_model,
    create_powerlaw_redshift_model,
    create_powerlaws,
    create_spin_magnitude_mixture_models,
    create_truncated_normal_distributions,
    create_two_truncated_normal_mixture,
    create_uniform_distributions,
)


_M1_GRID_SIZE: Final[int] = 1_000


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
    """Build the per-component marginals for every non-mass parameter that is enabled.

    Walks a fixed table of ``(flag, parameter name, factory)`` triples and, for each
    enabled flag, calls the factory in :mod:`~gwkokab.models.hybrids._ncombination` and
    appends its output to each component's list of marginals. The table order fixes
    the order of the marginals within a component, and hence the layout of the event
    axis of the resulting :class:`~gwkokab.models.utils.JointDistribution`.

    Parameters
    ----------
    N : int
        Number of components.
    component_type : Literal["spl", "bpl", "gpl"]
        Component tag identifying this family of components.
    mass_distributions : List[Distribution]
        Per-component lists of mass marginals, which the non-mass marginals are
        appended to.
    use_beta_spin_magnitude : bool
        Model both spin magnitudes with beta distributions.
    use_spin_magnitude_mixture : bool
        Model both spin magnitudes jointly with a two-truncated-normal mixture.
    use_truncated_normal_spin_x : bool
        Model both Cartesian ``x`` spin components with truncated normals.
    use_truncated_normal_spin_y : bool
        Model both Cartesian ``y`` spin components with truncated normals.
    use_truncated_normal_spin_z : bool
        Model both aligned spin components with truncated normals.
    use_chi_eff_mixture : bool
        Model the effective spin with a two-truncated-normal mixture.
    use_skew_normal_chi_eff : bool
        Model the effective spin with the GWTC-4 skew normal.
    use_truncated_normal_chi_p : bool
        Model the precessing spin with a truncated normal.
    use_tilt : bool
        Model both tilt cosines jointly with the generic tilt model.
    use_eccentricity_mixture : bool
        Model eccentricity with a two-truncated-normal mixture.
    use_eccentricity_powerlaw : bool
        Model eccentricity with a truncated power law.
    use_mean_anomaly : bool
        Model the mean anomaly with a uniform distribution.
    use_powerlaw_redshift : bool
        Model redshift with a power law rate evolution.
    use_madau_dickinson_redshift : bool
        Model redshift with the Madau-Dickinson rate evolution.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One list of marginals per component, mass marginals first.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
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
        (use_tilt, P.COS_TILT_1 + "_" + P.COS_TILT_2, create_generic_tilt_model),
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
    """Build the primary mass models and joint distributions for one mass family.

    The primary mass model is chosen by ``component_type``: a truncated power law
    (``spl``), a broken power law (``bpl``) or a truncated normal (``gpl``). Each is then
    wrapped in a :class:`~gwkokab.models.mass.GenericSmoothedPowerlawMassRatio` that adds
    the conditional mass ratio power law, the enabled non-mass marginals are appended,
    and each component is packed into a
    :class:`~gwkokab.models.utils.JointDistribution`.

    The bare primary mass models are returned alongside, because
    :func:`SubPopulationModel` needs them to build the primary mass mixture that carries
    the branching fractions.

    Parameters
    ----------
    N : int
        Number of components in this family. Zero yields two empty lists.
    component_type : Literal["spl", "bpl", "gpl"]
        Which primary mass model this family uses; also the tag in the hyper-parameter
        names.
    use_beta_spin_magnitude : bool
        Model both spin magnitudes with beta distributions.
    use_spin_magnitude_mixture : bool
        Model both spin magnitudes jointly with a two-truncated-normal mixture.
    use_truncated_normal_spin_x : bool
        Model both Cartesian ``x`` spin components with truncated normals.
    use_truncated_normal_spin_y : bool
        Model both Cartesian ``y`` spin components with truncated normals.
    use_truncated_normal_spin_z : bool
        Model both aligned spin components with truncated normals.
    use_chi_eff_mixture : bool
        Model the effective spin with a two-truncated-normal mixture.
    use_skew_normal_chi_eff : bool
        Model the effective spin with the GWTC-4 skew normal.
    use_truncated_normal_chi_p : bool
        Model the precessing spin with a truncated normal.
    use_tilt : bool
        Model both tilt cosines jointly with the generic tilt model.
    use_eccentricity_mixture : bool
        Model eccentricity with a two-truncated-normal mixture.
    use_eccentricity_powerlaw : bool
        Model eccentricity with a truncated power law.
    use_mean_anomaly : bool
        Model the mean anomaly with a uniform distribution.
    use_powerlaw_redshift : bool
        Model redshift with a power law rate evolution.
    use_madau_dickinson_redshift : bool
        Model redshift with the Madau-Dickinson rate evolution.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    Tuple[List[Distribution], List[JointDistribution]]
        The bare primary mass distributions, and the full joint distribution of each
        component.
    """
    if N == 0:
        return [], []

    if component_type == "spl":
        _mass_distributions = create_powerlaws(
            N=N,
            parameter_name="m1",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )

    if component_type == "bpl":
        _mass_distributions = create_broken_powerlaws(
            N=N,
            parameter_name="m1",
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
    r"""Create a mixture of three mass sub-populations sharing a single overall rate.

    Components are laid out in the order ``spl``, ``bpl``, ``gpl``. Their primary mass
    models are combined into one mixture with weights ``lambda_<i>`` -- the last weight
    is fixed by closure, so only :math:`N-1` are read -- tapered above ``m1min`` by a
    Planck window of width ``m1_delta`` and normalised numerically on a grid spanning
    ``[m1min, m1max]``. The per-component log scales are then
    :math:`\log\mathcal{R} - \log Z + \log\lambda_i`, so the mixture integrates to the
    single population rate ``log_rate``.

    Hyper-parameters follow the naming convention
    ``<role>_<component tag>_<component index>``, and are passed through ``**params``.

    Parameters
    ----------
    N_spl : int
        Number of power law sub-populations.
    N_bpl : int
        Number of broken power law sub-populations.
    N_gpl : int
        Number of truncated normal sub-populations.
    use_beta_spin_magnitude : bool
        Model both spin magnitudes with beta distributions. Defaults to :data:`False`.
    use_spin_magnitude_mixture : bool
        Model both spin magnitudes jointly with a two-truncated-normal mixture. Defaults to :data:`False`.
    use_truncated_normal_spin_x : bool
        Model both Cartesian ``x`` spin components with truncated normals. Defaults to :data:`False`.
    use_truncated_normal_spin_y : bool
        Model both Cartesian ``y`` spin components with truncated normals. Defaults to :data:`False`.
    use_truncated_normal_spin_z : bool
        Model both aligned spin components with truncated normals. Defaults to :data:`False`.
    use_chi_eff_mixture : bool
        Model the effective spin with a two-truncated-normal mixture. Defaults to :data:`False`.
    use_skew_normal_chi_eff : bool
        Model the effective spin with the GWTC-4 skew normal. Defaults to :data:`False`.
    use_truncated_normal_chi_p : bool
        Model the precessing spin with a truncated normal. Defaults to :data:`False`.
    use_tilt : bool
        Model both tilt cosines jointly with the generic tilt model. Defaults to :data:`False`.
    use_eccentricity_mixture : bool
        Model eccentricity with a two-truncated-normal mixture. Defaults to :data:`False`.
    use_eccentricity_powerlaw : bool
        Model eccentricity with a truncated power law. Defaults to :data:`False`.
    use_mean_anomaly : bool
        Model the mean anomaly with a uniform distribution. Defaults to :data:`False`.
    use_powerlaw_redshift : bool
        Model redshift with a power law rate evolution. Defaults to :data:`False`.
    use_madau_dickinson_redshift : bool
        Model redshift with the Madau-Dickinson rate evolution. Defaults to :data:`False`.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.
    **params : Array
        The population hyper-parameters. Beyond the per-component ones, this must carry
        ``lambda_0`` .. ``lambda_{N-2}``, ``m1_delta``, ``log_rate``, ``m1min`` and
        ``m1max``.

    Returns
    -------
    ScaledMixture
        The population model, whose components carry log rates rather than normalised
        weights.

    Raises
    ------
    KeyError
        If one of the mixture-level hyper-parameters is missing from ``params``.
    ValueError
        If a required per-component hyper-parameter is missing from ``params``.
    """
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

    delta_m1 = params.pop("m1_delta")
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
