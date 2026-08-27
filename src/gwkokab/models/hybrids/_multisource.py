# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The multi-source population model.

A :class:`~gwkokab.models.utils.ScaledMixture` over four kinds of mass component,
meant to describe a population drawn from several astrophysical formation channels
at once:

- ``spl`` -- smoothed power law primary mass with a conditional mass ratio power law;
- ``bpl`` -- smoothed *broken* power law, same conditional mass ratio;
- ``gpl`` -- Gaussian primary mass, same conditional mass ratio;
- ``gg`` -- independent truncated normals on :math:`m_1` and :math:`m_2`.

Components are laid out in that order, and their per-component log rates
``log_rate_<index>`` are indexed across the whole mixture. Non-mass parameters are
switched on by the ``use_*`` flags of :func:`MultiSourceModel`.

See Also
--------
gwkokab.models.hybrids._subpopulation :
    Same component families, but mixed at the level of the primary mass rather than
    the full component.
"""

from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from jax import numpy as jnp, tree as jtr
from jaxtyping import Array
from numpyro.distributions import Distribution

from ...parameters import Parameters as P
from ..constraints import any_constraint
from ..utils import (
    ExtendedSupportTransformedDistribution,
    JointDistribution,
    ScaledMixture,
)
from ._ncombination import (
    combine_distributions,
    create_beta_distributions,
    create_gaussian_primary_mass_ratio,
    create_generic_powerlaws,
    create_generic_tilt_model,
    create_gwtc4_effective_spin_skew_normal_models,
    create_madau_dickinson_redshift_model,
    create_powerlaw_redshift_model,
    create_smoothed_broken_powerlaws_mass_ratio_powerlaw,
    create_smoothed_powerlaw_primary_mass_ratio,
    create_spin_magnitude_mixture_models,
    create_truncated_normal_distributions,
    create_two_truncated_normal_mixture,
    create_uniform_distributions,
)


def _build_non_mass_distributions(
    N: int,
    component_type: Literal["spl", "bpl", "gpl", "gg"],
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
    component_type : Literal["spl", "bpl", "gpl", "gg"]
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
    component_type: Literal["spl", "bpl", "gpl", "gg"],
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
) -> List[JointDistribution]:
    """Build the joint distribution for each component of one mass family.

    The mass model is chosen by ``component_type``: a smoothed power law (``spl``), a
    smoothed broken power law (``bpl``), a Gaussian primary mass (``gpl``), or
    independent truncated normals on the two component masses (``gg``). The enabled
    non-mass marginals are then appended and each component is packed into a
    :class:`~gwkokab.models.utils.JointDistribution`.

    Parameters
    ----------
    N : int
        Number of components in this family. Zero yields an empty list.
    component_type : Literal["spl", "bpl", "gpl", "gg"]
        Which mass model this family uses; also the tag in the hyper-parameter names.
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
    List[JointDistribution]
        One joint distribution per component of this family.
    """
    if N == 0:
        return []
    if component_type == "spl":
        powerlaws = create_smoothed_powerlaw_primary_mass_ratio(
            N=N,
            parameter_name=None,  # type: ignore # unused parameter
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        mass_distributions = jtr.map(
            lambda powerlaw: [powerlaw],
            powerlaws,
            is_leaf=lambda x: isinstance(x, ExtendedSupportTransformedDistribution),
        )
    if component_type == "bpl":
        powerlaws = create_smoothed_broken_powerlaws_mass_ratio_powerlaw(
            N=N,
            parameter_name=None,  # type: ignore # unused parameter
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        mass_distributions = jtr.map(
            lambda powerlaw: [powerlaw],
            powerlaws,
            is_leaf=lambda x: isinstance(x, ExtendedSupportTransformedDistribution),
        )

    if component_type == "gpl":
        powerlaws = create_gaussian_primary_mass_ratio(
            N=N,
            parameter_name=None,  # type: ignore # unused parameter
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        mass_distributions = jtr.map(
            lambda powerlaw: [powerlaw],
            powerlaws,
            is_leaf=lambda x: isinstance(x, ExtendedSupportTransformedDistribution),
        )

    if component_type == "gg":
        m1_dists = create_truncated_normal_distributions(
            N=N,
            parameter_name="m1",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )
        m2_dists = create_truncated_normal_distributions(
            N=N,
            parameter_name="m2",
            component_type=component_type,
            params=params,
            validate_args=validate_args,
        )

        mass_distributions = jtr.map(
            lambda m1, m2: [JointDistribution(m1, m2, validate_args=validate_args)],
            m1_dists,
            m2_dists,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

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

    return [
        JointDistribution(*dists, validate_args=validate_args)
        for dists in build_distributions
    ]


def MultiSourceModel(
    N_spl: int,
    N_bpl: int,
    N_gpl: int,
    N_gg: int,
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
    """Create a multi-source mixture of four mass component families.

    Components are laid out in the order ``spl``, ``bpl``, ``gpl``, ``gg``, and the log
    rate of the :math:`i`-th component of the whole mixture is read from the
    hyper-parameter ``log_rate_<i>``, in natural logarithm. Because the component
    families have genuinely different supports, the mixture's support is the union of
    them (:func:`~gwkokab.models.constraints.any_constraint`) rather than a shared one.

    Hyper-parameters follow the naming convention
    ``<role>_<component tag>_<component index>``, and are passed through ``**params``.

    Parameters
    ----------
    N_spl : int
        Number of smoothed power law components.
    N_bpl : int
        Number of smoothed broken power law components.
    N_gpl : int
        Number of Gaussian primary mass components.
    N_gg : int
        Number of Gaussian-Gaussian components.
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
        The population hyper-parameters, including one ``log_rate_<i>`` per component.

    Returns
    -------
    ScaledMixture
        The population model, whose components carry log rates rather than normalised
        weights.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    component_dists = []
    for component_type, N in zip(
        ["spl", "bpl", "gpl", "gg"], [N_spl, N_bpl, N_gpl, N_gg]
    ):
        component_dists += _build_component_distributions(
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

    N = N_spl + N_bpl + N_gpl + N_gg
    log_rates = jnp.stack([params[f"log_rate_{i}"] for i in range(N)], axis=-1)

    return ScaledMixture(
        log_rates,
        component_dists,
        support=any_constraint([
            component_dists.support for component_dists in component_dists
        ]),
        validate_args=validate_args,
    )
