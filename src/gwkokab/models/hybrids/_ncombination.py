# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

r"""Per-parameter distribution factories shared by the hybrid model families.

Every ``create_*`` function here has the same shape: given a component count
:math:`N`, a parameter name, a component tag and a flat dictionary of
hyper-parameters, it returns a list of :math:`N` distributions -- one per mixture
component. Each factory knows which hyper-parameters it needs and looks them up by
the package-wide naming convention ``<role>_<component tag>_<index>``, so
``alpha_pl_0`` is the power-law index of the zeroth power-law component and
``spin_1z_loc_g_2`` the spin location of the second Gaussian one.

The family modules -- :mod:`~gwkokab.models.hybrids._npowerlawmgaussian`,
:mod:`~gwkokab.models.hybrids._multisource`,
:mod:`~gwkokab.models.hybrids._subpopulation` -- call these factories once per
physical parameter that was switched on, then :func:`combine_distributions` zips the
results together into per-component lists that become
:class:`~gwkokab.models.utils.JointDistribution`\ s.

Adding a new physical parameter to a family means adding a factory here and wiring
it into that family's ``_build_*_distributions``.
"""

from typing import Dict, List, Optional, TypeVar

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro.distributions import (
    Beta,
    Distribution,
    MixtureGeneral,
    TruncatedNormal,
    Uniform,
)

from ...parameters import Parameters as P
from ..mass import (
    BrokenPowerlaw,
    GaussianPrimaryMassRatio,
    GenericSmoothedPowerlawMassRatio,
    PowerlawPrimaryMassRatio,
    SmoothedBrokenPowerlawMassRatioPowerlaw,
    SmoothedGaussianPrimaryMassRatio,
    SmoothedPowerlawPrimaryMassRatio,
)
from ..redshift import MadauDickinsonRedshiftModel, PowerlawRedshiftModel
from ..spin import BetaFromMeanVar, GenericTiltModel, GWTC4EffectiveSpinSkewNormalModel
from ..sundry import NDTwoTruncatedNormalMixture, TwoTruncatedNormalMixture
from ..transformations import PrimaryMassAndMassRatioToComponentMassesTransform
from ..utils import DoublyTruncatedPowerLaw, ExtendedSupportTransformedDistribution


__all__ = [
    "combine_distributions",
    "create_beta_distributions",
    "create_broken_powerlaws",
    "create_gaussian_primary_mass_ratio",
    "create_generic_powerlaws",
    "create_generic_smoothed_powerlaw_mass_ratio",
    "create_gwtc4_effective_spin_skew_normal_models",
    "create_madau_dickinson_redshift_model",
    "create_generic_tilt_model",
    "create_powerlaw_primary_mass_ratios",
    "create_powerlaw_redshift_model",
    "create_powerlaws",
    "create_smoothed_broken_powerlaws_mass_ratio_powerlaw",
    "create_smoothed_gaussian_primary_mass_ratio",
    "create_smoothed_powerlaw_primary_mass_ratio",
    "create_spin_magnitude_mixture_models",
    "create_truncated_normal_distributions",
    "create_two_truncated_normal_mixture",
    "create_uniform_distributions",
]


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _get_parameter(
    params: Dict[_KT, _VT],
    name: _KT,
    is_necessary: bool = True,
    default: Optional[_VT] = None,
) -> Optional[_VT]:
    """Look a hyper-parameter up by name, with optional default and requirement.

    Parameters
    ----------
    params : Dict[_KT, _VT]
        The flat hyper-parameter dictionary.
    name : _KT
        The name to look up.
    is_necessary : bool, optional
        Whether a missing value is an error. Defaults to :data:`True`.
    default : Optional[_VT], optional
        Value to fall back on when ``name`` is absent. Defaults to :data:`None`.

    Returns
    -------
    Optional[_VT]
        The looked-up value, the default, or :data:`None` when the parameter is absent
        and optional.

    Raises
    ------
    ValueError
        If ``name`` is absent, no default was given and ``is_necessary`` is
        :data:`True`.
    """
    if (value := params.get(name, None)) is not None:
        return value
    if default is not None:
        return default
    if is_necessary:
        raise ValueError(f"Missing parameter {name}")
    return None


def combine_distributions(
    base_dists: List[List[Distribution]], add_dists: List[Distribution]
):
    """Append one extra distribution to each component's list of marginals.

    Used to bolt an additional physical parameter -- spin, tilt, eccentricity -- onto a
    set of per-component distribution lists that already carry the mass model.

    Parameters
    ----------
    base_dists : List[List[Distribution]]
        One list of marginals per mixture component.
    add_dists : List[Distribution]
        One additional marginal per mixture component, in the same order.

    Returns
    -------
    List[List[Distribution]]
        New per-component lists, each with its extra marginal appended.
    """
    return [dists + [add_dist] for dists, add_dist in zip(base_dists, add_dists)]


def create_beta_distributions(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Beta]:
    """Build per-component beta distributions from mean and variance.

    Uses the moment parameterisation of :func:`~gwkokab.models.spin.BetaFromMeanVar`,
    which is easier to place priors on than the concentration parameterisation.

    Reads the hyper-parameters ``<parameter_name>_mean_<component_type>``, ``<parameter_name>_variance_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the physical parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Beta]
        One :class:`~numpyro.distributions.Beta` per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    mean_name = f"{parameter_name}_mean_{component_type}"
    variance_name = f"{parameter_name}_variance_{component_type}"

    return [
        BetaFromMeanVar(
            mean=_get_parameter(params, f"{mean_name}_{i}"),  # type: ignore
            variance=_get_parameter(params, f"{variance_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_truncated_normal_distributions(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    r"""Build per-component truncated normal distributions.

    The truncation bounds are optional: omitting ``low`` or ``high`` leaves that side
    unbounded, since :class:`~numpyro.distributions.TruncatedNormal` defaults them to
    :math:`\mp\infty`.

    Reads the hyper-parameters ``<parameter_name>_loc_<component_type>``, ``<parameter_name>_scale_<component_type>``, ``<parameter_name>_low_<component_type>``, ``<parameter_name>_high_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the physical parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One :class:`~numpyro.distributions.TruncatedNormal` per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    loc_name = f"{parameter_name}_loc_{component_type}"
    scale_name = f"{parameter_name}_scale_{component_type}"
    low_name = f"{parameter_name}_low_{component_type}"
    high_name = f"{parameter_name}_high_{component_type}"

    # fmt: off
    return [
        TruncatedNormal(
            loc=_get_parameter(params, f"{loc_name}_{i}"),  # type: ignore
            scale=_get_parameter(params, f"{scale_name}_{i}"),  # type: ignore
            low=_get_parameter(params, f"{low_name}_{i}", is_necessary=False),  # type: ignore
            high=_get_parameter(params, f"{high_name}_{i}", is_necessary=False),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]
    # fmt: on


def create_powerlaw_primary_mass_ratios(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[ExtendedSupportTransformedDistribution]:
    """Build per-component power law mass models in component-mass coordinates.

    Each component is a :class:`~gwkokab.models.mass.PowerlawPrimaryMassRatio` in
    :math:`(m_1, q)` coordinates, pushed to :math:`(m_1, m_2)` by
    :class:`~gwkokab.models.transformations.PrimaryMassAndMassRatioToComponentMassesTransform`
    while keeping its mass sandwich support.

    Reads the hyper-parameters ``alpha_<component_type>``, ``beta_<component_type>``, ``mmin_<component_type>``, ``mmax_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Unused; the hyper-parameter names of this factory are fixed.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[ExtendedSupportTransformedDistribution]
        One transformed mass model per component, over :math:`(m_1, m_2)`.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    powerlaws_collection = []

    alpha_name = "alpha_" + component_type
    beta_name = "beta_" + component_type
    mmin_name = "mmin_" + component_type
    mmax_name = "mmax_" + component_type

    for i in range(N):
        powerlaw = PowerlawPrimaryMassRatio(
            alpha=_get_parameter(params, f"{alpha_name}_{i}"),  # type: ignore
            beta=_get_parameter(params, f"{beta_name}_{i}"),  # type: ignore
            mmin=_get_parameter(params, f"{mmin_name}_{i}"),  # type: ignore
            mmax=_get_parameter(params, f"{mmax_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        transformed_powerlaw = ExtendedSupportTransformedDistribution(
            base_distribution=powerlaw,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )
        powerlaws_collection.append(transformed_powerlaw)
    return powerlaws_collection


def create_powerlaw_redshift_model(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component power law redshift models.

    See :class:`~gwkokab.models.redshift.PowerlawRedshiftModel`.

    Reads the hyper-parameters ``<parameter_name>_kappa_<component_type>``, ``<parameter_name>_z_max_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the redshift parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One redshift model per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    kappa_name = parameter_name + "_kappa_" + component_type
    z_max_name = parameter_name + "_z_max_" + component_type

    return [
        PowerlawRedshiftModel(
            kappa=_get_parameter(params, f"{kappa_name}_{i}"),  # type: ignore
            z_max=_get_parameter(params, f"{z_max_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_madau_dickinson_redshift_model(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component Madau-Dickinson redshift models.

    See :class:`~gwkokab.models.redshift.MadauDickinsonRedshiftModel`.

    Reads the hyper-parameters ``<parameter_name>_kappa_<component_type>``, ``<parameter_name>_z_max_<component_type>``, ``<parameter_name>_gamma_<component_type>``, ``<parameter_name>_z_peak_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the redshift parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One redshift model per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    kappa_name = parameter_name + "_kappa_" + component_type
    z_max_name = parameter_name + "_z_max_" + component_type
    gamma_name = parameter_name + "_gamma_" + component_type
    z_peak_name = parameter_name + "_z_peak_" + component_type
    return [
        MadauDickinsonRedshiftModel(
            kappa=_get_parameter(params, f"{kappa_name}_{i}"),  # type: ignore
            z_max=_get_parameter(params, f"{z_max_name}_{i}"),  # type: ignore
            gamma=_get_parameter(params, f"{gamma_name}_{i}"),  # type: ignore
            z_peak=_get_parameter(params, f"{z_peak_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_uniform_distributions(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component uniform distributions.

    Useful for a parameter that should contribute no shape information, only support.

    Reads the hyper-parameters ``<parameter_name>_low_<component_type>``, ``<parameter_name>_high_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the physical parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One :class:`~numpyro.distributions.Uniform` per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    low_name = f"{parameter_name}_low_{component_type}"
    high_name = f"{parameter_name}_high_{component_type}"

    return [
        Uniform(
            low=_get_parameter(params, f"{low_name}_{i}"),  # type: ignore
            high=_get_parameter(params, f"{high_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_broken_powerlaws(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component one-dimensional broken power laws.

    See :class:`~gwkokab.models.mass.BrokenPowerlaw`.

    Reads the hyper-parameters ``<parameter_name>_alpha1_<component_type>``, ``<parameter_name>_alpha2_<component_type>``, ``<parameter_name>_break_<component_type>``, ``<parameter_name>_low_<component_type>``, ``<parameter_name>_high_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the physical parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One broken power law per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    alpha1_name = parameter_name + "_alpha1_" + component_type
    alpha2_name = parameter_name + "_alpha2_" + component_type
    mbreak_name = parameter_name + "_break_" + component_type
    mmax_name = parameter_name + "_high_" + component_type
    mmin_name = parameter_name + "_low_" + component_type
    return [
        BrokenPowerlaw(
            alpha1=_get_parameter(params, f"{alpha1_name}_{i}"),  # type: ignore
            alpha2=_get_parameter(params, f"{alpha2_name}_{i}"),  # type: ignore
            mbreak=_get_parameter(params, f"{mbreak_name}_{i}"),  # type: ignore
            mmin=_get_parameter(params, f"{mmin_name}_{i}"),  # type: ignore
            mmax=_get_parameter(params, f"{mmax_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_generic_tilt_model(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[MixtureGeneral]:
    """Build per-component spin tilt models.

    Each component is a :func:`~gwkokab.models.spin.GenericTiltModel`: a mixture of an
    isotropic and a normally distributed tilt, for each of the two spins.

    Reads the hyper-parameters ``cos_tilt_zeta_<component_type>`` together with ``loc``, ``scale``, ``low`` and
    ``high`` for each of ``cos_tilt_1`` and ``cos_tilt_2``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Unused; the hyper-parameter names of this factory are fixed.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[MixtureGeneral]
        One tilt model per component. The bounds default to :math:`[-1, 1]`.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    zeta_name = "cos_tilt_zeta_" + component_type
    loc1_name = P.COS_TILT_1 + "_loc_" + component_type
    loc2_name = P.COS_TILT_2 + "_loc_" + component_type
    scale1_name = P.COS_TILT_1 + "_scale_" + component_type
    scale2_name = P.COS_TILT_2 + "_scale_" + component_type
    low1_name = P.COS_TILT_1 + "_low_" + component_type
    low2_name = P.COS_TILT_2 + "_low_" + component_type
    high1_name = P.COS_TILT_1 + "_high_" + component_type
    high2_name = P.COS_TILT_2 + "_high_" + component_type

    return [
        GenericTiltModel(
            zeta=_get_parameter(params, f"{zeta_name}_{i}"),  # type: ignore
            loc1=_get_parameter(params, f"{loc1_name}_{i}"),  # type: ignore
            loc2=_get_parameter(params, f"{loc2_name}_{i}"),  # type: ignore
            scale1=_get_parameter(params, f"{scale1_name}_{i}"),  # type: ignore
            scale2=_get_parameter(params, f"{scale2_name}_{i}"),  # type: ignore
            low1=_get_parameter(params, f"{low1_name}_{i}", default=-1.0),  # type: ignore
            low2=_get_parameter(params, f"{low2_name}_{i}", default=-1.0),  # type: ignore
            high1=_get_parameter(params, f"{high1_name}_{i}", default=1.0),  # type: ignore
            high2=_get_parameter(params, f"{high2_name}_{i}", default=1.0),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_powerlaws(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    r"""Build per-component truncated power laws with a sign-flipped index.

    The index is negated on the way in, so a positive ``alpha`` hyper-parameter means a
    *falling* power law :math:`x^{-\alpha}`, the convention used for mass spectra.

    Reads the hyper-parameters ``<parameter_name>_alpha_<component_type>``, ``<parameter_name>_low_<component_type>``, ``<parameter_name>_high_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the physical parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One :class:`~gwkokab.models.utils.DoublyTruncatedPowerLaw` per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    alpha_name = parameter_name + "_alpha_" + component_type
    mmax_name = parameter_name + "_high_" + component_type
    mmin_name = parameter_name + "_low_" + component_type

    return [
        DoublyTruncatedPowerLaw(
            alpha=-_get_parameter(params, f"{alpha_name}_{i}"),  # type: ignore
            mmin=_get_parameter(params, f"{mmin_name}_{i}"),  # type: ignore
            mmax=_get_parameter(params, f"{mmax_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_generic_powerlaws(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    r"""Build per-component truncated power laws.

    As :func:`create_powerlaws`, but the index is taken at face value, so a positive
    ``alpha`` means a *rising* power law :math:`x^{\alpha}`.

    Reads the hyper-parameters ``<parameter_name>_alpha_<component_type>``, ``<parameter_name>_low_<component_type>``, ``<parameter_name>_high_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the physical parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One :class:`~gwkokab.models.utils.DoublyTruncatedPowerLaw` per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    alpha_name = parameter_name + "_alpha_" + component_type
    high_name = parameter_name + "_high_" + component_type
    low_name = parameter_name + "_low_" + component_type
    return [
        DoublyTruncatedPowerLaw(
            alpha=_get_parameter(params, f"{alpha_name}_{i}"),  # type: ignore
            low=_get_parameter(params, f"{low_name}_{i}"),  # type: ignore
            high=_get_parameter(params, f"{high_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_two_truncated_normal_mixture(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[MixtureGeneral]:
    """Build per-component mixtures of two truncated normals.

    See :func:`~gwkokab.models.sundry.TwoTruncatedNormalMixture`. The truncation bounds
    are optional on both sub-components.

    Reads the hyper-parameters ``<parameter_name>_zeta_<component_type>`` together with ``loc``, ``scale``,
    ``low`` and ``high`` for each of ``comp1`` and ``comp2``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the physical parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[MixtureGeneral]
        One two-component mixture per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    comp1_high_name = parameter_name + "_comp1_high_" + component_type
    comp2_high_name = parameter_name + "_comp2_high_" + component_type
    comp1_loc_name = parameter_name + "_comp1_loc_" + component_type
    comp2_loc_name = parameter_name + "_comp2_loc_" + component_type
    comp1_low_name = parameter_name + "_comp1_low_" + component_type
    comp2_low_name = parameter_name + "_comp2_low_" + component_type
    comp1_scale_name = parameter_name + "_comp1_scale_" + component_type
    comp2_scale_name = parameter_name + "_comp2_scale_" + component_type
    zeta_name = parameter_name + "_zeta_" + component_type

    # fmt: off
    return [
        TwoTruncatedNormalMixture(
            comp1_high=_get_parameter(params, f"{comp1_high_name}_{i}", is_necessary=False),  # type: ignore
            comp2_high=_get_parameter(params, f"{comp2_high_name}_{i}", is_necessary=False),  # type: ignore
            comp1_loc=_get_parameter(params, f"{comp1_loc_name}_{i}"),  # type: ignore
            comp2_loc=_get_parameter(params, f"{comp2_loc_name}_{i}"),  # type: ignore
            comp1_low=_get_parameter(params, f"{comp1_low_name}_{i}", is_necessary=False),  # type: ignore
            comp2_low=_get_parameter(params, f"{comp2_low_name}_{i}", is_necessary=False),  # type: ignore
            comp1_scale=_get_parameter(params, f"{comp1_scale_name}_{i}"),  # type: ignore
            comp2_scale=_get_parameter(params, f"{comp2_scale_name}_{i}"),  # type: ignore
            zeta=_get_parameter(params, f"{zeta_name}_{i}", zeta_name),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]
    # fmt: on


def create_spin_magnitude_mixture_models(
    N: int,
    parameter_name,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
):
    # fmt: off
    """Build per-component joint spin magnitude models for both spins.

    The primary and secondary spin magnitudes are stacked into one two-dimensional
    :func:`~gwkokab.models.sundry.NDTwoTruncatedNormalMixture`, so the two magnitudes
    share a single mixing fraction and are modelled jointly rather than independently.

    Reads the hyper-parameters ``a_zeta_<component_type>`` together with ``loc``, ``scale``, ``low`` and ``high``
    for each of ``comp1`` and ``comp2`` of both ``a_1`` and ``a_2``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Unused; the hyper-parameter names of this factory are fixed.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One two-dimensional spin magnitude model per component. Bounds default to
        :math:`[0, 1]`.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    zeta_name = "a_zeta_" + component_type
    a_1_comp1_high_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_" + component_type
    a_1_comp1_loc_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_" + component_type
    a_1_comp1_low_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_" + component_type
    a_1_comp1_scale_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_" + component_type
    a_1_comp2_high_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_" + component_type
    a_1_comp2_loc_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_" + component_type
    a_1_comp2_low_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_" + component_type
    a_1_comp2_scale_name = P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_" + component_type
    a_2_comp1_high_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_" + component_type
    a_2_comp1_loc_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_" + component_type
    a_2_comp1_low_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_" + component_type
    a_2_comp1_scale_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_" + component_type
    a_2_comp2_high_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_" + component_type
    a_2_comp2_loc_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_" + component_type
    a_2_comp2_low_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_" + component_type
    a_2_comp2_scale_name = P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_" + component_type
    # fmt: on

    spin_collection = []

    for i in range(N):
        # fmt: off
        zeta = _get_parameter(params, f"{zeta_name}_{i}", zeta_name)
        a_1_comp1_high: ArrayLike = _get_parameter(params, f"{a_1_comp1_high_name}_{i}", default=1.0) # type: ignore
        a_1_comp1_loc: ArrayLike = _get_parameter(params, f"{a_1_comp1_loc_name}_{i}") # type: ignore
        a_1_comp1_low: ArrayLike = _get_parameter(params, f"{a_1_comp1_low_name}_{i}", default=0.0) # type: ignore
        a_1_comp1_scale: ArrayLike = _get_parameter(params, f"{a_1_comp1_scale_name}_{i}") # type: ignore
        a_1_comp2_high: ArrayLike = _get_parameter(params, f"{a_1_comp2_high_name}_{i}", default=1.0) # type: ignore
        a_1_comp2_loc: ArrayLike = _get_parameter(params, f"{a_1_comp2_loc_name}_{i}") # type: ignore
        a_1_comp2_low: ArrayLike = _get_parameter(params, f"{a_1_comp2_low_name}_{i}", default=0.0) # type: ignore
        a_1_comp2_scale: ArrayLike = _get_parameter(params, f"{a_1_comp2_scale_name}_{i}") # type: ignore
        a_2_comp1_high: ArrayLike = _get_parameter(params, f"{a_2_comp1_high_name}_{i}", default=1.0) # type: ignore
        a_2_comp1_loc: ArrayLike = _get_parameter(params, f"{a_2_comp1_loc_name}_{i}") # type: ignore
        a_2_comp1_low: ArrayLike = _get_parameter(params, f"{a_2_comp1_low_name}_{i}", default=0.0) # type: ignore
        a_2_comp1_scale: ArrayLike = _get_parameter(params, f"{a_2_comp1_scale_name}_{i}") # type: ignore
        a_2_comp2_high: ArrayLike = _get_parameter(params, f"{a_2_comp2_high_name}_{i}", default=1.0) # type: ignore
        a_2_comp2_loc: ArrayLike = _get_parameter(params, f"{a_2_comp2_loc_name}_{i}") # type: ignore
        a_2_comp2_low: ArrayLike = _get_parameter(params, f"{a_2_comp2_low_name}_{i}", default=0.0) # type: ignore
        a_2_comp2_scale: ArrayLike = _get_parameter(params, f"{a_2_comp2_scale_name}_{i}") # type: ignore
        # fmt: on

        comp1_high = jnp.stack((a_1_comp1_high, a_2_comp1_high), axis=-1)
        comp1_loc = jnp.stack((a_1_comp1_loc, a_2_comp1_loc), axis=-1)
        comp1_low = jnp.stack((a_1_comp1_low, a_2_comp1_low), axis=-1)
        comp1_scale = jnp.stack((a_1_comp1_scale, a_2_comp1_scale), axis=-1)
        comp2_high = jnp.stack((a_1_comp2_high, a_2_comp2_high), axis=-1)
        comp2_loc = jnp.stack((a_1_comp2_loc, a_2_comp2_loc), axis=-1)
        comp2_low = jnp.stack((a_1_comp2_low, a_2_comp2_low), axis=-1)
        comp2_scale = jnp.stack((a_1_comp2_scale, a_2_comp2_scale), axis=-1)

        spin_dist = NDTwoTruncatedNormalMixture(
            zeta=zeta,
            comp1_high=comp1_high,
            comp1_loc=comp1_loc,
            comp1_low=comp1_low,
            comp1_scale=comp1_scale,
            comp2_high=comp2_high,
            comp2_loc=comp2_loc,
            comp2_low=comp2_low,
            comp2_scale=comp2_scale,
            validate_args=validate_args,
        )

        spin_collection.append(spin_dist)

    return spin_collection


def create_gwtc4_effective_spin_skew_normal_models(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component GWTC-4 effective spin skew normal models.

    See :class:`~gwkokab.models.spin.GWTC4EffectiveSpinSkewNormalModel`.

    Reads the hyper-parameters ``<parameter_name>_loc_<component_type>``, ``<parameter_name>_scale_<component_type>``, ``<parameter_name>_epsilon_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Name of the physical parameter, used as the prefix of the hyper-parameter
        names.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One effective spin model per component.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    loc_name = parameter_name + "_loc_" + component_type
    scale_name = parameter_name + "_scale_" + component_type
    epsilon_name = parameter_name + "_epsilon_" + component_type

    return [
        GWTC4EffectiveSpinSkewNormalModel(
            loc=_get_parameter(params, f"{loc_name}_{i}"),  # type: ignore
            scale=_get_parameter(params, f"{scale_name}_{i}"),  # type: ignore
            epsilon=_get_parameter(params, f"{epsilon_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]


def create_smoothed_broken_powerlaws_mass_ratio_powerlaw(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component smoothed broken power law mass models.

    See :class:`~gwkokab.models.mass.SmoothedBrokenPowerlawMassRatioPowerlaw`. Each component is built in :math:`(m_1, q)` coordinates and pushed to
    :math:`(m_1, m_2)` by
    :class:`~gwkokab.models.transformations.PrimaryMassAndMassRatioToComponentMassesTransform`
    while keeping its mass sandwich support.

    Reads the hyper-parameters ``m1_alpha1``, ``m1_alpha2``, ``beta``, ``m1_delta``, ``m2_delta``, ``m1_low``,
    ``m2_low``, ``m1_break`` and ``m1_high``, each tagged with ``_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Unused; the hyper-parameter names of this factory are fixed.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One transformed mass model per component, over :math:`(m_1, m_2)`.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    collection = []

    alpha1_name = "m1_alpha1_" + component_type
    alpha2_name = "m1_alpha2_" + component_type
    beta_name = "beta_" + component_type
    delta_m1_name = "m1_delta_" + component_type
    delta_m2_name = "m2_delta_" + component_type
    m1min_name = "m1_low_" + component_type
    m2min_name = "m2_low_" + component_type
    mbreak_name = "m1_break_" + component_type
    mmax_name = "m1_high_" + component_type

    for i in range(N):
        suffix = f"_{i}"
        broken_powerlaw = SmoothedBrokenPowerlawMassRatioPowerlaw(
            alpha1=_get_parameter(params, alpha1_name + suffix),  # type: ignore
            alpha2=_get_parameter(params, alpha2_name + suffix),  # type: ignore
            beta=_get_parameter(params, beta_name + suffix),  # type: ignore
            delta_m1=_get_parameter(params, delta_m1_name + suffix),  # type: ignore
            delta_m2=_get_parameter(params, delta_m2_name + suffix),  # type: ignore
            m1min=_get_parameter(params, m1min_name + suffix),  # type: ignore
            m2min=_get_parameter(params, m2min_name + suffix),  # type: ignore
            mbreak=_get_parameter(params, mbreak_name + suffix),  # type: ignore
            mmax=_get_parameter(params, mmax_name + suffix),  # type: ignore
            validate_args=validate_args,
        )
        distribution = ExtendedSupportTransformedDistribution(
            base_distribution=broken_powerlaw,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )

        collection.append(distribution)
    return collection


def create_smoothed_gaussian_primary_mass_ratio(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component smoothed Gaussian mass models.

    See :class:`~gwkokab.models.mass.SmoothedGaussianPrimaryMassRatio`. Each component is built in :math:`(m_1, q)` coordinates and pushed to
    :math:`(m_1, m_2)` by
    :class:`~gwkokab.models.transformations.PrimaryMassAndMassRatioToComponentMassesTransform`
    while keeping its mass sandwich support.

    Reads the hyper-parameters ``loc``, ``scale``, ``beta``, ``m1min``, ``m2min``, ``mmax``, ``delta_m1`` and
    ``delta_m2``, each tagged with ``_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Unused; the hyper-parameter names of this factory are fixed.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One transformed mass model per component, over :math:`(m_1, m_2)`.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    collection = []

    loc_name = "loc_" + component_type
    scale_name = "scale_" + component_type
    beta_name = "beta_" + component_type
    m1min_name = "m1min_" + component_type
    m2min_name = "m2min_" + component_type
    mmax_name = "mmax_" + component_type
    delta_m1_name = "delta_m1_" + component_type
    delta_m2_name = "delta_m2_" + component_type

    for i in range(N):
        suffix = f"_{i}"

        smoothed_gaussian = SmoothedGaussianPrimaryMassRatio(
            loc=_get_parameter(params, loc_name + suffix),  # type: ignore
            scale=_get_parameter(params, scale_name + suffix),  # type: ignore
            beta=_get_parameter(params, beta_name + suffix),  # type: ignore
            m1min=_get_parameter(params, m1min_name + suffix),  # type: ignore
            m2min=_get_parameter(params, m2min_name + suffix),  # type: ignore
            mmax=_get_parameter(params, mmax_name + suffix),  # type: ignore
            delta_m1=_get_parameter(params, delta_m1_name + suffix),  # type: ignore
            delta_m2=_get_parameter(params, delta_m2_name + suffix),  # type: ignore
            validate_args=validate_args,
        )

        distribution = ExtendedSupportTransformedDistribution(
            base_distribution=smoothed_gaussian,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )

        collection.append(distribution)
    return collection


def create_gaussian_primary_mass_ratio(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component Gaussian mass models.

    See :class:`~gwkokab.models.mass.GaussianPrimaryMassRatio`. Each component is built in :math:`(m_1, q)` coordinates and pushed to
    :math:`(m_1, m_2)` by
    :class:`~gwkokab.models.transformations.PrimaryMassAndMassRatioToComponentMassesTransform`
    while keeping its mass sandwich support.

    Reads the hyper-parameters ``m1_loc``, ``m1_scale``, ``beta``, ``m1_low`` and ``m1_high``, each tagged with
    ``_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Unused; the hyper-parameter names of this factory are fixed.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One transformed mass model per component, over :math:`(m_1, m_2)`.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    collection = []

    loc_name = "m1_loc_" + component_type
    scale_name = "m1_scale_" + component_type
    beta_name = "beta_" + component_type
    mmin_name = "m1_low_" + component_type
    mmax_name = "m1_high_" + component_type

    for i in range(N):
        suffix = f"_{i}"
        gaussian = GaussianPrimaryMassRatio(
            loc=_get_parameter(params, loc_name + suffix),  # type: ignore
            scale=_get_parameter(params, scale_name + suffix),  # type: ignore
            beta=_get_parameter(params, beta_name + suffix),  # type: ignore
            mmin=_get_parameter(params, mmin_name + suffix),  # type: ignore
            mmax=_get_parameter(params, mmax_name + suffix),  # type: ignore
            validate_args=validate_args,
        )

        distribution = ExtendedSupportTransformedDistribution(
            base_distribution=gaussian,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )

        collection.append(distribution)
    return collection


def create_smoothed_powerlaw_primary_mass_ratio(
    N: int,
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component smoothed power law mass models.

    See :class:`~gwkokab.models.mass.SmoothedPowerlawPrimaryMassRatio`. Each component is built in :math:`(m_1, q)` coordinates and pushed to
    :math:`(m_1, m_2)` by
    :class:`~gwkokab.models.transformations.PrimaryMassAndMassRatioToComponentMassesTransform`
    while keeping its mass sandwich support.

    Reads the hyper-parameters ``m1_alpha``, ``beta``, ``m1_delta``, ``m2_delta``, ``m1_low``, ``m2_low`` and
    ``m1_high``, each tagged with ``_<component_type>``, each suffixed with ``_<index>``.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    parameter_name : str
        Unused; the hyper-parameter names of this factory are fixed.
    component_type : str
        Component tag naming the family of components, e.g. ``"pl"``, ``"g"``,
        ``"spl"``, ``"bpl"``, ``"gpl"`` or ``"gg"``.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values, keyed by
        ``<role>_<component_type>_<index>``.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One transformed mass model per component, over :math:`(m_1, m_2)`.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    collection = []

    alpha_name = "m1_alpha_" + component_type
    beta_name = "beta_" + component_type
    delta_m1_name = "m1_delta_" + component_type
    delta_m2_name = "m2_delta_" + component_type
    m1min_name = "m1_low_" + component_type
    m2min_name = "m2_low_" + component_type
    mmax_name = "m1_high_" + component_type

    for i in range(N):
        suffix = f"_{i}"
        broken_powerlaw = SmoothedPowerlawPrimaryMassRatio(
            alpha=_get_parameter(params, alpha_name + suffix),  # type: ignore
            beta=_get_parameter(params, beta_name + suffix),  # type: ignore
            delta_m1=_get_parameter(params, delta_m1_name + suffix),  # type: ignore
            delta_m2=_get_parameter(params, delta_m2_name + suffix),  # type: ignore
            m1min=_get_parameter(params, m1min_name + suffix),  # type: ignore
            m2min=_get_parameter(params, m2min_name + suffix),  # type: ignore
            mmax=_get_parameter(params, mmax_name + suffix),  # type: ignore
            validate_args=validate_args,
        )

        distribution = ExtendedSupportTransformedDistribution(
            base_distribution=broken_powerlaw,
            transforms=PrimaryMassAndMassRatioToComponentMassesTransform(),
            validate_args=validate_args,
        )

        collection.append(distribution)
    return collection


def create_generic_smoothed_powerlaw_mass_ratio(
    N: int,
    primary_mass_distributions: List[Distribution],
    parameter_name: str,
    component_type: str,
    params: Dict[str, Array],
    validate_args: Optional[bool] = None,
) -> List[Distribution]:
    """Build per-component mass models over supplied primary mass distributions.

    Wraps each given primary mass distribution in a
    :class:`~gwkokab.models.mass.GenericSmoothedPowerlawMassRatio`, which adds the
    Planck-tapered conditional mass ratio power law. Unlike the other mass factories,
    the primary mass model is supplied rather than built, and the results are left in
    :math:`(m_1, q)` coordinates.

    Reads the hyper-parameters ``beta_<component_type>``, ``m2_delta_<component_type>``
    and ``m2_low_<component_type>``, each suffixed with ``_<index>``, plus a single
    ``m1_delta`` shared by every component.

    Parameters
    ----------
    N : int
        Number of mixture components to build.
    primary_mass_distributions : List[Distribution]
        One primary mass distribution per component. Each must have an interval
        support, which supplies the mass bounds.
    parameter_name : str
        Unused; the hyper-parameter names of this factory are fixed.
    component_type : str
        Component tag naming the family of components.
    params : Dict[str, Array]
        Flat dictionary of hyper-parameter values.
    validate_args : Optional[bool], optional
        Whether to validate distribution parameters and inputs. Defaults to
        :data:`None`.

    Returns
    -------
    List[Distribution]
        One mass model per component, over :math:`(m_1, q)`.

    Raises
    ------
    ValueError
        If a required hyper-parameter is missing from ``params``.
    """
    beta_name = "beta_" + component_type
    delta_m1_name = "m1_delta"
    delta_m2_name = "m2_delta_" + component_type
    m2min_name = "m2_low_" + component_type

    delta_m1 = _get_parameter(params, delta_m1_name)

    return [
        GenericSmoothedPowerlawMassRatio(
            primary_mass_distribution=primary_mass_distributions[i],
            delta_m1=delta_m1,  # type: ignore
            beta=_get_parameter(params, f"{beta_name}_{i}"),  # type: ignore
            delta_m2=_get_parameter(params, f"{delta_m2_name}_{i}"),  # type: ignore
            m2min=_get_parameter(params, f"{m2min_name}_{i}"),  # type: ignore
            validate_args=validate_args,
        )
        for i in range(N)
    ]
