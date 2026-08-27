# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared definitions for the subpopulation analysis family.

SubPopulationModelCore is the family half of the analysis matrix: it names the event
coordinates the data must supply (:attr:`~SubPopulationModelCore.parameters`) and the flat
list of population hyper-parameters to be inferred
(:attr:`~SubPopulationModelCore.model_parameters`). Both are derived from the ``use_*``
flags, so switching a physical parameter on at the command line extends the
parameter list automatically.

The model is :func:`~gwkokab.models.hybrids.SubPopulationModel`, built from ``N_spl`` power-law, ``N_bpl`` broken power-law and ``N_gpl`` truncated
normal mass sub-populations. Hyper-parameter names carry the
component tags ``spl``, ``bpl`` and ``gpl``, giving forms like ``alpha_pl_0``.

This module is mixed with a data-representation base and a sampler mixin to form a
complete analysis; see the ``discrete`` and ``analytical_gwalk`` modules alongside
it.
"""

from argparse import ArgumentParser
from typing import Callable, List, Optional

from jax import numpy as jnp
from jaxtyping import Array

from gwkokab.analysis.utils.checks import check_min_concentration_for_beta_dist
from gwkokab.analysis.utils.common import expand_arguments
from gwkokab.models import SubPopulationModel
from gwkokab.parameters import Parameters as P


def where_fns_list(
    use_beta_spin_magnitude: bool,
) -> Optional[List[Callable[..., Array]]]:
    r"""Build the extra validity predicates this configuration needs.

    These are applied on top of the prior support: a point failing any of them is
    assigned :math:`-\infty` without the model ever being built.

    Parameters
    ----------
    use_beta_spin_magnitude : bool
        Whether spin magnitudes are modelled with beta distributions, which need
        their mean/variance pairs checked for validity.

    Returns
    -------
    Optional[List[Callable[..., Array]]]
        The predicates, or :data:`None` when none are needed.
    """
    where_fns = []

    if use_beta_spin_magnitude:

        def positive_concentration(**kwargs) -> Array:
            """Check that every beta spin prior has positive concentrations.

            Not every mean/variance pair corresponds to a valid beta distribution; this
            screens the pairs over every sub-population family.

            Parameters
            ----------
            **kwargs : Any
                The component counts and the sampled hyper-parameters.

            Returns
            -------
            Array
                Boolean mask, true where every spin magnitude prior is valid.
            """
            N_spl: int = kwargs.get("N_spl")  # type: ignore
            N_bpl: int = kwargs.get("N_bpl")  # type: ignore
            N_gpl: int = kwargs.get("N_gpl")  # type: ignore

            mask = jnp.ones((), dtype=bool)

            for ctype, n in zip(("spl", "bpl", "gpl"), (N_spl, N_bpl, N_gpl)):
                for n_c in range(n):
                    chi_mean: Array = kwargs.get(
                        P.PRIMARY_SPIN_MAGNITUDE + f"_mean_{ctype}_{n_c}"
                    )  # type: ignore
                    chi_variance: Array = kwargs.get(
                        P.PRIMARY_SPIN_MAGNITUDE + f"_variance_{ctype}_{n_c}"
                    )  # type: ignore
                    mask &= check_min_concentration_for_beta_dist(
                        chi_mean, chi_variance
                    )
            return mask

        where_fns.append(positive_concentration)

    return where_fns if len(where_fns) > 0 else None


class SubPopulationModelCore:
    """Family half of the subpopulation analysis.

    Mixed with a data-representation base -- :class:`~gwkokab.analysis.core.discrete_base.DiscreteBase`
    or :class:`~gwkokab.analysis.core.analytical_gwalk_base.AnalyticalGWalkBase` -- and
    a sampler mixin, to form a complete analysis.

    See :meth:`__init__` for the constructor arguments.
    """

    model_fn = SubPopulationModel
    """The population model factory,
    :func:`~gwkokab.models.hybrids.SubPopulationModel`.
    """

    def __init__(
        self,
        N_spl: int,
        N_bpl: int,
        N_gpl: int,
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
    ) -> None:
        """Record which components and physical parameters this analysis uses.

        The ``use_*`` flags decide both what the model contains and, through
        :attr:`parameters` and :attr:`model_parameters`, what the data must supply and
        what will be inferred.

        Parameters
        ----------
        N_spl : int
            Number of power-law sub-populations.
        N_bpl : int
            Number of broken power-law sub-populations.
        N_gpl : int
            Number of truncated normal sub-populations.
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
        """
        self.N_spl = N_spl
        self.N_bpl = N_bpl
        self.N_gpl = N_gpl
        self.use_beta_spin_magnitude = use_beta_spin_magnitude
        self.use_spin_magnitude_mixture = use_spin_magnitude_mixture
        self.use_truncated_normal_spin_x = use_truncated_normal_spin_x
        self.use_truncated_normal_spin_y = use_truncated_normal_spin_y
        self.use_truncated_normal_spin_z = use_truncated_normal_spin_z
        self.use_chi_eff_mixture = use_chi_eff_mixture
        self.use_skew_normal_chi_eff = use_skew_normal_chi_eff
        self.use_truncated_normal_chi_p = use_truncated_normal_chi_p
        self.use_eccentricity_powerlaw = use_eccentricity_powerlaw
        self.use_tilt = use_tilt
        self.use_eccentricity_mixture = use_eccentricity_mixture
        self.use_mean_anomaly = use_mean_anomaly
        self.use_powerlaw_redshift = use_powerlaw_redshift
        self.use_madau_dickinson_redshift = use_madau_dickinson_redshift

    def modify_model_params(self, params: dict) -> dict:
        """Inject the component counts and ``use_*`` flags into the model parameters.

        The prior configuration supplies only the hyper-parameters to be inferred; the
        model factory also needs to know how many components of each kind to build and
        which physical parameters were switched on. This hook adds them.

        Parameters
        ----------
        params : dict
            The processed priors, keyed by hyper-parameter name.

        Returns
        -------
        dict
            ``params``, with the counts and flags added.
        """
        params.update({
            "N_spl": self.N_spl,
            "N_bpl": self.N_bpl,
            "N_gpl": self.N_gpl,
            "use_beta_spin_magnitude": self.use_beta_spin_magnitude,
            "use_spin_magnitude_mixture": self.use_spin_magnitude_mixture,
            "use_truncated_normal_spin_x": self.use_truncated_normal_spin_x,
            "use_truncated_normal_spin_y": self.use_truncated_normal_spin_y,
            "use_truncated_normal_spin_z": self.use_truncated_normal_spin_z,
            "use_chi_eff_mixture": self.use_chi_eff_mixture,
            "use_skew_normal_chi_eff": self.use_skew_normal_chi_eff,
            "use_truncated_normal_chi_p": self.use_truncated_normal_chi_p,
            "use_tilt": self.use_tilt,
            "use_eccentricity_mixture": self.use_eccentricity_mixture,
            "use_eccentricity_powerlaw": self.use_eccentricity_powerlaw,
            "use_mean_anomaly": self.use_mean_anomaly,
            "use_powerlaw_redshift": self.use_powerlaw_redshift,
            "use_madau_dickinson_redshift": self.use_madau_dickinson_redshift,
        })
        return params

    @property
    def parameters(self) -> tuple[str, ...]:
        """The event coordinates this analysis reads from the data.

        Always the two component masses, plus whichever spin, tilt, eccentricity,
        redshift and extrinsic coordinates the ``use_*`` flags switched on.

        Returns
        -------
        tuple[str, ...]
            Parameter names, in the order the model's event axis expects them.
        """
        names = [P.PRIMARY_MASS_SOURCE, P.MASS_RATIO]
        if self.use_beta_spin_magnitude or self.use_spin_magnitude_mixture:
            names.append(P.PRIMARY_SPIN_MAGNITUDE)
            names.append(P.SECONDARY_SPIN_MAGNITUDE)
        if self.use_truncated_normal_spin_x:
            names.append(P.PRIMARY_SPIN_X)
            names.append(P.SECONDARY_SPIN_X)
        if self.use_truncated_normal_spin_y:
            names.append(P.PRIMARY_SPIN_Y)
            names.append(P.SECONDARY_SPIN_Y)
        if self.use_truncated_normal_spin_z:
            names.append(P.PRIMARY_SPIN_Z)
            names.append(P.SECONDARY_SPIN_Z)
        if self.use_chi_eff_mixture or self.use_skew_normal_chi_eff:
            names.append(P.EFFECTIVE_SPIN)
        if self.use_truncated_normal_chi_p:
            names.append(P.PRECESSING_SPIN)
        if self.use_tilt:
            names.extend([P.COS_TILT_1, P.COS_TILT_2])
        if self.use_eccentricity_mixture or self.use_eccentricity_powerlaw:
            names.append(P.ECCENTRICITY)
        if self.use_mean_anomaly:
            names.append(P.MEAN_ANOMALY)
        if self.use_powerlaw_redshift or self.use_madau_dickinson_redshift:
            names.append(P.REDSHIFT)
        return names

    @property
    def model_parameters(self) -> list[str]:
        # fmt: off
        """The flat list of population hyper-parameters to be inferred.

        Each entry is a per-component name of the form ``<role>_<component tag>_<index>``,
        expanded from the enabled ``use_*`` flags by
        :func:`~gwkokab.analysis.utils.common.expand_arguments`. These names are what the
        regex keys of ``prior_cfg.json`` are matched against.

        Returns
        -------
        list[str]
            Hyper-parameter names.
        """
        component_types_and_count = zip(
            ("spl"     , "bpl"     , "gpl"     ),
            (self.N_spl, self.N_bpl, self.N_gpl),
        )
        # fmt: on

        all_params: list[tuple[str, int]] = [
            ("lambda", self.N_spl + self.N_bpl + self.N_gpl - 1)
        ]

        for ct, count in component_types_and_count:
            all_params_names = [
                "beta_",
                "m2_delta_",
                "m2_low_",
            ]

            if ct == "spl":
                all_params_names.extend([
                    "m1_alpha_",
                    "m1_low_",
                    "m1_high_",
                ])

            if ct == "bpl":
                all_params_names.extend([
                    "m1_alpha1_",
                    "m1_alpha2_",
                    "m1_break_",
                    "m1_high_",
                    "m1_low_",
                ])

            if ct == "gpl":
                all_params_names.extend([
                    "m1_high_",
                    "m1_loc_",
                    "m1_low_",
                    "m1_scale_",
                ])

            if self.use_spin_magnitude_mixture:
                all_params_names.extend([
                    "a_zeta_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_",
                ])

            if self.use_beta_spin_magnitude:
                all_params_names.extend([
                    P.PRIMARY_SPIN_MAGNITUDE + "_mean_",
                    P.PRIMARY_SPIN_MAGNITUDE + "_variance_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_mean_",
                    P.SECONDARY_SPIN_MAGNITUDE + "_variance_",
                ])

            if self.use_truncated_normal_spin_x:
                all_params_names.extend([
                    P.PRIMARY_SPIN_X + "_high_",
                    P.PRIMARY_SPIN_X + "_loc_",
                    P.PRIMARY_SPIN_X + "_low_",
                    P.PRIMARY_SPIN_X + "_scale_",
                    P.SECONDARY_SPIN_X + "_high_",
                    P.SECONDARY_SPIN_X + "_loc_",
                    P.SECONDARY_SPIN_X + "_low_",
                    P.SECONDARY_SPIN_X + "_scale_",
                ])

            if self.use_truncated_normal_spin_y:
                all_params_names.extend([
                    P.PRIMARY_SPIN_Y + "_high_",
                    P.PRIMARY_SPIN_Y + "_loc_",
                    P.PRIMARY_SPIN_Y + "_low_",
                    P.PRIMARY_SPIN_Y + "_scale_",
                    P.SECONDARY_SPIN_Y + "_high_",
                    P.SECONDARY_SPIN_Y + "_loc_",
                    P.SECONDARY_SPIN_Y + "_low_",
                    P.SECONDARY_SPIN_Y + "_scale_",
                ])

            if self.use_truncated_normal_spin_z:
                all_params_names.extend([
                    P.PRIMARY_SPIN_Z + "_high_",
                    P.PRIMARY_SPIN_Z + "_loc_",
                    P.PRIMARY_SPIN_Z + "_low_",
                    P.PRIMARY_SPIN_Z + "_scale_",
                    P.SECONDARY_SPIN_Z + "_high_",
                    P.SECONDARY_SPIN_Z + "_loc_",
                    P.SECONDARY_SPIN_Z + "_low_",
                    P.SECONDARY_SPIN_Z + "_scale_",
                ])

            if self.use_chi_eff_mixture:
                all_params_names.extend([
                    P.EFFECTIVE_SPIN + "_comp1_high_",
                    P.EFFECTIVE_SPIN + "_comp1_loc_",
                    P.EFFECTIVE_SPIN + "_comp1_low_",
                    P.EFFECTIVE_SPIN + "_comp1_scale_",
                    P.EFFECTIVE_SPIN + "_comp2_high_",
                    P.EFFECTIVE_SPIN + "_comp2_loc_",
                    P.EFFECTIVE_SPIN + "_comp2_low_",
                    P.EFFECTIVE_SPIN + "_comp2_scale_",
                    P.EFFECTIVE_SPIN + "_zeta_",
                ])

            if self.use_skew_normal_chi_eff:
                all_params_names.extend([
                    P.EFFECTIVE_SPIN + "_epsilon_",
                    P.EFFECTIVE_SPIN + "_loc_",
                    P.EFFECTIVE_SPIN + "_scale_",
                ])

            if self.use_truncated_normal_chi_p:
                all_params_names.extend([
                    P.PRECESSING_SPIN + "_high_",
                    P.PRECESSING_SPIN + "_loc_",
                    P.PRECESSING_SPIN + "_low_",
                    P.PRECESSING_SPIN + "_scale_",
                ])

            if self.use_tilt:
                all_params_names.extend([
                    "cos_tilt_zeta_",
                    P.COS_TILT_1 + "_high_",
                    P.COS_TILT_1 + "_loc_",
                    P.COS_TILT_1 + "_low_",
                    P.COS_TILT_1 + "_scale_",
                    P.COS_TILT_2 + "_high_",
                    P.COS_TILT_2 + "_loc_",
                    P.COS_TILT_2 + "_low_",
                    P.COS_TILT_2 + "_scale_",
                ])

            if self.use_eccentricity_mixture:
                all_params_names.extend([
                    P.ECCENTRICITY + "_comp1_high_",
                    P.ECCENTRICITY + "_comp1_loc_",
                    P.ECCENTRICITY + "_comp1_low_",
                    P.ECCENTRICITY + "_comp1_scale_",
                    P.ECCENTRICITY + "_comp2_high_",
                    P.ECCENTRICITY + "_comp2_loc_",
                    P.ECCENTRICITY + "_comp2_low_",
                    P.ECCENTRICITY + "_comp2_scale_",
                    P.ECCENTRICITY + "_zeta_",
                ])

            if self.use_eccentricity_powerlaw:
                all_params_names.extend([
                    P.ECCENTRICITY + "_alpha_",
                    P.ECCENTRICITY + "_high_",
                    P.ECCENTRICITY + "_low_",
                ])

            if self.use_mean_anomaly:
                all_params_names.extend([
                    P.MEAN_ANOMALY + "_high_",
                    P.MEAN_ANOMALY + "_low_",
                ])

            if self.use_powerlaw_redshift:
                all_params_names.extend([
                    P.REDSHIFT + "_kappa_",
                    P.REDSHIFT + "_z_max_",
                ])

            if self.use_madau_dickinson_redshift:
                all_params_names.extend([
                    P.REDSHIFT + "_gamma_",
                    P.REDSHIFT + "_kappa_",
                    P.REDSHIFT + "_z_max_",
                    P.REDSHIFT + "_z_peak_",
                ])

            all_params.extend([(name + ct, count) for name in all_params_names])

        extended_params = [
            "m1_delta",
            "log_rate",
            "m1max",
            "m1min",
        ]
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with this family's model arguments.

    Adds the component counts and one ``--add-*`` flag per optional physical
    parameter.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to.

    Returns
    -------
    ArgumentParser
        The same parser, with the arguments added.
    """
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--n-spl",
        type=int,
        default=0,
        help="Number of smoothed power law for primary mass.",
    )
    model_group.add_argument(
        "--n-bpl",
        type=int,
        default=0,
        help="Number of smoothed broken power law.",
    )
    model_group.add_argument(
        "--n-gpl",
        type=int,
        default=0,
        help="Number of Gaussian components for primary mass.",
    )

    spin_group = model_group.add_mutually_exclusive_group()
    spin_group.add_argument(
        "--add-beta-spin-magnitude",
        action="store_true",
        help=(
            "Model the dimensionless spin magnitudes of both component compact objects "
            "independently using Beta distributions. Parameters are parameterized via the "
            "mean and variance: '{PARAM}_mean_{TYPE}_{i}' and '{PARAM}_variance_{TYPE}_{i}', "
            "naturally bounded between 0 and 1."
        ),
    )
    spin_group.add_argument(
        "--add-spin-magnitude-mixture",
        action="store_true",
        help=(
            "Model the joint primary and secondary spin magnitudes using an independent "
            "2-component Truncated Normal mixture model. The distribution takes the form "
            "(1-zeta) * comp1 + zeta * comp2, where 'a_zeta_{TYPE}_{i}' governs the mixture "
            "fraction, and each component has its own location, scale, and [0, 1] truncation bounds."
        ),
    )

    model_group.add_argument(
        "--add-truncated-normal-spin-x",
        action="store_true",
        help=(
            "Include independent Cartesian spin x-component parameters modeled via "
            "Truncated Normal distributions. Requires location, scale, and optional bounding "
            "parameters ('_low_' and '_high_') for each mixture/population component."
        ),
    )
    model_group.add_argument(
        "--add-truncated-normal-spin-y",
        action="store_true",
        help=(
            "Include independent Cartesian spin y-component parameters modeled via "
            "Truncated Normal distributions. Requires location, scale, and optional bounding "
            "parameters ('_low_' and '_high_') for each mixture/population component."
        ),
    )
    model_group.add_argument(
        "--add-truncated-normal-spin-z",
        action="store_true",
        help=(
            "Include independent Cartesian spin z-component (aligned spin) parameters modeled "
            "via Truncated Normal distributions. Requires location, scale, and optional bounding "
            "parameters ('_low_' and '_high_') for each mixture/population component."
        ),
    )

    chi_eff_group = model_group.add_mutually_exclusive_group()
    chi_eff_group.add_argument(
        "--add-chi-eff-mixture",
        action="store_true",
        help=(
            "Model the effective inspiral spin parameter (chi_eff) using a 2-component "
            "Truncated Normal mixture distribution: (1-zeta) * comp1 + zeta * comp2. Bounded "
            "by default on [-1, 1], tracking parameters for relative weight ('_zeta_'), locations, "
            "and scales for both sub-populations."
        ),
    )
    chi_eff_group.add_argument(
        "--add-skew-normal-chi-eff",
        action="store_true",
        help=(
            "Model the effective inspiral spin parameter (chi_eff) using a Skew Normal distribution "
            "truncated to [-1, 1], following the GWTC-4 population convention. Uses location ('_loc_'), "
            "scale ('_scale_'), and asymmetry/skewness ('_epsilon_') parameters."
        ),
    )

    model_group.add_argument(
        "--add-truncated-normal-chi-p",
        action="store_true",
        help=(
            "Model the effective precessing spin parameter (chi_p) using a Truncated Normal "
            "distribution. Bounded by definition on [0, 1], parameterized by its location, "
            "scale, and optional custom limits."
        ),
    )
    model_group.add_argument(
        "--add-tilt",
        action="store_true",
        help=(
            "Model the spin tilt angles jointly via their cosines (cos_tilt_1, cos_tilt_2) using "
            "a mixture distribution. Combines an isotropic component (uniform in cos tilt) with a "
            "preferentially aligned component (truncated normals peaked at cos_tilt = 1), weighted "
            "by 'cos_tilt_zeta_{TYPE}_{i}'."
        ),
    )

    eccentricity_group = model_group.add_mutually_exclusive_group()
    eccentricity_group.add_argument(
        "--add-eccentricity-mixture",
        action="store_true",
        help=(
            "Model orbital eccentricity at reference frequency using a 2-component Truncated "
            "Normal mixture model ((1-zeta) * comp1 + zeta * comp2). Allows modeling a population "
            "split between highly circularized and dynamically excited eccentric binaries."
        ),
    )
    eccentricity_group.add_argument(
        "--add-eccentricity-powerlaw",
        action="store_true",
        help=(
            "Model orbital eccentricity using a doubly truncated power-law distribution. "
            "Requires power-law index ('_alpha_') along with low and high parameter limits "
            "to capture power-law decay profiles typical of specific astrophysical environments."
        ),
    )

    model_group.add_argument(
        "--add-mean-anomaly",
        action="store_true",
        help=(
            "Include orbital mean anomaly at reference frequency in the model, distributed "
            "uniformly between specified '_low_' and '_high_' boundary parameters."
        ),
    )

    redshift_group = model_group.add_mutually_exclusive_group()
    redshift_group.add_argument(
        "--add-powerlaw-redshift",
        action="store_true",
        help=(
            "Model the merger rate evolution over redshift z using a standard power-law model "
            "where dR/dz proportional to (1+z)^(kappa-1) * dV_c/dz. Parameterized by evolution "
            "index 'kappa' and a hard cutoff maximum redshift 'z_max'."
        ),
    )
    redshift_group.add_argument(
        "--add-madau-dickinson-redshift",
        action="store_true",
        help=(
            "Model the merger rate evolution over redshift z using a Madau-Dickinson cosmic star "
            "formation profile, modified by a power-law time delay distribution. Tracks slope parameters "
            "'_kappa_' (low-z), '_gamma_' (high-z), turnover peak position '_z_peak_', and maximum cutoff '_z_max_'."
        ),
    )

    return parser
