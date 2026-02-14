# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from typing import Callable, List, Optional, Tuple

from jax import numpy as jnp
from jaxtyping import Array

from gwkanal.utils.checks import check_min_concentration_for_beta_dist
from gwkanal.utils.common import expand_arguments
from gwkokab.parameters import Parameters as P


def where_fns_list(
    use_beta_spin_magnitude: bool,
) -> Optional[List[Callable[..., Array]]]:
    where_fns = []

    if use_beta_spin_magnitude:

        def positive_concentration(**kwargs) -> Array:
            N_pl: int = kwargs.get("N_pl")  # type: ignore
            N_g: int = kwargs.get("N_g")  # type: ignore
            mask = jnp.ones((), dtype=bool)
            for n_pl in range(N_pl):
                chi_mean: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE + "_mean_pl_" + str(n_pl)
                )  # type: ignore
                chi_variance: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE + "_variance_pl_" + str(n_pl)
                )  # type: ignore
                mask &= check_min_concentration_for_beta_dist(chi_mean, chi_variance)
            for n_g in range(N_g):
                chi_mean: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE + "_mean_g_" + str(n_g)
                )  # type: ignore
                chi_variance: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE + "_variance_g_" + str(n_g)
                )  # type: ignore
                mask &= check_min_concentration_for_beta_dist(chi_mean, chi_variance)
            return mask

        where_fns.append(positive_concentration)

    return where_fns if len(where_fns) > 0 else None


class NSmoothedPowerlawMSmoothedGaussianCore:
    def __init__(
        self,
        N_pl: int,
        N_g: int,
        use_beta_spin_magnitude: bool,
        use_spin_magnitude_mixture: bool,
        use_chi_eff_mixture: bool,
        use_skew_normal_chi_eff: bool,
        use_truncated_normal_chi_p: bool,
        use_tilt: bool,
        use_eccentricity_mixture: bool,
        use_redshift: bool,
    ) -> None:
        self.N_pl = N_pl
        self.N_g = N_g
        self.use_beta_spin_magnitude = use_beta_spin_magnitude
        self.use_spin_magnitude_mixture = use_spin_magnitude_mixture
        self.use_chi_eff_mixture = use_chi_eff_mixture
        self.use_skew_normal_chi_eff = use_skew_normal_chi_eff
        self.use_truncated_normal_chi_p = use_truncated_normal_chi_p
        self.use_tilt = use_tilt
        self.use_eccentricity_mixture = use_eccentricity_mixture
        self.use_redshift = use_redshift

    def modify_model_params(self, params: dict) -> dict:
        params.update(
            {
                "N_pl": self.N_pl,
                "N_g": self.N_g,
                "use_beta_spin_magnitude": self.use_beta_spin_magnitude,
                "use_spin_magnitude_mixture": self.use_spin_magnitude_mixture,
                "use_chi_eff_mixture": self.use_chi_eff_mixture,
                "use_skew_normal_chi_eff": self.use_skew_normal_chi_eff,
                "use_truncated_normal_chi_p": self.use_truncated_normal_chi_p,
                "use_tilt": self.use_tilt,
                "use_eccentricity_mixture": self.use_eccentricity_mixture,
                "use_redshift": self.use_redshift,
            }
        )
        return params

    @property
    def parameters(self) -> List[str]:
        names = [P.PRIMARY_MASS_SOURCE]
        if self.use_beta_spin_magnitude or self.use_spin_magnitude_mixture:
            names.append(P.PRIMARY_SPIN_MAGNITUDE)
            names.append(P.SECONDARY_SPIN_MAGNITUDE)
        if self.use_chi_eff_mixture or self.use_skew_normal_chi_eff:
            names.append(P.EFFECTIVE_SPIN)
        if self.use_truncated_normal_chi_p:
            names.append(P.PRECESSING_SPIN)
        if self.use_tilt:
            names.extend([P.COS_TILT_1, P.COS_TILT_2])
        if self.use_eccentricity_mixture:
            names.append(P.ECCENTRICITY)
        if self.use_redshift:
            names.append(P.REDSHIFT)
        names.append(P.MASS_RATIO)
        return [name.value for name in names]

    @property
    def model_parameters(self) -> List[str]:
        all_params: List[Tuple[str, int]] = [
            ("alpha_pl", self.N_pl),
            ("lambda", self.N_pl + self.N_g - 1),
            ("m1_high_g", self.N_g),
            ("m1_loc_g", self.N_g),
            ("m1_low_g", self.N_g),
            ("m1_scale_g", self.N_g),
            ("mmax_pl", self.N_pl),
            ("mmin_pl", self.N_pl),
        ]

        if self.use_spin_magnitude_mixture:
            all_params.extend(
                [
                    ("a_zeta_g", self.N_g),
                    ("a_zeta_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_pl", self.N_pl),
                ]
            )

        if self.use_beta_spin_magnitude:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_MAGNITUDE + "_mean_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_mean_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_variance_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_variance_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_mean_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_mean_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_variance_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_variance_pl", self.N_pl),
                ]
            )

        if self.use_chi_eff_mixture:
            all_params.extend(
                [
                    (P.EFFECTIVE_SPIN + "_comp1_high_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_comp1_high_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_comp1_loc_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_comp1_loc_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_comp1_low_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_comp1_low_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_comp1_scale_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_comp1_scale_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_comp2_high_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_comp2_high_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_comp2_loc_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_comp2_loc_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_comp2_low_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_comp2_low_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_comp2_scale_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_comp2_scale_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_zeta_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_zeta_pl", self.N_pl),
                ]
            )

        if self.use_skew_normal_chi_eff:
            all_params.extend(
                [
                    (P.EFFECTIVE_SPIN + "_epsilon_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_epsilon_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_loc_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_loc_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN + "_scale_g", self.N_g),
                    (P.EFFECTIVE_SPIN + "_scale_pl", self.N_pl),
                ]
            )

        if self.use_truncated_normal_chi_p:
            all_params.extend(
                [
                    (P.PRECESSING_SPIN + "_high_g", self.N_g),
                    (P.PRECESSING_SPIN + "_high_pl", self.N_pl),
                    (P.PRECESSING_SPIN + "_loc_g", self.N_g),
                    (P.PRECESSING_SPIN + "_loc_pl", self.N_pl),
                    (P.PRECESSING_SPIN + "_low_g", self.N_g),
                    (P.PRECESSING_SPIN + "_low_pl", self.N_pl),
                    (P.PRECESSING_SPIN + "_scale_g", self.N_g),
                    (P.PRECESSING_SPIN + "_scale_pl", self.N_pl),
                ]
            )

        if self.use_tilt:
            all_params.extend(
                [
                    ("cos_tilt_zeta_g", self.N_g),
                    ("cos_tilt_zeta_pl", self.N_pl),
                    (P.COS_TILT_1 + "_loc_pl", self.N_pl),
                    (P.COS_TILT_1 + "_loc_g", self.N_g),
                    (P.COS_TILT_1 + "_minimum_pl", self.N_pl),
                    (P.COS_TILT_1 + "_minimum_g", self.N_g),
                    (P.COS_TILT_1 + "_scale_pl", self.N_pl),
                    (P.COS_TILT_1 + "_scale_g", self.N_g),
                    (P.COS_TILT_2 + "_loc_pl", self.N_pl),
                    (P.COS_TILT_2 + "_loc_g", self.N_g),
                    (P.COS_TILT_2 + "_minimum_pl", self.N_pl),
                    (P.COS_TILT_2 + "_minimum_g", self.N_g),
                    (P.COS_TILT_2 + "_scale_pl", self.N_pl),
                    (P.COS_TILT_2 + "_scale_g", self.N_g),
                ]
            )

        if self.use_eccentricity_mixture:
            all_params.extend(
                [
                    (P.ECCENTRICITY + "_comp1_high_g", self.N_g),
                    (P.ECCENTRICITY + "_comp1_high_pl", self.N_pl),
                    (P.ECCENTRICITY + "_comp1_loc_g", self.N_g),
                    (P.ECCENTRICITY + "_comp1_loc_pl", self.N_pl),
                    (P.ECCENTRICITY + "_comp1_low_g", self.N_g),
                    (P.ECCENTRICITY + "_comp1_low_pl", self.N_pl),
                    (P.ECCENTRICITY + "_comp1_scale_g", self.N_g),
                    (P.ECCENTRICITY + "_comp1_scale_pl", self.N_pl),
                    (P.ECCENTRICITY + "_comp2_high_g", self.N_g),
                    (P.ECCENTRICITY + "_comp2_high_pl", self.N_pl),
                    (P.ECCENTRICITY + "_comp2_loc_g", self.N_g),
                    (P.ECCENTRICITY + "_comp2_loc_pl", self.N_pl),
                    (P.ECCENTRICITY + "_comp2_low_g", self.N_g),
                    (P.ECCENTRICITY + "_comp2_low_pl", self.N_pl),
                    (P.ECCENTRICITY + "_comp2_scale_g", self.N_g),
                    (P.ECCENTRICITY + "_comp2_scale_pl", self.N_pl),
                    (P.ECCENTRICITY + "_zeta_g", self.N_g),
                    (P.ECCENTRICITY + "_zeta_pl", self.N_pl),
                ]
            )

        if self.use_redshift:
            all_params.extend(
                [
                    (P.REDSHIFT + "_kappa_g", self.N_g),
                    (P.REDSHIFT + "_kappa_pl", self.N_pl),
                    (P.REDSHIFT + "_z_max_g", self.N_g),
                    (P.REDSHIFT + "_z_max_pl", self.N_pl),
                ]
            )

        extended_params = [
            "beta",
            "delta_m",
            "log_rate",
            "mmax",
            "mmin",
        ]
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--n-pl",
        type=int,
        help="Number of power-law components in the mass model.",
    )
    model_group.add_argument(
        "--n-g",
        type=int,
        help="Number of Gaussian components in the mass model.",
    )

    spin_group = model_group.add_mutually_exclusive_group()
    spin_group.add_argument(
        "--add-beta-spin-magnitude",
        action="store_true",
        help="Include beta spin magnitude parameters in the model.",
    )
    spin_group.add_argument(
        "--add-spin-magnitude-mixture",
        action="store_true",
        help="Include truncated normal spin magnitude parameters in the model.",
    )

    chi_eff_group = model_group.add_mutually_exclusive_group()
    chi_eff_group.add_argument(
        "--add-chi-eff-mixture",
        action="store_true",
        help="Include chi_eff mixture parameters in the model.",
    )
    chi_eff_group.add_argument(
        "--add-skew-normal-chi-eff",
        action="store_true",
        help="Include skew normal chi_eff parameters in the model.",
    )

    model_group.add_argument(
        "--add-truncated-normal-chi-p",
        action="store_true",
        help="Include truncated normal chi_p parameters in the model.",
    )
    model_group.add_argument(
        "--add-tilt",
        action="store_true",
        help="Include tilt parameters in the model.",
    )
    model_group.add_argument(
        "--add-eccentricity-mixture",
        action="store_true",
        help="Include eccentricity mixture in the model.",
    )
    model_group.add_argument(
        "--add-redshift",
        action="store_true",
        help="Include redshift parameter in the model",
    )

    return parser
