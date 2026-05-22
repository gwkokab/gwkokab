# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from typing import Callable, List, Optional

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
            N_spl: int = kwargs.get("N_spl")  # type: ignore
            N_bpl: int = kwargs.get("N_bpl")  # type: ignore
            N_gpl: int = kwargs.get("N_gpl")  # type: ignore
            N_gg: int = kwargs.get("N_gg")  # type: ignore

            mask = jnp.ones((), dtype=bool)

            for ctype, n in zip(
                ("spl", "bpl", "gpl", "gg"), (N_spl, N_bpl, N_gpl, N_gg)
            ):
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


class MultiSourceModelCore:
    def __init__(
        self,
        N_spl: int,
        N_bpl: int,
        N_gpl: int,
        N_gg: int,
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
        self.N_spl = N_spl
        self.N_bpl = N_bpl
        self.N_gpl = N_gpl
        self.N_gg = N_gg
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
        params.update({
            "N_spl": self.N_spl,
            "N_bpl": self.N_bpl,
            "N_gpl": self.N_gpl,
            "N_gg": self.N_gg,
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
        names = [P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE]
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
        all_params: list[tuple[str, int]] = [
            ("log_rate", self.N_spl + self.N_bpl + self.N_gpl + self.N_gg)
        ]

        component_types_and_count = zip(
            ["spl", "bpl", "gpl", "gg"],
            [self.N_spl, self.N_bpl, self.N_gpl, self.N_gg],
        )

        for ct, count in component_types_and_count:
            all_params_names = []
            if ct == "spl":
                all_params_names.extend([
                    "alpha_",
                    "beta_",
                    "delta_m1_",
                    "delta_m2_",
                    "m1min_",
                    "m2min_",
                    "mmax_",
                ])
            if ct == "bpl":
                all_params_names.extend([
                    "alpha1_",
                    "alpha2_",
                    "beta_",
                    "delta_m1_",
                    "delta_m2_",
                    "m1min_",
                    "m2min_",
                    "mbreak_",
                    "mmax_",
                ])
            if ct == "gpl":
                all_params_names.extend([
                    "beta_",
                    "loc_",
                    "mmax_",
                    "mmin_",
                    "scale_",
                ])
            if ct == "gg":
                all_params_names.extend([
                    "m1_high_",
                    "m1_loc_",
                    "m1_low_",
                    "m1_scale_",
                    "m2_high_",
                    "m2_loc_",
                    "m2_low_",
                    "m2_scale_",
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

        extended_params = []
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--n-spl",
        type=int,
        default=0,
        help="Number of smoothed power law primary mass ratio components in the mass model.",
    )
    model_group.add_argument(
        "--n-bpl",
        type=int,
        default=0,
        help="Number of smoothed broken power law components in the mass model.",
    )
    model_group.add_argument(
        "--n-gpl",
        type=int,
        default=0,
        help="Number of smoothed Gaussian components in the mass model.",
    )
    model_group.add_argument(
        "--n-gg",
        type=int,
        default=0,
        help="Number of Gaussian components for both component masses model.",
    )

    spin_group = model_group.add_mutually_exclusive_group()
    spin_group.add_argument(
        "--add-beta-spin-magnitude",
        action="store_true",
        help="Include beta spin parameters in the model.",
    )
    spin_group.add_argument(
        "--add-spin-magnitude-mixture",
        action="store_true",
        help="Include spin parameters mixture in the model.",
    )

    model_group.add_argument(
        "--add-truncated-normal-spin-x",
        action="store_true",
        help="Include truncated normal spin x parameters in the model.",
    )
    model_group.add_argument(
        "--add-truncated-normal-spin-y",
        action="store_true",
        help="Include truncated normal spin y parameters in the model.",
    )
    model_group.add_argument(
        "--add-truncated-normal-spin-z",
        action="store_true",
        help="Include truncated normal spin z parameters in the model.",
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

    eccentricity_group = model_group.add_mutually_exclusive_group()
    eccentricity_group.add_argument(
        "--add-eccentricity-mixture",
        action="store_true",
        help="Include truncated normal eccentricity in the model.",
    )
    eccentricity_group.add_argument(
        "--add-eccentricity-powerlaw",
        action="store_true",
        help="Include power law eccentricity in the model.",
    )

    model_group.add_argument(
        "--add-mean-anomaly",
        action="store_true",
        help="Include mean_anomaly parameter in the model",
    )

    redshift_group = model_group.add_mutually_exclusive_group()
    redshift_group.add_argument(
        "--add-powerlaw-redshift",
        action="store_true",
        help="Include redshift parameter in the model",
    )
    redshift_group.add_argument(
        "--add-madau-dickinson-redshift",
        action="store_true",
        help="Redshift modeled by Madau-Dickinson Model",
    )

    return parser
