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
            N_g: int = kwargs.get("N_g")  # type: ignore

            mask = jnp.ones((), dtype=bool)

            for ctype, n in zip(("spl", "bpl", "g"), (N_spl, N_bpl, N_g)):
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
    def __init__(
        self,
        N_spl: int,
        N_bpl: int,
        N_g: int,
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
        use_redshift: bool,
        use_cos_iota: bool,
        use_phi_12: bool,
        use_polarization_angle: bool,
        use_right_ascension: bool,
        use_sin_declination: bool,
        use_detection_time: bool,
        use_phi_1: bool,
        use_phi_2: bool,
        use_phi_orb: bool,
        use_mean_anomaly: bool,
    ) -> None:
        self.N_spl = N_spl
        self.N_bpl = N_bpl
        self.N_g = N_g
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
        self.use_redshift = use_redshift
        self.use_cos_iota = use_cos_iota
        self.use_phi_12 = use_phi_12
        self.use_polarization_angle = use_polarization_angle
        self.use_right_ascension = use_right_ascension
        self.use_sin_declination = use_sin_declination
        self.use_detection_time = use_detection_time
        self.use_phi_1 = use_phi_1
        self.use_phi_2 = use_phi_2
        self.use_phi_orb = use_phi_orb
        self.use_mean_anomaly = use_mean_anomaly

    def modify_model_params(self, params: dict) -> dict:
        params.update(
            {
                "N_spl": self.N_spl,
                "N_bpl": self.N_bpl,
                "N_g": self.N_g,
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
                "use_redshift": self.use_redshift,
                "use_cos_iota": self.use_cos_iota,
                "use_phi_12": self.use_phi_12,
                "use_polarization_angle": self.use_polarization_angle,
                "use_right_ascension": self.use_right_ascension,
                "use_sin_declination": self.use_sin_declination,
                "use_detection_time": self.use_detection_time,
                "use_phi_1": self.use_phi_1,
                "use_phi_2": self.use_phi_2,
                "use_phi_orb": self.use_phi_orb,
                "use_mean_anomaly": self.use_mean_anomaly,
            }
        )
        return params

    @property
    def parameters(self) -> tuple[str, ...]:
        names = [P.PRIMARY_MASS_SOURCE]
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
        if self.use_phi_12:
            names.append(P.PHI_12)
        if self.use_eccentricity_mixture or self.use_eccentricity_powerlaw:
            names.append(P.ECCENTRICITY)
        if self.use_redshift:
            names.append(P.REDSHIFT)
        if self.use_right_ascension:
            names.append(P.RIGHT_ASCENSION)
        if self.use_sin_declination:
            names.append(P.SIN_DECLINATION)
        if self.use_detection_time:
            names.append(P.DETECTION_TIME)
        if self.use_cos_iota:
            names.append(P.COS_IOTA)
        if self.use_polarization_angle:
            names.append(P.POLARIZATION_ANGLE)
        if self.use_phi_1:
            names.append(P.PHI_1)
        if self.use_phi_2:
            names.append(P.PHI_2)
        if self.use_phi_orb:
            names.append(P.PHI_ORB)
        if self.use_mean_anomaly:
            names.append(P.MEAN_ANOMALY)
        names.append(P.MASS_RATIO)
        return names

    @property
    def model_parameters(self) -> list[str]:
        # fmt: off
        component_types_and_count = zip(
            ("spl"     , "bpl"     , "g"     ),
            (self.N_spl, self.N_bpl, self.N_g),
        )
        # fmt: on

        all_params: list[tuple[str, int]] = []

        for ct, count in component_types_and_count:
            all_params_names = []

            if ct == "spl":
                all_params_names.extend(["alpha_", "mmax_", "mmin_"])

            if ct == "bpl":
                all_params_names.extend(
                    [
                        "alpha_",
                        "m1min_",
                        "m2min_",
                        "mmax_",
                    ]
                )

            if ct == "g":
                all_params_names.extend(["m1_loc_", "m1_scale_", "m1_low_", "m1_high_"])

            if self.use_spin_magnitude_mixture:
                all_params_names.extend(
                    [
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
                    ]
                )

            if self.use_beta_spin_magnitude:
                all_params_names.extend(
                    [
                        P.PRIMARY_SPIN_MAGNITUDE + "_mean_",
                        P.PRIMARY_SPIN_MAGNITUDE + "_variance_",
                        P.SECONDARY_SPIN_MAGNITUDE + "_mean_",
                        P.SECONDARY_SPIN_MAGNITUDE + "_variance_",
                    ]
                )

            if self.use_truncated_normal_spin_x:
                all_params_names.extend(
                    [
                        P.PRIMARY_SPIN_X + "_high_",
                        P.PRIMARY_SPIN_X + "_loc_",
                        P.PRIMARY_SPIN_X + "_low_",
                        P.PRIMARY_SPIN_X + "_scale_",
                        P.SECONDARY_SPIN_X + "_high_",
                        P.SECONDARY_SPIN_X + "_loc_",
                        P.SECONDARY_SPIN_X + "_low_",
                        P.SECONDARY_SPIN_X + "_scale_",
                    ]
                )

            if self.use_truncated_normal_spin_y:
                all_params_names.extend(
                    [
                        P.PRIMARY_SPIN_Y + "_high_",
                        P.PRIMARY_SPIN_Y + "_loc_",
                        P.PRIMARY_SPIN_Y + "_low_",
                        P.PRIMARY_SPIN_Y + "_scale_",
                        P.SECONDARY_SPIN_Y + "_high_",
                        P.SECONDARY_SPIN_Y + "_loc_",
                        P.SECONDARY_SPIN_Y + "_low_",
                        P.SECONDARY_SPIN_Y + "_scale_",
                    ]
                )

            if self.use_truncated_normal_spin_z:
                all_params_names.extend(
                    [
                        P.PRIMARY_SPIN_Z + "_high_",
                        P.PRIMARY_SPIN_Z + "_loc_",
                        P.PRIMARY_SPIN_Z + "_low_",
                        P.PRIMARY_SPIN_Z + "_scale_",
                        P.SECONDARY_SPIN_Z + "_high_",
                        P.SECONDARY_SPIN_Z + "_loc_",
                        P.SECONDARY_SPIN_Z + "_low_",
                        P.SECONDARY_SPIN_Z + "_scale_",
                    ]
                )

            if self.use_chi_eff_mixture:
                all_params_names.extend(
                    [
                        P.EFFECTIVE_SPIN + "_comp1_high_",
                        P.EFFECTIVE_SPIN + "_comp1_loc_",
                        P.EFFECTIVE_SPIN + "_comp1_low_",
                        P.EFFECTIVE_SPIN + "_comp1_scale_",
                        P.EFFECTIVE_SPIN + "_comp2_high_",
                        P.EFFECTIVE_SPIN + "_comp2_loc_",
                        P.EFFECTIVE_SPIN + "_comp2_low_",
                        P.EFFECTIVE_SPIN + "_comp2_scale_",
                        P.EFFECTIVE_SPIN + "_zeta_",
                    ]
                )

            if self.use_skew_normal_chi_eff:
                all_params_names.extend(
                    [
                        P.EFFECTIVE_SPIN + "_epsilon_",
                        P.EFFECTIVE_SPIN + "_loc_",
                        P.EFFECTIVE_SPIN + "_scale_",
                    ]
                )

            if self.use_truncated_normal_chi_p:
                all_params_names.extend(
                    [
                        P.PRECESSING_SPIN + "_high_",
                        P.PRECESSING_SPIN + "_loc_",
                        P.PRECESSING_SPIN + "_low_",
                        P.PRECESSING_SPIN + "_scale_",
                    ]
                )

            if self.use_tilt:
                all_params_names.extend(
                    [
                        "cos_tilt_zeta_",
                        P.COS_TILT_1 + "_scale_",
                        P.COS_TILT_2 + "_scale_",
                    ]
                )

            if self.use_phi_12:
                all_params_names.extend(
                    [
                        P.PHI_12 + "_high_",
                        P.PHI_12 + "_loc_",
                        P.PHI_12 + "_low_",
                        P.PHI_12 + "_scale_",
                    ]
                )

            if self.use_eccentricity_mixture:
                all_params_names.extend(
                    [
                        P.ECCENTRICITY + "_comp1_high_",
                        P.ECCENTRICITY + "_comp1_loc_",
                        P.ECCENTRICITY + "_comp1_low_",
                        P.ECCENTRICITY + "_comp1_scale_",
                        P.ECCENTRICITY + "_comp2_high_",
                        P.ECCENTRICITY + "_comp2_loc_",
                        P.ECCENTRICITY + "_comp2_low_",
                        P.ECCENTRICITY + "_comp2_scale_",
                        P.ECCENTRICITY + "_zeta_",
                    ]
                )

            if self.use_redshift:
                all_params_names.extend(
                    [P.REDSHIFT + "_kappa_", P.REDSHIFT + "_z_max_"]
                )

            if self.use_right_ascension:
                all_params_names.extend(
                    [
                        P.RIGHT_ASCENSION + "_high_",
                        P.RIGHT_ASCENSION + "_loc_",
                        P.RIGHT_ASCENSION + "_low_",
                        P.RIGHT_ASCENSION + "_scale_",
                    ]
                )

            if self.use_sin_declination:
                all_params_names.extend(
                    [
                        P.SIN_DECLINATION + "_high_",
                        P.SIN_DECLINATION + "_loc_",
                        P.SIN_DECLINATION + "_low_",
                        P.SIN_DECLINATION + "_scale_",
                    ]
                )

            if self.use_detection_time:
                all_params_names.extend(
                    [
                        P.DETECTION_TIME + "_high_",
                        P.DETECTION_TIME + "_low_",
                    ]
                )

            if self.use_cos_iota:
                all_params_names.extend(
                    [
                        P.COS_IOTA + "_high_",
                        P.COS_IOTA + "_loc_",
                        P.COS_IOTA + "_low_",
                        P.COS_IOTA + "_scale_",
                    ]
                )

            if self.use_polarization_angle:
                all_params_names.extend(
                    [
                        P.POLARIZATION_ANGLE + "_high_",
                        P.POLARIZATION_ANGLE + "_loc_",
                        P.POLARIZATION_ANGLE + "_low_",
                        P.POLARIZATION_ANGLE + "_scale_",
                    ]
                )

            all_params.extend([(name + ct, count) for name in all_params_names])

        extended_params = [
            "beta",
            "delta_m1",
            "delta_m2",
            "log_rate",
            "m1max",
            "m1min",
            "m2min",
        ]
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
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
        "--n-g",
        type=int,
        default=0,
        help="Number of Gaussian components for primary mass.",
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
    model_group.add_argument(
        "--add-redshift",
        action="store_true",
        help="Include redshift parameter in the model",
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
        "--add-cos-iota",
        action="store_true",
        help="Include cos_iota parameter in the model",
    )
    model_group.add_argument(
        "--add-phi-12",
        action="store_true",
        help="Include phi_12 parameter in the model",
    )
    model_group.add_argument(
        "--add-polarization-angle",
        action="store_true",
        help="Include polarization_angle parameter in the model",
    )
    model_group.add_argument(
        "--add-right-ascension",
        action="store_true",
        help="Include right_ascension parameter in the model",
    )
    model_group.add_argument(
        "--add-sin-declination",
        action="store_true",
        help="Include sin_declination parameter in the model",
    )
    model_group.add_argument(
        "--add-detection-time",
        action="store_true",
        help="Include detection_time parameter in the model",
    )
    model_group.add_argument(
        "--add-phi-1",
        action="store_true",
        help="Include phi_1 parameter in the model",
    )
    model_group.add_argument(
        "--add-phi-2",
        action="store_true",
        help="Include phi_2 parameter in the model",
    )
    model_group.add_argument(
        "--add-phi-orb",
        action="store_true",
        help="Include phi_orb parameter in the model",
    )
    model_group.add_argument(
        "--add-mean-anomaly",
        action="store_true",
        help="Include mean_anomaly parameter in the model",
    )
    return parser
