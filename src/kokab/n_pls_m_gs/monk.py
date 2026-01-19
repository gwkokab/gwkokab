# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Dict, List, Tuple, Union

from gwkokab.models import NPowerlawMGaussian
from gwkokab.parameters import Parameters as P
from kokab.core.monk import Monk, monk_arg_parser
from kokab.utils.common import expand_arguments
from kokab.utils.logger import log_info


class NPowerlawMGaussianMonk(Monk):
    def __init__(
        self,
        N_pl: int,
        N_g: int,
        has_beta_spin_magnitude: bool,
        use_spin_magnitude_mixture: bool,
        has_truncated_normal_spin_x: bool,
        has_truncated_normal_spin_y: bool,
        has_truncated_normal_spin_z: bool,
        use_chi_eff_mixture: bool,
        use_skew_normal_chi_eff: bool,
        use_truncated_normal_chi_p: bool,
        has_tilt: bool,
        use_eccentricity_mixture: bool,
        has_redshift: bool,
        has_cos_iota: bool,
        has_phi_12: bool,
        has_polarization_angle: bool,
        has_right_ascension: bool,
        has_sin_declination: bool,
        has_detection_time: bool,
        data_filename: str,
        seed: int,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_samples: int,
        minimum_mc_error: float,
        n_checkpoints: int,
        n_max_steps: int,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        self.N_pl = N_pl
        self.N_g = N_g
        self.has_beta_spin_magnitude = has_beta_spin_magnitude
        self.use_spin_magnitude_mixture = use_spin_magnitude_mixture
        self.has_truncated_normal_spin_x = has_truncated_normal_spin_x
        self.has_truncated_normal_spin_y = has_truncated_normal_spin_y
        self.has_truncated_normal_spin_z = has_truncated_normal_spin_z
        self.use_chi_eff_mixture = use_chi_eff_mixture
        self.use_skew_normal_chi_eff = use_skew_normal_chi_eff
        self.use_truncated_normal_chi_p = use_truncated_normal_chi_p
        self.has_tilt = has_tilt
        self.use_eccentricity_mixture = use_eccentricity_mixture
        self.has_redshift = has_redshift
        self.has_cos_iota = has_cos_iota
        self.has_phi_12 = has_phi_12
        self.has_polarization_angle = has_polarization_angle
        self.has_right_ascension = has_right_ascension
        self.has_sin_declination = has_sin_declination
        self.has_detection_time = has_detection_time

        super().__init__(
            NPowerlawMGaussian,
            data_filename,
            seed,
            prior_filename,
            poisson_mean_filename,
            sampler_settings_filename,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            analysis_name="n_pls_m_gs",
            n_samples=n_samples,
            minimum_mc_error=minimum_mc_error,
            n_checkpoints=n_checkpoints,
            n_max_steps=n_max_steps,
        )

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        return {
            "N_pl": self.N_pl,
            "N_g": self.N_g,
            "use_beta_spin_magnitude": self.has_beta_spin_magnitude,
            "use_spin_magnitude_mixture": self.use_spin_magnitude_mixture,
            "use_truncated_normal_spin_x": self.has_truncated_normal_spin_x,
            "use_truncated_normal_spin_y": self.has_truncated_normal_spin_y,
            "use_truncated_normal_spin_z": self.has_truncated_normal_spin_z,
            "use_chi_eff_mixture": self.use_chi_eff_mixture,
            "use_skew_normal_chi_eff": self.use_skew_normal_chi_eff,
            "use_truncated_normal_chi_p": self.use_truncated_normal_chi_p,
            "use_tilt": self.has_tilt,
            "use_eccentricity_mixture": self.use_eccentricity_mixture,
            "use_redshift": self.has_redshift,
            "use_cos_iota": self.has_cos_iota,
            "use_phi_12": self.has_phi_12,
            "use_polarization_angle": self.has_polarization_angle,
            "use_right_ascension": self.has_right_ascension,
            "use_sin_declination": self.has_sin_declination,
            "use_detection_time": self.has_detection_time,
        }

    @property
    def parameters(self) -> List[str]:
        names = [P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE]
        if self.has_beta_spin_magnitude or self.use_spin_magnitude_mixture:
            names.append(P.PRIMARY_SPIN_MAGNITUDE)
            names.append(P.SECONDARY_SPIN_MAGNITUDE)
        if self.has_truncated_normal_spin_x:
            names.append(P.PRIMARY_SPIN_X)
            names.append(P.SECONDARY_SPIN_X)
        if self.has_truncated_normal_spin_y:
            names.append(P.PRIMARY_SPIN_Y)
            names.append(P.SECONDARY_SPIN_Y)
        if self.has_truncated_normal_spin_z:
            names.append(P.PRIMARY_SPIN_Z)
            names.append(P.SECONDARY_SPIN_Z)
        if self.use_chi_eff_mixture or self.use_skew_normal_chi_eff:
            names.append(P.EFFECTIVE_SPIN)
        if self.use_truncated_normal_chi_p:
            names.append(P.PRECESSING_SPIN)
        if self.has_tilt:
            names.extend([P.COS_TILT_1, P.COS_TILT_2])
        if self.has_phi_12:
            names.append(P.PHI_12)
        if self.use_eccentricity_mixture:
            names.append(P.ECCENTRICITY)
        if self.has_redshift:
            names.append(P.REDSHIFT)
        if self.has_right_ascension:
            names.append(P.RIGHT_ASCENSION)
        if self.has_sin_declination:
            names.append(P.SIN_DECLINATION)
        if self.has_detection_time:
            names.append(P.DETECTION_TIME)
        if self.has_cos_iota:
            names.append(P.COS_IOTA)
        if self.has_polarization_angle:
            names.append(P.POLARIZATION_ANGLE)
        return names

    @property
    def model_parameters(self) -> List[str]:
        all_params: List[Tuple[str, int]] = [
            ("log_rate", self.N_pl + self.N_g),
            ("alpha_pl", self.N_pl),
            ("beta_pl", self.N_pl),
            ("loc_g", self.N_g),
            ("scale_g", self.N_g),
            ("beta_g", self.N_g),
            ("m1min_g", self.N_g),
            ("m2min_g", self.N_g),
            ("m1max_g", self.N_g),
            ("m2max_g", self.N_g),
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

        if self.has_beta_spin_magnitude:
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

        if self.has_truncated_normal_spin_x:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_X + "_high_g", self.N_g),
                    (P.PRIMARY_SPIN_X + "_high_pl", self.N_pl),
                    (P.PRIMARY_SPIN_X + "_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_X + "_loc_pl", self.N_pl),
                    (P.PRIMARY_SPIN_X + "_low_g", self.N_g),
                    (P.PRIMARY_SPIN_X + "_low_pl", self.N_pl),
                    (P.PRIMARY_SPIN_X + "_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_X + "_scale_pl", self.N_pl),
                    (P.SECONDARY_SPIN_X + "_high_g", self.N_g),
                    (P.SECONDARY_SPIN_X + "_high_pl", self.N_pl),
                    (P.SECONDARY_SPIN_X + "_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_X + "_loc_pl", self.N_pl),
                    (P.SECONDARY_SPIN_X + "_low_g", self.N_g),
                    (P.SECONDARY_SPIN_X + "_low_pl", self.N_pl),
                    (P.SECONDARY_SPIN_X + "_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_X + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_truncated_normal_spin_y:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_Y + "_high_g", self.N_g),
                    (P.PRIMARY_SPIN_Y + "_high_pl", self.N_pl),
                    (P.PRIMARY_SPIN_Y + "_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_Y + "_loc_pl", self.N_pl),
                    (P.PRIMARY_SPIN_Y + "_low_g", self.N_g),
                    (P.PRIMARY_SPIN_Y + "_low_pl", self.N_pl),
                    (P.PRIMARY_SPIN_Y + "_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_Y + "_scale_pl", self.N_pl),
                    (P.SECONDARY_SPIN_Y + "_high_g", self.N_g),
                    (P.SECONDARY_SPIN_Y + "_high_pl", self.N_pl),
                    (P.SECONDARY_SPIN_Y + "_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_Y + "_loc_pl", self.N_pl),
                    (P.SECONDARY_SPIN_Y + "_low_g", self.N_g),
                    (P.SECONDARY_SPIN_Y + "_low_pl", self.N_pl),
                    (P.SECONDARY_SPIN_Y + "_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_Y + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_truncated_normal_spin_z:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_Z + "_high_g", self.N_g),
                    (P.PRIMARY_SPIN_Z + "_high_pl", self.N_pl),
                    (P.PRIMARY_SPIN_Z + "_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_Z + "_loc_pl", self.N_pl),
                    (P.PRIMARY_SPIN_Z + "_low_g", self.N_g),
                    (P.PRIMARY_SPIN_Z + "_low_pl", self.N_pl),
                    (P.PRIMARY_SPIN_Z + "_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_Z + "_scale_pl", self.N_pl),
                    (P.SECONDARY_SPIN_Z + "_high_g", self.N_g),
                    (P.SECONDARY_SPIN_Z + "_high_pl", self.N_pl),
                    (P.SECONDARY_SPIN_Z + "_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_Z + "_loc_pl", self.N_pl),
                    (P.SECONDARY_SPIN_Z + "_low_g", self.N_g),
                    (P.SECONDARY_SPIN_Z + "_low_pl", self.N_pl),
                    (P.SECONDARY_SPIN_Z + "_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_Z + "_scale_pl", self.N_pl),
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

        if self.has_tilt:
            all_params.extend(
                [
                    ("cos_tilt_zeta_g", self.N_g),
                    ("cos_tilt_zeta_pl", self.N_pl),
                    (P.COS_TILT_1 + "_scale_g", self.N_g),
                    (P.COS_TILT_1 + "_scale_pl", self.N_pl),
                    (P.COS_TILT_2 + "_scale_g", self.N_g),
                    (P.COS_TILT_2 + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_phi_12:
            all_params.extend(
                [
                    (P.PHI_12 + "_high_g", self.N_g),
                    (P.PHI_12 + "_high_pl", self.N_pl),
                    (P.PHI_12 + "_loc_g", self.N_g),
                    (P.PHI_12 + "_loc_pl", self.N_pl),
                    (P.PHI_12 + "_low_g", self.N_g),
                    (P.PHI_12 + "_low_pl", self.N_pl),
                    (P.PHI_12 + "_scale_g", self.N_g),
                    (P.PHI_12 + "_scale_pl", self.N_pl),
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

        if self.has_redshift:
            all_params.extend(
                [
                    (P.REDSHIFT + "_kappa_g", self.N_g),
                    (P.REDSHIFT + "_kappa_pl", self.N_pl),
                    (P.REDSHIFT + "_z_max_g", self.N_g),
                    (P.REDSHIFT + "_z_max_pl", self.N_pl),
                ]
            )

        if self.has_right_ascension:
            all_params.extend(
                [
                    (P.RIGHT_ASCENSION + "_high_g", self.N_g),
                    (P.RIGHT_ASCENSION + "_high_pl", self.N_pl),
                    (P.RIGHT_ASCENSION + "_loc_g", self.N_g),
                    (P.RIGHT_ASCENSION + "_loc_pl", self.N_pl),
                    (P.RIGHT_ASCENSION + "_low_g", self.N_g),
                    (P.RIGHT_ASCENSION + "_low_pl", self.N_pl),
                    (P.RIGHT_ASCENSION + "_scale_g", self.N_g),
                    (P.RIGHT_ASCENSION + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_sin_declination:
            all_params.extend(
                [
                    (P.SIN_DECLINATION + "_high_g", self.N_g),
                    (P.SIN_DECLINATION + "_high_pl", self.N_pl),
                    (P.SIN_DECLINATION + "_loc_g", self.N_g),
                    (P.SIN_DECLINATION + "_loc_pl", self.N_pl),
                    (P.SIN_DECLINATION + "_low_g", self.N_g),
                    (P.SIN_DECLINATION + "_low_pl", self.N_pl),
                    (P.SIN_DECLINATION + "_scale_g", self.N_g),
                    (P.SIN_DECLINATION + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_detection_time:
            all_params.extend(
                [
                    (P.DETECTION_TIME + "_high_g", self.N_g),
                    (P.DETECTION_TIME + "_high_pl", self.N_pl),
                    (P.DETECTION_TIME + "_low_g", self.N_g),
                    (P.DETECTION_TIME + "_low_pl", self.N_pl),
                ]
            )

        if self.has_cos_iota:
            all_params.extend(
                [
                    (P.COS_IOTA + "_high_g", self.N_g),
                    (P.COS_IOTA + "_high_pl", self.N_pl),
                    (P.COS_IOTA + "_loc_g", self.N_g),
                    (P.COS_IOTA + "_loc_pl", self.N_pl),
                    (P.COS_IOTA + "_low_g", self.N_g),
                    (P.COS_IOTA + "_low_pl", self.N_pl),
                    (P.COS_IOTA + "_scale_g", self.N_g),
                    (P.COS_IOTA + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_polarization_angle:
            all_params.extend(
                [
                    (P.POLARIZATION_ANGLE + "_high_g", self.N_g),
                    (P.POLARIZATION_ANGLE + "_high_pl", self.N_pl),
                    (P.POLARIZATION_ANGLE + "_loc_g", self.N_g),
                    (P.POLARIZATION_ANGLE + "_loc_pl", self.N_pl),
                    (P.POLARIZATION_ANGLE + "_low_g", self.N_g),
                    (P.POLARIZATION_ANGLE + "_low_pl", self.N_pl),
                    (P.POLARIZATION_ANGLE + "_scale_g", self.N_g),
                    (P.POLARIZATION_ANGLE + "_scale_pl", self.N_pl),
                ]
            )

        extended_params = []
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = monk_arg_parser(parser)

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
    model_group.add_argument(
        "--add-eccentricity-mixture",
        action="store_true",
        help="Include eccentricity mixture in the model.",
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

    args = parser.parse_args()

    log_info(start=True)

    NPowerlawMGaussianMonk(
        N_pl=args.n_pl,
        N_g=args.n_g,
        has_beta_spin_magnitude=args.add_beta_spin_magnitude,
        use_spin_magnitude_mixture=args.add_spin_magnitude_mixture,
        has_truncated_normal_spin_x=args.add_truncated_normal_spin_x,
        has_truncated_normal_spin_y=args.add_truncated_normal_spin_y,
        has_truncated_normal_spin_z=args.add_truncated_normal_spin_z,
        use_chi_eff_mixture=args.add_chi_eff_mixture,
        use_skew_normal_chi_eff=args.add_skew_normal_chi_eff,
        use_truncated_normal_chi_p=args.add_truncated_normal_chi_p,
        has_tilt=args.add_tilt,
        use_eccentricity_mixture=args.add_eccentricity_mixture,
        has_redshift=args.add_redshift,
        has_cos_iota=args.add_cos_iota,
        has_phi_12=args.add_phi_12,
        has_polarization_angle=args.add_polarization_angle,
        has_right_ascension=args.add_right_ascension,
        has_sin_declination=args.add_sin_declination,
        has_detection_time=args.add_detection_time,
        data_filename=args.data_filename,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        n_samples=args.n_samples,
        minimum_mc_error=args.minimum_mc_error,
        n_checkpoints=args.n_checkpoints,
        n_max_steps=args.n_max_steps,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
