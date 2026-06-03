# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from typing import Callable, List, Optional

from jax import numpy as jnp
from jaxtyping import Array

from gwkokab.analysis.utils.checks import check_min_concentration_for_beta_dist
from gwkokab.analysis.utils.common import expand_arguments
from gwkokab.models import NPowerlawMGaussian
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


class NPowerlawMGaussianCore:
    model_fn = NPowerlawMGaussian

    def __init__(
        self,
        N_pl: int,
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
        use_powerlaw_redshift: bool,
        use_madau_dickinson_redshift: bool,
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
        self.N_pl = N_pl
        self.N_g = N_g
        self.use_beta_spin_magnitude = use_beta_spin_magnitude
        self.use_spin_magnitude_mixture = use_spin_magnitude_mixture
        self.use_truncated_normal_spin_x = use_truncated_normal_spin_x
        self.use_truncated_normal_spin_y = use_truncated_normal_spin_y
        self.use_truncated_normal_spin_z = use_truncated_normal_spin_z
        self.use_chi_eff_mixture = use_chi_eff_mixture
        self.use_skew_normal_chi_eff = use_skew_normal_chi_eff
        self.use_truncated_normal_chi_p = use_truncated_normal_chi_p
        self.use_tilt = use_tilt
        self.use_eccentricity_mixture = use_eccentricity_mixture
        self.use_eccentricity_powerlaw = use_eccentricity_powerlaw
        self.use_powerlaw_redshift = use_powerlaw_redshift
        self.use_madau_dickinson_redshift = use_madau_dickinson_redshift
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
        params.update({
            "N_pl": self.N_pl,
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
            "use_powerlaw_redshift": self.use_powerlaw_redshift,
            "use_madau_dickinson_redshift": self.use_madau_dickinson_redshift,
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
        })
        return params

    @property
    def parameters(self) -> tuple[str, ...]:
        names = [P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE]
        if self.use_beta_spin_magnitude:
            names.extend([P.PRIMARY_SPIN_MAGNITUDE, P.SECONDARY_SPIN_MAGNITUDE])
        if self.use_spin_magnitude_mixture:
            names.extend([P.PRIMARY_SPIN_MAGNITUDE, P.SECONDARY_SPIN_MAGNITUDE])
        if self.use_truncated_normal_spin_x:
            names.extend([P.PRIMARY_SPIN_X, P.SECONDARY_SPIN_X])
        if self.use_truncated_normal_spin_y:
            names.extend([P.PRIMARY_SPIN_Y, P.SECONDARY_SPIN_Y])
        if self.use_truncated_normal_spin_z:
            names.extend([P.PRIMARY_SPIN_Z, P.SECONDARY_SPIN_Z])
        if self.use_chi_eff_mixture:
            names.append(P.EFFECTIVE_SPIN)
        if self.use_skew_normal_chi_eff:
            names.append(P.EFFECTIVE_SPIN)
        if self.use_truncated_normal_chi_p:
            names.append(P.PRECESSING_SPIN)
        if self.use_tilt:
            names.extend([P.COS_TILT_1, P.COS_TILT_2])
        if self.use_phi_1:
            names.append(P.PHI_1)
        if self.use_phi_2:
            names.append(P.PHI_2)
        if self.use_phi_12:
            names.append(P.PHI_12)
        if self.use_eccentricity_mixture or self.use_eccentricity_powerlaw:
            names.append(P.ECCENTRICITY)
        if self.use_mean_anomaly:
            names.append(P.MEAN_ANOMALY)
        if self.use_powerlaw_redshift or self.use_madau_dickinson_redshift:
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
        if self.use_phi_orb:
            names.append(P.PHI_ORB)
        return names

    @property
    def model_parameters(self) -> list[str]:
        all_params: list[tuple[str, int]] = [
            ("log_rate", self.N_pl + self.N_g),
            ("alpha_pl", self.N_pl),
            ("beta_pl", self.N_pl),
            ("m1_loc_g", self.N_g),
            ("m2_loc_g", self.N_g),
            ("m1_scale_g", self.N_g),
            ("m2_scale_g", self.N_g),
            ("m1_low_g", self.N_g),
            ("m2_low_g", self.N_g),
            ("m1_high_g", self.N_g),
            ("m2_high_g", self.N_g),
            ("mmax_pl", self.N_pl),
            ("mmin_pl", self.N_pl),
        ]

        if self.use_spin_magnitude_mixture:
            all_params.extend([
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
            ])

        if self.use_beta_spin_magnitude:
            all_params.extend([
                (P.PRIMARY_SPIN_MAGNITUDE + "_mean_g", self.N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_mean_pl", self.N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_variance_g", self.N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_variance_pl", self.N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_mean_g", self.N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_mean_pl", self.N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_variance_g", self.N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_variance_pl", self.N_pl),
            ])

        if self.use_truncated_normal_spin_x:
            all_params.extend([
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
            ])

        if self.use_truncated_normal_spin_y:
            all_params.extend([
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
            ])

        if self.use_truncated_normal_spin_z:
            all_params.extend([
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
            ])

        if self.use_chi_eff_mixture:
            all_params.extend([
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
            ])

        if self.use_skew_normal_chi_eff:
            all_params.extend([
                (P.EFFECTIVE_SPIN + "_epsilon_g", self.N_g),
                (P.EFFECTIVE_SPIN + "_epsilon_pl", self.N_pl),
                (P.EFFECTIVE_SPIN + "_loc_g", self.N_g),
                (P.EFFECTIVE_SPIN + "_loc_pl", self.N_pl),
                (P.EFFECTIVE_SPIN + "_scale_g", self.N_g),
                (P.EFFECTIVE_SPIN + "_scale_pl", self.N_pl),
            ])

        if self.use_truncated_normal_chi_p:
            all_params.extend([
                (P.PRECESSING_SPIN + "_high_g", self.N_g),
                (P.PRECESSING_SPIN + "_high_pl", self.N_pl),
                (P.PRECESSING_SPIN + "_loc_g", self.N_g),
                (P.PRECESSING_SPIN + "_loc_pl", self.N_pl),
                (P.PRECESSING_SPIN + "_low_g", self.N_g),
                (P.PRECESSING_SPIN + "_low_pl", self.N_pl),
                (P.PRECESSING_SPIN + "_scale_g", self.N_g),
                (P.PRECESSING_SPIN + "_scale_pl", self.N_pl),
            ])

        if self.use_tilt:
            all_params.extend([
                ("cos_tilt_zeta_g", self.N_g),
                ("cos_tilt_zeta_pl", self.N_pl),
                (P.COS_TILT_1 + "_high_g", self.N_g),
                (P.COS_TILT_1 + "_high_pl", self.N_pl),
                (P.COS_TILT_1 + "_loc_g", self.N_g),
                (P.COS_TILT_1 + "_loc_pl", self.N_pl),
                (P.COS_TILT_1 + "_low_g", self.N_g),
                (P.COS_TILT_1 + "_low_pl", self.N_pl),
                (P.COS_TILT_1 + "_scale_g", self.N_g),
                (P.COS_TILT_1 + "_scale_pl", self.N_pl),
                (P.COS_TILT_2 + "_high_g", self.N_g),
                (P.COS_TILT_2 + "_high_pl", self.N_pl),
                (P.COS_TILT_2 + "_loc_g", self.N_g),
                (P.COS_TILT_2 + "_loc_pl", self.N_pl),
                (P.COS_TILT_2 + "_low_g", self.N_g),
                (P.COS_TILT_2 + "_low_pl", self.N_pl),
                (P.COS_TILT_2 + "_scale_g", self.N_g),
                (P.COS_TILT_2 + "_scale_pl", self.N_pl),
            ])

        if self.use_phi_1:
            all_params.extend([
                (P.PHI_1 + "_high_g", self.N_g),
                (P.PHI_1 + "_high_pl", self.N_pl),
                (P.PHI_1 + "_low_g", self.N_g),
                (P.PHI_1 + "_low_pl", self.N_pl),
            ])

        if self.use_phi_2:
            all_params.extend([
                (P.PHI_2 + "_high_g", self.N_g),
                (P.PHI_2 + "_high_pl", self.N_pl),
                (P.PHI_2 + "_low_g", self.N_g),
                (P.PHI_2 + "_low_pl", self.N_pl),
            ])

        if self.use_phi_12:
            all_params.extend([
                (P.PHI_12 + "_high_g", self.N_g),
                (P.PHI_12 + "_high_pl", self.N_pl),
                (P.PHI_12 + "_low_g", self.N_g),
                (P.PHI_12 + "_low_pl", self.N_pl),
            ])

        if self.use_eccentricity_mixture:
            all_params.extend([
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
            ])

        if self.use_eccentricity_powerlaw:
            all_params.extend([
                (P.ECCENTRICITY + "_alpha_g", self.N_g),
                (P.ECCENTRICITY + "_alpha_pl", self.N_pl),
                (P.ECCENTRICITY + "_high_g", self.N_g),
                (P.ECCENTRICITY + "_high_pl", self.N_pl),
                (P.ECCENTRICITY + "_low_g", self.N_g),
                (P.ECCENTRICITY + "_low_pl", self.N_pl),
            ])

        if self.use_mean_anomaly:
            all_params.extend([
                (P.MEAN_ANOMALY + "_high_g", self.N_g),
                (P.MEAN_ANOMALY + "_high_pl", self.N_pl),
                (P.MEAN_ANOMALY + "_low_g", self.N_g),
                (P.MEAN_ANOMALY + "_low_pl", self.N_pl),
            ])

        if self.use_powerlaw_redshift:
            all_params.extend([
                (P.REDSHIFT + "_kappa_g", self.N_g),
                (P.REDSHIFT + "_kappa_pl", self.N_pl),
                (P.REDSHIFT + "_z_max_g", self.N_g),
                (P.REDSHIFT + "_z_max_pl", self.N_pl),
            ])

        if self.use_madau_dickinson_redshift:
            all_params.extend([
                (P.REDSHIFT + "_gamma_g", self.N_g),
                (P.REDSHIFT + "_gamma_pl", self.N_pl),
                (P.REDSHIFT + "_kappa_g", self.N_g),
                (P.REDSHIFT + "_kappa_pl", self.N_pl),
                (P.REDSHIFT + "_z_max_g", self.N_g),
                (P.REDSHIFT + "_z_max_pl", self.N_pl),
                (P.REDSHIFT + "_z_peak_g", self.N_g),
                (P.REDSHIFT + "_z_peak_pl", self.N_pl),
            ])

        if self.use_right_ascension:
            all_params.extend([
                (P.RIGHT_ASCENSION + "_high_g", self.N_g),
                (P.RIGHT_ASCENSION + "_high_pl", self.N_pl),
                (P.RIGHT_ASCENSION + "_low_g", self.N_g),
                (P.RIGHT_ASCENSION + "_low_pl", self.N_pl),
            ])

        if self.use_sin_declination:
            all_params.extend([
                (P.SIN_DECLINATION + "_high_g", self.N_g),
                (P.SIN_DECLINATION + "_high_pl", self.N_pl),
                (P.SIN_DECLINATION + "_low_g", self.N_g),
                (P.SIN_DECLINATION + "_low_pl", self.N_pl),
            ])

        if self.use_detection_time:
            all_params.extend([
                (P.DETECTION_TIME + "_high_g", self.N_g),
                (P.DETECTION_TIME + "_high_pl", self.N_pl),
                (P.DETECTION_TIME + "_low_g", self.N_g),
                (P.DETECTION_TIME + "_low_pl", self.N_pl),
            ])

        if self.use_cos_iota:
            all_params.extend([
                (P.COS_IOTA + "_high_g", self.N_g),
                (P.COS_IOTA + "_high_pl", self.N_pl),
                (P.COS_IOTA + "_low_g", self.N_g),
                (P.COS_IOTA + "_low_pl", self.N_pl),
            ])

        if self.use_polarization_angle:
            all_params.extend([
                (P.POLARIZATION_ANGLE + "_high_g", self.N_g),
                (P.POLARIZATION_ANGLE + "_high_pl", self.N_pl),
                (P.POLARIZATION_ANGLE + "_low_g", self.N_g),
                (P.POLARIZATION_ANGLE + "_low_pl", self.N_pl),
            ])
        if self.use_phi_orb:
            all_params.extend([
                (P.PHI_ORB + "_high_g", self.N_g),
                (P.PHI_ORB + "_high_pl", self.N_pl),
                (P.PHI_ORB + "_low_g", self.N_g),
                (P.PHI_ORB + "_low_pl", self.N_pl),
            ])

        extended_params = []
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

    model_group.add_argument(
        "--add-phi-1",
        action="store_true",
        help=(
            "Include the azimuthal angle of the primary spin component (phi_1) in the model. "
            "Modeled via a Uniform distribution between specified '_low_' and '_high_' bounds, "
            "typically capturing random orientation in the horizontal plane over [0, 2*pi]."
        ),
    )
    model_group.add_argument(
        "--add-phi-2",
        action="store_true",
        help=(
            "Include the azimuthal angle of the secondary spin component (phi_2) in the model. "
            "Modeled via a Uniform distribution between specified '_low_' and '_high_' bounds, "
            "typically capturing random orientation in the horizontal plane over [0, 2*pi]."
        ),
    )
    model_group.add_argument(
        "--add-phi-12",
        action="store_true",
        help=(
            "Include the relative azimuthal angle between the spin vectors of the two component "
            "compact objects (phi_12) in the model. Modeled via a Uniform distribution between "
            "'_low_' and '_high_' parameters to capture spin-precession orientations."
        ),
    )

    model_group.add_argument(
        "--add-right-ascension",
        action="store_true",
        help=(
            "Model the celestial right ascension coordinate of the gravitational wave source "
            "using a Uniform distribution. Tracks population bounding constraints ('_low_', '_high_') "
            "across the celestial sphere."
        ),
    )
    model_group.add_argument(
        "--add-sin-declination",
        action="store_true",
        help=(
            "Model the sine of the celestial declination coordinate of the source. Distributed "
            "uniformly between '_low_' and '_high_' bounds (ideally [-1, 1]) to enforce an "
            "isotropic spatial distribution across the sky."
        ),
    )
    model_group.add_argument(
        "--add-cos-iota",
        action="store_true",
        help=(
            "Model the cosine of the total orbital inclination angle (cos_iota), which measures "
            "the orientation of the binary's orbital angular momentum relative to the line of sight. "
            "Distributed uniformly between '_low_' and '_high_' bounds to simulate random spatial orientation."
        ),
    )
    model_group.add_argument(
        "--add-polarization-angle",
        action="store_true",
        help=(
            "Include the gravitational-wave polarization angle in the model. Distributed uniformly "
            "between specified '_low_' and '_high_' boundary parameters to capture the relative wave "
            "frame rotation relative to a standard celestial framework."
        ),
    )
    model_group.add_argument(
        "--add-phi-orb",
        action="store_true",
        help=(
            "Include the orbital phase of the binary system at a designated reference epoch (phi_orb). "
            "Modeled as a Uniform distribution over specified low and high parameter bounds."
        ),
    )
    model_group.add_argument(
        "--add-detection-time",
        action="store_true",
        help=(
            "Include the geocentric arrival time of the merger event in the model. Distributed uniformly "
            "between specific temporal boundaries ('_low_' and '_high_') to establish flat rate "
            "priors over the observation run runtime window."
        ),
    )

    return parser
