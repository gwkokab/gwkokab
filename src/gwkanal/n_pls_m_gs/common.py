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

        if self.use_truncated_normal_spin_x:
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

        if self.use_truncated_normal_spin_y:
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

        if self.use_truncated_normal_spin_z:
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

        if self.use_tilt:
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

        if self.use_phi_1:
            all_params.extend(
                [
                    (P.PHI_1 + "_high_g", self.N_g),
                    (P.PHI_1 + "_high_pl", self.N_pl),
                    (P.PHI_1 + "_low_g", self.N_g),
                    (P.PHI_1 + "_low_pl", self.N_pl),
                ]
            )

        if self.use_phi_2:
            all_params.extend(
                [
                    (P.PHI_2 + "_high_g", self.N_g),
                    (P.PHI_2 + "_high_pl", self.N_pl),
                    (P.PHI_2 + "_low_g", self.N_g),
                    (P.PHI_2 + "_low_pl", self.N_pl),
                ]
            )

        if self.use_phi_12:
            all_params.extend(
                [
                    (P.PHI_12 + "_high_g", self.N_g),
                    (P.PHI_12 + "_high_pl", self.N_pl),
                    (P.PHI_12 + "_low_g", self.N_g),
                    (P.PHI_12 + "_low_pl", self.N_pl),
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

        if self.use_eccentricity_powerlaw:
            all_params.extend(
                [
                    (P.ECCENTRICITY + "_alpha_g", self.N_g),
                    (P.ECCENTRICITY + "_alpha_pl", self.N_pl),
                    (P.ECCENTRICITY + "_high_g", self.N_g),
                    (P.ECCENTRICITY + "_high_pl", self.N_pl),
                    (P.ECCENTRICITY + "_low_g", self.N_g),
                    (P.ECCENTRICITY + "_low_pl", self.N_pl),
                ]
            )

        if self.use_mean_anomaly:
            all_params.extend(
                [
                    (P.MEAN_ANOMALY + "_high_g", self.N_g),
                    (P.MEAN_ANOMALY + "_high_pl", self.N_pl),
                    (P.MEAN_ANOMALY + "_low_g", self.N_g),
                    (P.MEAN_ANOMALY + "_low_pl", self.N_pl),
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

        if self.use_right_ascension:
            all_params.extend(
                [
                    (P.RIGHT_ASCENSION + "_high_g", self.N_g),
                    (P.RIGHT_ASCENSION + "_high_pl", self.N_pl),
                    (P.RIGHT_ASCENSION + "_low_g", self.N_g),
                    (P.RIGHT_ASCENSION + "_low_pl", self.N_pl),
                ]
            )

        if self.use_sin_declination:
            all_params.extend(
                [
                    (P.SIN_DECLINATION + "_high_g", self.N_g),
                    (P.SIN_DECLINATION + "_high_pl", self.N_pl),
                    (P.SIN_DECLINATION + "_low_g", self.N_g),
                    (P.SIN_DECLINATION + "_low_pl", self.N_pl),
                ]
            )

        if self.use_detection_time:
            all_params.extend(
                [
                    (P.DETECTION_TIME + "_high_g", self.N_g),
                    (P.DETECTION_TIME + "_high_pl", self.N_pl),
                    (P.DETECTION_TIME + "_low_g", self.N_g),
                    (P.DETECTION_TIME + "_low_pl", self.N_pl),
                ]
            )

        if self.use_cos_iota:
            all_params.extend(
                [
                    (P.COS_IOTA + "_high_g", self.N_g),
                    (P.COS_IOTA + "_high_pl", self.N_pl),
                    (P.COS_IOTA + "_low_g", self.N_g),
                    (P.COS_IOTA + "_low_pl", self.N_pl),
                ]
            )

        if self.use_polarization_angle:
            all_params.extend(
                [
                    (P.POLARIZATION_ANGLE + "_high_g", self.N_g),
                    (P.POLARIZATION_ANGLE + "_high_pl", self.N_pl),
                    (P.POLARIZATION_ANGLE + "_low_g", self.N_g),
                    (P.POLARIZATION_ANGLE + "_low_pl", self.N_pl),
                ]
            )
        if self.use_phi_orb:
            all_params.extend(
                [
                    (P.PHI_ORB + "_high_g", self.N_g),
                    (P.PHI_ORB + "_high_pl", self.N_pl),
                    (P.PHI_ORB + "_low_g", self.N_g),
                    (P.PHI_ORB + "_low_pl", self.N_pl),
                ]
            )

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
