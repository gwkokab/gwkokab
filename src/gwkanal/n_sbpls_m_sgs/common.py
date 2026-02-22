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
            N_pl: int = kwargs.get("N_sbpl")  # type: ignore
            N_g: int = kwargs.get("N_sgpl")  # type: ignore
            mask = jnp.ones((), dtype=bool)
            for n_pl in range(N_pl):
                chi_mean: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE + "_mean_sbpl_" + str(n_pl)
                )  # type: ignore
                chi_variance: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE + "_variance_sbpl_" + str(n_pl)
                )  # type: ignore
                mask &= check_min_concentration_for_beta_dist(chi_mean, chi_variance)
            for n_g in range(N_g):
                chi_mean: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE + "_mean_sgpl_" + str(n_g)
                )  # type: ignore
                chi_variance: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE + "_variance_sg_" + str(n_g)
                )  # type: ignore
                mask &= check_min_concentration_for_beta_dist(chi_mean, chi_variance)
            return mask

        where_fns.append(positive_concentration)

    return where_fns if len(where_fns) > 0 else None


class NSmoothedBrokenPowerlawMSmoothedGaussianCore:
    def __init__(
        self,
        N_sbpl: int,
        N_sgpl: int,
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
        use_redshift: bool,
        use_cos_iota: bool,
        use_phi_12: bool,
        use_polarization_angle: bool,
        use_right_ascension: bool,
        use_sin_declination: bool,
        use_detection_time: bool,
    ) -> None:
        self.N_sbpl = N_sbpl
        self.N_sgpl = N_sgpl
        self.N_gg = N_gg
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
        self.use_redshift = use_redshift
        self.use_cos_iota = use_cos_iota
        self.use_phi_12 = use_phi_12
        self.use_polarization_angle = use_polarization_angle
        self.use_right_ascension = use_right_ascension
        self.use_sin_declination = use_sin_declination
        self.use_detection_time = use_detection_time

    def modify_model_params(self, params: dict) -> dict:
        params.update(
            {
                "N_sbpl": self.N_sbpl,
                "N_sgpl": self.N_sgpl,
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
                "use_redshift": self.use_redshift,
                "use_cos_iota": self.use_cos_iota,
                "use_phi_12": self.use_phi_12,
                "use_polarization_angle": self.use_polarization_angle,
                "use_right_ascension": self.use_right_ascension,
                "use_sin_declination": self.use_sin_declination,
                "use_detection_time": self.use_detection_time,
            }
        )
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
        if self.use_phi_12:
            names.append(P.PHI_12)
        if self.use_eccentricity_mixture:
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
        return names

    @property
    def model_parameters(self) -> list[str]:
        all_params: list[tuple[str, int]] = [
            ("log_rate", self.N_sbpl + self.N_sgpl + self.N_gg),
            ("alpha1_sbpl", self.N_sbpl),
            ("alpha2_sbpl", self.N_sbpl),
            ("beta_sbpl", self.N_sbpl),
            ("beta_sgpl", self.N_sgpl),
            ("delta_m1_sbpl", self.N_sbpl),
            ("delta_m1_sgpl", self.N_sgpl),
            ("delta_m2_sbpl", self.N_sbpl),
            ("delta_m2_sgpl", self.N_sgpl),
            ("loc_sgpl", self.N_sgpl),
            ("m1_high_gg", self.N_gg),
            ("m1_loc_gg", self.N_gg),
            ("m1_low_gg", self.N_gg),
            ("m1_scale_gg", self.N_gg),
            ("m1min_sbpl", self.N_sbpl),
            ("m1min_sgpl", self.N_sgpl),
            ("m2_high_gg", self.N_gg),
            ("m2_loc_gg", self.N_gg),
            ("m2_low_gg", self.N_gg),
            ("m2_scale_gg", self.N_gg),
            ("m2min_sbpl", self.N_sbpl),
            ("m2min_sgpl", self.N_sgpl),
            ("mbreak_sbpl", self.N_sbpl),
            ("mmax_sbpl", self.N_sbpl),
            ("mmax_sgpl", self.N_sgpl),
            ("scale_sgpl", self.N_sgpl),
        ]

        if self.use_spin_magnitude_mixture:
            all_params.extend(
                [
                    ("a_zeta_gg", self.N_gg),
                    ("a_zeta_sbpl", self.N_sbpl),
                    ("a_zeta_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_beta_spin_magnitude:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_MAGNITUDE + "_mean_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_mean_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_mean_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_variance_gg", self.N_gg),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_variance_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_MAGNITUDE + "_variance_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_mean_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_mean_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_mean_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_variance_gg", self.N_gg),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_variance_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_MAGNITUDE + "_variance_sgpl", self.N_sgpl),
                ]
            )

        if self.use_truncated_normal_spin_x:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_X + "_high_gg", self.N_gg),
                    (P.PRIMARY_SPIN_X + "_high_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_X + "_high_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_X + "_loc_gg", self.N_gg),
                    (P.PRIMARY_SPIN_X + "_loc_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_X + "_loc_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_X + "_low_gg", self.N_gg),
                    (P.PRIMARY_SPIN_X + "_low_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_X + "_low_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_X + "_scale_gg", self.N_gg),
                    (P.PRIMARY_SPIN_X + "_scale_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_X + "_scale_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_X + "_high_gg", self.N_gg),
                    (P.SECONDARY_SPIN_X + "_high_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_X + "_high_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_X + "_loc_gg", self.N_gg),
                    (P.SECONDARY_SPIN_X + "_loc_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_X + "_loc_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_X + "_low_gg", self.N_gg),
                    (P.SECONDARY_SPIN_X + "_low_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_X + "_low_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_X + "_scale_gg", self.N_gg),
                    (P.SECONDARY_SPIN_X + "_scale_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_X + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_truncated_normal_spin_y:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_Y + "_high_gg", self.N_gg),
                    (P.PRIMARY_SPIN_Y + "_high_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_Y + "_high_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_Y + "_loc_gg", self.N_gg),
                    (P.PRIMARY_SPIN_Y + "_loc_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_Y + "_loc_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_Y + "_low_gg", self.N_gg),
                    (P.PRIMARY_SPIN_Y + "_low_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_Y + "_low_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_Y + "_scale_gg", self.N_gg),
                    (P.PRIMARY_SPIN_Y + "_scale_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_Y + "_scale_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_Y + "_high_gg", self.N_gg),
                    (P.SECONDARY_SPIN_Y + "_high_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_Y + "_high_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_Y + "_loc_gg", self.N_gg),
                    (P.SECONDARY_SPIN_Y + "_loc_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_Y + "_loc_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_Y + "_low_gg", self.N_gg),
                    (P.SECONDARY_SPIN_Y + "_low_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_Y + "_low_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_Y + "_scale_gg", self.N_gg),
                    (P.SECONDARY_SPIN_Y + "_scale_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_Y + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_truncated_normal_spin_z:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_Z + "_high_gg", self.N_gg),
                    (P.PRIMARY_SPIN_Z + "_high_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_Z + "_high_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_Z + "_loc_gg", self.N_gg),
                    (P.PRIMARY_SPIN_Z + "_loc_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_Z + "_loc_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_Z + "_low_gg", self.N_gg),
                    (P.PRIMARY_SPIN_Z + "_low_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_Z + "_low_sgpl", self.N_sgpl),
                    (P.PRIMARY_SPIN_Z + "_scale_gg", self.N_gg),
                    (P.PRIMARY_SPIN_Z + "_scale_sbpl", self.N_sbpl),
                    (P.PRIMARY_SPIN_Z + "_scale_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_Z + "_high_gg", self.N_gg),
                    (P.SECONDARY_SPIN_Z + "_high_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_Z + "_high_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_Z + "_loc_gg", self.N_gg),
                    (P.SECONDARY_SPIN_Z + "_loc_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_Z + "_loc_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_Z + "_low_gg", self.N_gg),
                    (P.SECONDARY_SPIN_Z + "_low_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_Z + "_low_sgpl", self.N_sgpl),
                    (P.SECONDARY_SPIN_Z + "_scale_gg", self.N_gg),
                    (P.SECONDARY_SPIN_Z + "_scale_sbpl", self.N_sbpl),
                    (P.SECONDARY_SPIN_Z + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_chi_eff_mixture:
            all_params.extend(
                [
                    (P.EFFECTIVE_SPIN + "_comp1_high_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_comp1_high_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_comp1_high_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_comp1_loc_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_comp1_loc_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_comp1_loc_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_comp1_low_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_comp1_low_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_comp1_low_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_comp1_scale_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_comp1_scale_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_comp1_scale_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_comp2_high_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_comp2_high_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_comp2_high_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_comp2_loc_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_comp2_loc_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_comp2_loc_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_comp2_low_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_comp2_low_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_comp2_low_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_comp2_scale_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_comp2_scale_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_comp2_scale_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_zeta_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_zeta_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_zeta_sgpl", self.N_sgpl),
                ]
            )

        if self.use_skew_normal_chi_eff:
            all_params.extend(
                [
                    (P.EFFECTIVE_SPIN + "_epsilon_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_epsilon_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_epsilon_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_loc_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_loc_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_loc_sgpl", self.N_sgpl),
                    (P.EFFECTIVE_SPIN + "_scale_gg", self.N_gg),
                    (P.EFFECTIVE_SPIN + "_scale_sbpl", self.N_sbpl),
                    (P.EFFECTIVE_SPIN + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_truncated_normal_chi_p:
            all_params.extend(
                [
                    (P.PRECESSING_SPIN + "_high_gg", self.N_gg),
                    (P.PRECESSING_SPIN + "_high_sbpl", self.N_sbpl),
                    (P.PRECESSING_SPIN + "_high_sgpl", self.N_sgpl),
                    (P.PRECESSING_SPIN + "_loc_gg", self.N_gg),
                    (P.PRECESSING_SPIN + "_loc_sbpl", self.N_sbpl),
                    (P.PRECESSING_SPIN + "_loc_sgpl", self.N_sgpl),
                    (P.PRECESSING_SPIN + "_low_gg", self.N_gg),
                    (P.PRECESSING_SPIN + "_low_sbpl", self.N_sbpl),
                    (P.PRECESSING_SPIN + "_low_sgpl", self.N_sgpl),
                    (P.PRECESSING_SPIN + "_scale_gg", self.N_gg),
                    (P.PRECESSING_SPIN + "_scale_sbpl", self.N_sbpl),
                    (P.PRECESSING_SPIN + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_tilt:
            all_params.extend(
                [
                    ("cos_tilt_zeta_gg", self.N_gg),
                    ("cos_tilt_zeta_sbpl", self.N_sbpl),
                    ("cos_tilt_zeta_sgpl", self.N_sgpl),
                    (P.COS_TILT_1 + "_scale_gg", self.N_gg),
                    (P.COS_TILT_1 + "_scale_sbpl", self.N_sbpl),
                    (P.COS_TILT_1 + "_scale_sgpl", self.N_sgpl),
                    (P.COS_TILT_2 + "_scale_gg", self.N_gg),
                    (P.COS_TILT_2 + "_scale_sbpl", self.N_sbpl),
                    (P.COS_TILT_2 + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_phi_12:
            all_params.extend(
                [
                    (P.PHI_12 + "_high_gg", self.N_gg),
                    (P.PHI_12 + "_high_sbpl", self.N_sbpl),
                    (P.PHI_12 + "_high_sgpl", self.N_sgpl),
                    (P.PHI_12 + "_loc_gg", self.N_gg),
                    (P.PHI_12 + "_loc_sbpl", self.N_sbpl),
                    (P.PHI_12 + "_loc_sgpl", self.N_sgpl),
                    (P.PHI_12 + "_low_gg", self.N_gg),
                    (P.PHI_12 + "_low_sbpl", self.N_sbpl),
                    (P.PHI_12 + "_low_sgpl", self.N_sgpl),
                    (P.PHI_12 + "_scale_gg", self.N_gg),
                    (P.PHI_12 + "_scale_sbpl", self.N_sbpl),
                    (P.PHI_12 + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_eccentricity_mixture:
            all_params.extend(
                [
                    (P.ECCENTRICITY + "_comp1_high_gg", self.N_gg),
                    (P.ECCENTRICITY + "_comp1_high_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_comp1_high_sgpl", self.N_sgpl),
                    (P.ECCENTRICITY + "_comp1_loc_gg", self.N_gg),
                    (P.ECCENTRICITY + "_comp1_loc_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_comp1_loc_sgpl", self.N_sgpl),
                    (P.ECCENTRICITY + "_comp1_low_gg", self.N_gg),
                    (P.ECCENTRICITY + "_comp1_low_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_comp1_low_sgpl", self.N_sgpl),
                    (P.ECCENTRICITY + "_comp1_scale_gg", self.N_gg),
                    (P.ECCENTRICITY + "_comp1_scale_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_comp1_scale_sgpl", self.N_sgpl),
                    (P.ECCENTRICITY + "_comp2_high_gg", self.N_gg),
                    (P.ECCENTRICITY + "_comp2_high_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_comp2_high_sgpl", self.N_sgpl),
                    (P.ECCENTRICITY + "_comp2_loc_gg", self.N_gg),
                    (P.ECCENTRICITY + "_comp2_loc_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_comp2_loc_sgpl", self.N_sgpl),
                    (P.ECCENTRICITY + "_comp2_low_gg", self.N_gg),
                    (P.ECCENTRICITY + "_comp2_low_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_comp2_low_sgpl", self.N_sgpl),
                    (P.ECCENTRICITY + "_comp2_scale_gg", self.N_gg),
                    (P.ECCENTRICITY + "_comp2_scale_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_comp2_scale_sgpl", self.N_sgpl),
                    (P.ECCENTRICITY + "_zeta_gg", self.N_gg),
                    (P.ECCENTRICITY + "_zeta_sbpl", self.N_sbpl),
                    (P.ECCENTRICITY + "_zeta_sgpl", self.N_sgpl),
                ]
            )

        if self.use_redshift:
            all_params.extend(
                [
                    (P.REDSHIFT + "_kappa_gg", self.N_gg),
                    (P.REDSHIFT + "_kappa_sbpl", self.N_sbpl),
                    (P.REDSHIFT + "_kappa_sgpl", self.N_sgpl),
                    (P.REDSHIFT + "_z_max_gg", self.N_gg),
                    (P.REDSHIFT + "_z_max_sbpl", self.N_sbpl),
                    (P.REDSHIFT + "_z_max_sgpl", self.N_sgpl),
                ]
            )

        if self.use_right_ascension:
            all_params.extend(
                [
                    (P.RIGHT_ASCENSION + "_high_gg", self.N_gg),
                    (P.RIGHT_ASCENSION + "_high_sbpl", self.N_sbpl),
                    (P.RIGHT_ASCENSION + "_high_sgpl", self.N_sgpl),
                    (P.RIGHT_ASCENSION + "_loc_gg", self.N_gg),
                    (P.RIGHT_ASCENSION + "_loc_sbpl", self.N_sbpl),
                    (P.RIGHT_ASCENSION + "_loc_sgpl", self.N_sgpl),
                    (P.RIGHT_ASCENSION + "_low_gg", self.N_gg),
                    (P.RIGHT_ASCENSION + "_low_sbpl", self.N_sbpl),
                    (P.RIGHT_ASCENSION + "_low_sgpl", self.N_sgpl),
                    (P.RIGHT_ASCENSION + "_scale_gg", self.N_gg),
                    (P.RIGHT_ASCENSION + "_scale_sbpl", self.N_sbpl),
                    (P.RIGHT_ASCENSION + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_sin_declination:
            all_params.extend(
                [
                    (P.SIN_DECLINATION + "_high_gg", self.N_gg),
                    (P.SIN_DECLINATION + "_high_sbpl", self.N_sbpl),
                    (P.SIN_DECLINATION + "_high_sgpl", self.N_sgpl),
                    (P.SIN_DECLINATION + "_loc_gg", self.N_gg),
                    (P.SIN_DECLINATION + "_loc_sbpl", self.N_sbpl),
                    (P.SIN_DECLINATION + "_loc_sgpl", self.N_sgpl),
                    (P.SIN_DECLINATION + "_low_gg", self.N_gg),
                    (P.SIN_DECLINATION + "_low_sbpl", self.N_sbpl),
                    (P.SIN_DECLINATION + "_low_sgpl", self.N_sgpl),
                    (P.SIN_DECLINATION + "_scale_gg", self.N_gg),
                    (P.SIN_DECLINATION + "_scale_sbpl", self.N_sbpl),
                    (P.SIN_DECLINATION + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_detection_time:
            all_params.extend(
                [
                    (P.DETECTION_TIME + "_high_gg", self.N_gg),
                    (P.DETECTION_TIME + "_high_sbpl", self.N_sbpl),
                    (P.DETECTION_TIME + "_high_sgpl", self.N_sgpl),
                    (P.DETECTION_TIME + "_low_gg", self.N_gg),
                    (P.DETECTION_TIME + "_low_sbpl", self.N_sbpl),
                    (P.DETECTION_TIME + "_low_sgpl", self.N_sgpl),
                ]
            )

        if self.use_cos_iota:
            all_params.extend(
                [
                    (P.COS_IOTA + "_high_gg", self.N_gg),
                    (P.COS_IOTA + "_high_sbpl", self.N_sbpl),
                    (P.COS_IOTA + "_high_sgpl", self.N_sgpl),
                    (P.COS_IOTA + "_loc_gg", self.N_gg),
                    (P.COS_IOTA + "_loc_sbpl", self.N_sbpl),
                    (P.COS_IOTA + "_loc_sgpl", self.N_sgpl),
                    (P.COS_IOTA + "_low_gg", self.N_gg),
                    (P.COS_IOTA + "_low_sbpl", self.N_sbpl),
                    (P.COS_IOTA + "_low_sgpl", self.N_sgpl),
                    (P.COS_IOTA + "_scale_gg", self.N_gg),
                    (P.COS_IOTA + "_scale_sbpl", self.N_sbpl),
                    (P.COS_IOTA + "_scale_sgpl", self.N_sgpl),
                ]
            )

        if self.use_polarization_angle:
            all_params.extend(
                [
                    (P.POLARIZATION_ANGLE + "_high_gg", self.N_gg),
                    (P.POLARIZATION_ANGLE + "_high_sbpl", self.N_sbpl),
                    (P.POLARIZATION_ANGLE + "_high_sgpl", self.N_sgpl),
                    (P.POLARIZATION_ANGLE + "_loc_gg", self.N_gg),
                    (P.POLARIZATION_ANGLE + "_loc_sbpl", self.N_sbpl),
                    (P.POLARIZATION_ANGLE + "_loc_sgpl", self.N_sgpl),
                    (P.POLARIZATION_ANGLE + "_low_gg", self.N_gg),
                    (P.POLARIZATION_ANGLE + "_low_sbpl", self.N_sbpl),
                    (P.POLARIZATION_ANGLE + "_low_sgpl", self.N_sgpl),
                    (P.POLARIZATION_ANGLE + "_scale_gg", self.N_gg),
                    (P.POLARIZATION_ANGLE + "_scale_sbpl", self.N_sbpl),
                    (P.POLARIZATION_ANGLE + "_scale_sgpl", self.N_sgpl),
                ]
            )

        extended_params = []
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--n-sbpl",
        type=int,
        default=0,
        help="Number of smoothed broken power law components in the mass model.",
    )
    model_group.add_argument(
        "--n-sgpl",
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
    model_group.add_argument(
        "--add-redshift",
        action="store_true",
        help="Include redshift parameter in the model",
    )
    model_group.add_argument(
        "--add-eccentricity-mixture",
        action="store_true",
        help="Include truncated normal eccentricity in the model.",
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
    return parser
