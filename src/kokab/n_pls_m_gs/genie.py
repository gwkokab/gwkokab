# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from typing import List, Tuple

from jax import numpy as jnp, random as jrd
from loguru import logger
from numpyro import distributions as dist

from gwkokab.errors import banana_error_m1_m2, truncated_normal_error
from gwkokab.models import NPowerlawMGaussian
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean import get_selection_fn_and_poisson_mean_estimator
from kokab.core.population import error_magazine, PopulationFactory
from kokab.utils import genie_parser
from kokab.utils.common import expand_arguments, read_json
from kokab.utils.logger import log_info
from kokab.utils.regex import match_all


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = genie_parser.get_parser(parser)

    model_group = parser.add_argument_group("Model Options")

    model_group.add_argument(
        "--model-json",
        help="Path to the JSON file containing the model parameters",
        type=str,
        required=True,
    )

    spin_magnitude_group = model_group.add_mutually_exclusive_group()
    spin_magnitude_group.add_argument(
        "--add-beta-spin-magnitude",
        action="store_true",
        help="Include beta-distributed spin magnitudes a1,a2 (dimensionless Kerr spins; 0≤a<1).",
    )
    spin_magnitude_group.add_argument(
        "--add-spin-magnitude-mixture",
        action="store_true",
        help="Include truncated-normal spin magnitudes a1,a2 (dimensionless Kerr spins; 0≤a<1).",
    )

    model_group.add_argument(
        "--add-truncated-normal-spin-x",
        action="store_true",
        help="Include truncated-normal spin x components.",
    )
    model_group.add_argument(
        "--add-truncated-normal-spin-y",
        action="store_true",
        help="Include truncated-normal spin y components.",
    )
    model_group.add_argument(
        "--add-truncated-normal-spin-z",
        action="store_true",
        help="Include truncated-normal spin z components.",
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
        help="Include truncated-normal chi_p components.",
    )
    model_group.add_argument(
        "--add-tilt",
        action="store_true",
        help="Include spin-orbit tilt cosines cos_tilt1, cos_tilt2 (cosines of angles between each spin and orbital angular momentum; physical range -1 to 1).",
    )
    model_group.add_argument(
        "--add-eccentricity-mixture",
        action="store_true",
        help="Include orbital eccentricity e via a truncated-normal prior (dimensionless; physical range 0≤e<1 at the reference frequency/time).",
    )
    model_group.add_argument(
        "--add-mean-anomaly",
        action="store_true",
        help="Include mean anomaly M (Kepler mean orbital phase; radians; domain [0,2π)).",
    )
    model_group.add_argument(
        "--add-redshift",
        action="store_true",
        help="Include cosmological redshift z (dimensionless).",
    )
    model_group.add_argument(
        "--add-cos-iota",
        action="store_true",
        help="Include inclination cosine cos(iota) (cosine of angle between orbital angular momentum and line of sight; isotropy ⇒ Uniform[-1,1]).",
    )
    model_group.add_argument(
        "--add-polarization-angle",
        action="store_true",
        help="Include GW polarization angle ψ (radians; domain [0,π); period π).",
    )
    model_group.add_argument(
        "--add-right-ascension",
        action="store_true",
        help="Include right ascension RA (sky longitude; radians; domain [0,2π); uniform for isotropic sky).",
    )
    model_group.add_argument(
        "--add-sin-declination",
        action="store_true",
        help="Include sin(declination) (store sine of declination to ensure uniform sky prior; domain [-1,1]).",
    )
    model_group.add_argument(
        "--add-detection-time",
        action="store_true",
        help="Include geocentric coalescence time (detection_time); typically seconds or GPS time; choose your window in model JSON.",
    )
    model_group.add_argument(
        "--add-phi-1",
        action="store_true",
        help="Include spin azimuth φ₁ (radians; domain [0,2π); affects precession).",
    )
    model_group.add_argument(
        "--add-phi-2",
        action="store_true",
        help="Include spin azimuth φ₂ (radians; domain [0,2π); affects precession).",
    )
    model_group.add_argument(
        "--add-phi-12",
        action="store_true",
        help="Include relative spin azimuth φ₁₂ (radians; domain [0,2π); precession-sensitive).",
    )
    model_group.add_argument(
        "--add-phi-orb",
        action="store_true",
        help="Include orbital phase φ_orb at the reference time/frequency (radians; domain [0,2π)).",
    )

    err_group = parser.add_argument_group("Error Options")
    err_group.add_argument(
        "--err-json",
        help="Path to error JSON defining Gaussian/truncated-normal error scales and bounds (periodic angles are wrapped mod π/2π).",
        type=str,
        required=True,
    )

    return parser


def main() -> None:
    """Main function of the script."""
    parser = make_parser()
    args = parser.parse_args()

    log_info(start=True)

    with open(args.model_json, "r") as f:
        model_json = json.load(f)

    with open(args.err_json, "r") as f:
        err_json = json.load(f)

    N_pl = model_json["N_pl"]
    N_g = model_json["N_g"]

    has_beta_spin_magnitude = args.add_beta_spin_magnitude
    has_spin_magnitude_mixture = args.add_spin_magnitude_mixture
    has_truncated_normal_spin_x = args.add_truncated_normal_spin_x
    has_truncated_normal_spin_y = args.add_truncated_normal_spin_y
    has_truncated_normal_spin_z = args.add_truncated_normal_spin_z
    has_chi_eff_mixture = args.add_chi_eff_mixture
    use_skew_normal_chi_eff = args.add_skew_normal_chi_eff
    has_truncated_normal_chi_p = args.add_truncated_normal_chi_p
    has_tilt = args.add_tilt
    use_eccentricity_mixture = args.add_eccentricity_mixture
    has_mean_anomaly = args.add_mean_anomaly
    has_redshift = args.add_redshift
    has_cos_iota = args.add_cos_iota
    has_polarization_angle = args.add_polarization_angle
    has_right_ascension = args.add_right_ascension
    has_sin_declination = args.add_sin_declination
    has_detection_time = args.add_detection_time
    has_phi_1 = args.add_phi_1
    has_phi_2 = args.add_phi_2
    has_phi_12 = args.add_phi_12
    has_phi_orb = args.add_phi_orb

    err_params_name = ["scale_eta", "scale_Mc"]
    if has_beta_spin_magnitude or has_spin_magnitude_mixture:
        err_params_name.extend(
            [
                P.PRIMARY_SPIN_MAGNITUDE + "_high",
                P.PRIMARY_SPIN_MAGNITUDE + "_low",
                P.PRIMARY_SPIN_MAGNITUDE + "_scale",
                P.SECONDARY_SPIN_MAGNITUDE + "_high",
                P.SECONDARY_SPIN_MAGNITUDE + "_low",
                P.SECONDARY_SPIN_MAGNITUDE + "_scale",
            ]
        )
    if has_truncated_normal_spin_x:
        err_params_name.extend(
            [
                P.PRIMARY_SPIN_X + "_high",
                P.PRIMARY_SPIN_X + "_low",
                P.PRIMARY_SPIN_X + "_scale",
                P.SECONDARY_SPIN_X + "_high",
                P.SECONDARY_SPIN_X + "_low",
                P.SECONDARY_SPIN_X + "_scale",
            ]
        )
    if has_truncated_normal_spin_y:
        err_params_name.extend(
            [
                P.PRIMARY_SPIN_Y + "_high",
                P.PRIMARY_SPIN_Y + "_low",
                P.PRIMARY_SPIN_Y + "_scale",
                P.SECONDARY_SPIN_Y + "_high",
                P.SECONDARY_SPIN_Y + "_low",
                P.SECONDARY_SPIN_Y + "_scale",
            ]
        )
    if has_truncated_normal_spin_z:
        err_params_name.extend(
            [
                P.PRIMARY_SPIN_Z + "_high",
                P.PRIMARY_SPIN_Z + "_low",
                P.PRIMARY_SPIN_Z + "_scale",
                P.SECONDARY_SPIN_Z + "_high",
                P.SECONDARY_SPIN_Z + "_low",
                P.SECONDARY_SPIN_Z + "_scale",
            ]
        )
    if has_chi_eff_mixture or use_skew_normal_chi_eff:
        err_params_name.extend(
            [
                P.EFFECTIVE_SPIN + "_high",
                P.EFFECTIVE_SPIN + "_low",
                P.EFFECTIVE_SPIN + "_scale",
                P.EFFECTIVE_SPIN + "_cut_low",
                P.EFFECTIVE_SPIN + "_cut_high",
            ]
        )
    if has_truncated_normal_chi_p:
        err_params_name.extend(
            [
                P.PRECESSING_SPIN + "_high",
                P.PRECESSING_SPIN + "_low",
                P.PRECESSING_SPIN + "_scale",
                P.PRECESSING_SPIN + "_cut_low",
                P.PRECESSING_SPIN + "_cut_high",
            ]
        )
    if has_tilt:
        err_params_name.extend(
            [
                P.COS_TILT_1 + "_high",
                P.COS_TILT_1 + "_low",
                P.COS_TILT_1 + "_scale",
                P.COS_TILT_2 + "_high",
                P.COS_TILT_2 + "_low",
                P.COS_TILT_2 + "_scale",
            ]
        )
    if use_eccentricity_mixture:
        err_params_name.extend(
            [
                P.ECCENTRICITY + "_high",
                P.ECCENTRICITY + "_low",
                P.ECCENTRICITY + "_scale",
            ]
        )
    if has_mean_anomaly:
        err_params_name.extend(
            [
                P.MEAN_ANOMALY + "_high",
                P.MEAN_ANOMALY + "_low",
                P.MEAN_ANOMALY + "_scale",
            ]
        )
    if has_redshift:
        err_params_name.extend(
            [
                P.REDSHIFT + "_high",
                P.REDSHIFT + "_low",
                P.REDSHIFT + "_scale",
            ]
        )
    if has_cos_iota:
        err_params_name.extend(
            [
                P.COS_IOTA + "_high",
                P.COS_IOTA + "_low",
                P.COS_IOTA + "_scale",
            ]
        )
    if has_polarization_angle:
        err_params_name.extend(
            [
                P.POLARIZATION_ANGLE + "_high",
                P.POLARIZATION_ANGLE + "_low",
                P.POLARIZATION_ANGLE + "_scale",
            ]
        )
    if has_right_ascension:
        err_params_name.extend(
            [
                P.RIGHT_ASCENSION + "_high",
                P.RIGHT_ASCENSION + "_low",
                P.RIGHT_ASCENSION + "_scale",
            ]
        )
    if has_sin_declination:
        err_params_name.extend(
            [
                P.SIN_DECLINATION + "_high",
                P.SIN_DECLINATION + "_low",
                P.SIN_DECLINATION + "_scale",
            ]
        )
    if has_detection_time:
        err_params_name.append(P.DETECTION_TIME + "_scale")
    if has_phi_1:
        err_params_name.extend(
            [
                P.PHI_1 + "_high",
                P.PHI_1 + "_low",
                P.PHI_1 + "_scale",
            ]
        )
    if has_phi_2:
        err_params_name.extend(
            [
                P.PHI_2 + "_high",
                P.PHI_2 + "_low",
                P.PHI_2 + "_scale",
            ]
        )
    if has_phi_12:
        err_params_name.extend(
            [
                P.PHI_12 + "_high",
                P.PHI_12 + "_low",
                P.PHI_12 + "_scale",
            ]
        )
    if has_phi_orb:
        err_params_name.extend(
            [
                P.PHI_ORB + "_high",
                P.PHI_ORB + "_low",
                P.PHI_ORB + "_scale",
            ]
        )

    err_params_value = match_all(err_params_name, err_json)

    all_params: List[Tuple[str, int]] = [
        ("alpha_pl", N_pl),
        ("beta_pl", N_pl),
        ("log_rate", N_pl + N_g),
        ("loc_g", N_g),
        ("scale_g", N_g),
        ("beta_g", N_g),
        ("m1min_g", N_g),
        ("m2min_g", N_g),
        ("m1max_g", N_g),
        ("m2max_g", N_g),
        ("mmax_pl", N_pl),
        ("mmin_pl", N_pl),
    ]

    parameters_name: Tuple[str, ...] = (
        P.PRIMARY_MASS_SOURCE,
        P.SECONDARY_MASS_SOURCE,
    )

    error_magazine.register(
        (P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE),
        lambda x, size, key: banana_error_m1_m2(
            x,
            size,
            key,
            scale_Mc=err_params_value["scale_Mc"],
            scale_eta=err_params_value["scale_eta"],
        ),
    )

    if has_beta_spin_magnitude:
        parameters_name += (
            P.PRIMARY_SPIN_MAGNITUDE,
            P.SECONDARY_SPIN_MAGNITUDE,
        )
        all_params.extend(
            [
                (P.PRIMARY_SPIN_MAGNITUDE + "_mean_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_mean_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_variance_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_variance_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_mean_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_mean_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_variance_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_variance_pl", N_pl),
            ]
        )

    if has_spin_magnitude_mixture:
        parameters_name += (
            P.PRIMARY_SPIN_MAGNITUDE,
            P.SECONDARY_SPIN_MAGNITUDE,
        )
        all_params.extend(
            [
                ("a_zeta_g", N_g),
                ("a_zeta_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_high_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_loc_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_low_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp1_scale_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_high_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_loc_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_low_pl", N_pl),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_g", N_g),
                (P.PRIMARY_SPIN_MAGNITUDE + "_comp2_scale_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_high_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_loc_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_low_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp1_scale_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_high_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_loc_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_low_pl", N_pl),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_g", N_g),
                (P.SECONDARY_SPIN_MAGNITUDE + "_comp2_scale_pl", N_pl),
            ]
        )

    if has_beta_spin_magnitude or has_spin_magnitude_mixture:
        error_magazine.register(
            P.PRIMARY_SPIN_MAGNITUDE,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.PRIMARY_SPIN_MAGNITUDE + "_scale"],
                low=err_params_value.get(P.PRIMARY_SPIN_MAGNITUDE + "_low"),
                high=err_params_value.get(P.PRIMARY_SPIN_MAGNITUDE + "_high"),
                cut_low=0.0,
                cut_high=1.0,
            ),
        )

        error_magazine.register(
            P.SECONDARY_SPIN_MAGNITUDE,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.SECONDARY_SPIN_MAGNITUDE + "_scale"],
                low=err_params_value.get(P.SECONDARY_SPIN_MAGNITUDE + "_low"),
                high=err_params_value.get(P.SECONDARY_SPIN_MAGNITUDE + "_high"),
                cut_low=0.0,
                cut_high=1.0,
            ),
        )

    if has_truncated_normal_spin_x:
        parameters_name += (P.PRIMARY_SPIN_X, P.SECONDARY_SPIN_X)
        all_params.extend(
            [
                (P.PRIMARY_SPIN_X + "_high_g", N_g),
                (P.PRIMARY_SPIN_X + "_high_pl", N_pl),
                (P.PRIMARY_SPIN_X + "_loc_g", N_g),
                (P.PRIMARY_SPIN_X + "_loc_pl", N_pl),
                (P.PRIMARY_SPIN_X + "_low_g", N_g),
                (P.PRIMARY_SPIN_X + "_low_pl", N_pl),
                (P.PRIMARY_SPIN_X + "_scale_g", N_g),
                (P.PRIMARY_SPIN_X + "_scale_pl", N_pl),
                (P.SECONDARY_SPIN_X + "_high_g", N_g),
                (P.SECONDARY_SPIN_X + "_high_pl", N_pl),
                (P.SECONDARY_SPIN_X + "_loc_g", N_g),
                (P.SECONDARY_SPIN_X + "_loc_pl", N_pl),
                (P.SECONDARY_SPIN_X + "_low_g", N_g),
                (P.SECONDARY_SPIN_X + "_low_pl", N_pl),
                (P.SECONDARY_SPIN_X + "_scale_g", N_g),
                (P.SECONDARY_SPIN_X + "_scale_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.PRIMARY_SPIN_X,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.PRIMARY_SPIN_X + "_scale"],
                low=err_params_value.get(P.PRIMARY_SPIN_X + "_low"),
                high=err_params_value.get(P.PRIMARY_SPIN_X + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

        error_magazine.register(
            P.SECONDARY_SPIN_X,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.SECONDARY_SPIN_X + "_scale"],
                low=err_params_value.get(P.SECONDARY_SPIN_X + "_low"),
                high=err_params_value.get(P.SECONDARY_SPIN_X + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_truncated_normal_spin_y:
        parameters_name += (P.PRIMARY_SPIN_Y, P.SECONDARY_SPIN_Y)
        all_params.extend(
            [
                (P.PRIMARY_SPIN_Y + "_high_g", N_g),
                (P.PRIMARY_SPIN_Y + "_high_pl", N_pl),
                (P.PRIMARY_SPIN_Y + "_loc_g", N_g),
                (P.PRIMARY_SPIN_Y + "_loc_pl", N_pl),
                (P.PRIMARY_SPIN_Y + "_low_g", N_g),
                (P.PRIMARY_SPIN_Y + "_low_pl", N_pl),
                (P.PRIMARY_SPIN_Y + "_scale_g", N_g),
                (P.PRIMARY_SPIN_Y + "_scale_pl", N_pl),
                (P.SECONDARY_SPIN_Y + "_high_g", N_g),
                (P.SECONDARY_SPIN_Y + "_high_pl", N_pl),
                (P.SECONDARY_SPIN_Y + "_loc_g", N_g),
                (P.SECONDARY_SPIN_Y + "_loc_pl", N_pl),
                (P.SECONDARY_SPIN_Y + "_low_g", N_g),
                (P.SECONDARY_SPIN_Y + "_low_pl", N_pl),
                (P.SECONDARY_SPIN_Y + "_scale_g", N_g),
                (P.SECONDARY_SPIN_Y + "_scale_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.PRIMARY_SPIN_Y,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.PRIMARY_SPIN_Y + "_scale"],
                low=err_params_value.get(P.PRIMARY_SPIN_Y + "_low"),
                high=err_params_value.get(P.PRIMARY_SPIN_Y + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

        error_magazine.register(
            P.SECONDARY_SPIN_Y,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.SECONDARY_SPIN_Y + "_scale"],
                low=err_params_value.get(P.SECONDARY_SPIN_Y + "_low"),
                high=err_params_value.get(P.SECONDARY_SPIN_Y + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_truncated_normal_spin_z:
        parameters_name += (P.PRIMARY_SPIN_Z, P.SECONDARY_SPIN_Z)
        all_params.extend(
            [
                (P.PRIMARY_SPIN_Z + "_high_g", N_g),
                (P.PRIMARY_SPIN_Z + "_high_pl", N_pl),
                (P.PRIMARY_SPIN_Z + "_loc_g", N_g),
                (P.PRIMARY_SPIN_Z + "_loc_pl", N_pl),
                (P.PRIMARY_SPIN_Z + "_low_g", N_g),
                (P.PRIMARY_SPIN_Z + "_low_pl", N_pl),
                (P.PRIMARY_SPIN_Z + "_scale_g", N_g),
                (P.PRIMARY_SPIN_Z + "_scale_pl", N_pl),
                (P.SECONDARY_SPIN_Z + "_high_g", N_g),
                (P.SECONDARY_SPIN_Z + "_high_pl", N_pl),
                (P.SECONDARY_SPIN_Z + "_loc_g", N_g),
                (P.SECONDARY_SPIN_Z + "_loc_pl", N_pl),
                (P.SECONDARY_SPIN_Z + "_low_g", N_g),
                (P.SECONDARY_SPIN_Z + "_low_pl", N_pl),
                (P.SECONDARY_SPIN_Z + "_scale_g", N_g),
                (P.SECONDARY_SPIN_Z + "_scale_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.PRIMARY_SPIN_Z,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.PRIMARY_SPIN_Z + "_scale"],
                low=err_params_value.get(P.PRIMARY_SPIN_Z + "_low"),
                high=err_params_value.get(P.PRIMARY_SPIN_Z + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

        error_magazine.register(
            P.SECONDARY_SPIN_Z,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.SECONDARY_SPIN_Z + "_scale"],
                low=err_params_value.get(P.SECONDARY_SPIN_Z + "_low"),
                high=err_params_value.get(P.SECONDARY_SPIN_Z + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_chi_eff_mixture:
        parameters_name += (P.EFFECTIVE_SPIN,)
        all_params.extend(
            [
                (P.EFFECTIVE_SPIN + "_comp1_high_g", N_g),
                (P.EFFECTIVE_SPIN + "_comp1_high_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_comp1_loc_g", N_g),
                (P.EFFECTIVE_SPIN + "_comp1_loc_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_comp1_low_g", N_g),
                (P.EFFECTIVE_SPIN + "_comp1_low_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_comp1_scale_g", N_g),
                (P.EFFECTIVE_SPIN + "_comp1_scale_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_comp2_high_g", N_g),
                (P.EFFECTIVE_SPIN + "_comp2_high_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_comp2_loc_g", N_g),
                (P.EFFECTIVE_SPIN + "_comp2_loc_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_comp2_low_g", N_g),
                (P.EFFECTIVE_SPIN + "_comp2_low_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_comp2_scale_g", N_g),
                (P.EFFECTIVE_SPIN + "_comp2_scale_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_zeta_g", N_g),
                (P.EFFECTIVE_SPIN + "_zeta_pl", N_pl),
            ]
        )
        error_magazine.register(
            P.EFFECTIVE_SPIN,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.EFFECTIVE_SPIN + "_scale"],
                low=err_params_value.get(P.EFFECTIVE_SPIN + "_low"),
                high=err_params_value.get(P.EFFECTIVE_SPIN + "_high"),
                cut_low=err_params_value.get(P.EFFECTIVE_SPIN + "_cut_low"),
                cut_high=err_params_value.get(P.EFFECTIVE_SPIN + "_cut_high"),
            ),
        )
    if use_skew_normal_chi_eff:
        parameters_name += (P.EFFECTIVE_SPIN,)
        all_params.extend(
            [
                (P.EFFECTIVE_SPIN + "_epsilon_g", N_g),
                (P.EFFECTIVE_SPIN + "_epsilon_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_loc_g", N_g),
                (P.EFFECTIVE_SPIN + "_loc_pl", N_pl),
                (P.EFFECTIVE_SPIN + "_scale_g", N_g),
                (P.EFFECTIVE_SPIN + "_scale_pl", N_pl),
            ]
        )
        error_magazine.register(
            P.EFFECTIVE_SPIN,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.EFFECTIVE_SPIN + "_scale"],
                low=err_params_value.get(P.EFFECTIVE_SPIN + "_low"),
                high=err_params_value.get(P.EFFECTIVE_SPIN + "_high"),
                cut_low=err_params_value.get(P.EFFECTIVE_SPIN + "_cut_low"),
                cut_high=err_params_value.get(P.EFFECTIVE_SPIN + "_cut_high"),
            ),
        )

    if has_truncated_normal_chi_p:
        parameters_name += (P.PRECESSING_SPIN,)
        all_params.extend(
            [
                (P.PRECESSING_SPIN + "_high_g", N_g),
                (P.PRECESSING_SPIN + "_high_pl", N_pl),
                (P.PRECESSING_SPIN + "_loc_g", N_g),
                (P.PRECESSING_SPIN + "_loc_pl", N_pl),
                (P.PRECESSING_SPIN + "_low_g", N_g),
                (P.PRECESSING_SPIN + "_low_pl", N_pl),
                (P.PRECESSING_SPIN + "_scale_g", N_g),
                (P.PRECESSING_SPIN + "_scale_pl", N_pl),
            ]
        )
        error_magazine.register(
            P.PRECESSING_SPIN,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.PRECESSING_SPIN + "_scale"],
                low=err_params_value.get(P.PRECESSING_SPIN + "_low"),
                high=err_params_value.get(P.PRECESSING_SPIN + "_high"),
                cut_low=err_params_value.get(P.PRECESSING_SPIN + "_cut_low"),
                cut_high=err_params_value.get(P.PRECESSING_SPIN + "_cut_high"),
            ),
        )

    if has_tilt:
        parameters_name += (P.COS_TILT_1, P.COS_TILT_2)
        all_params.extend(
            [
                ("cos_tilt_zeta_g", N_g),
                ("cos_tilt_zeta_pl", N_pl),
                (P.COS_TILT_1 + "_scale_g", N_g),
                (P.COS_TILT_1 + "_scale_pl", N_pl),
                (P.COS_TILT_2 + "_scale_g", N_g),
                (P.COS_TILT_2 + "_scale_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.COS_TILT_1,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.COS_TILT_1 + "_scale"],
                low=err_params_value.get(P.COS_TILT_1 + "_low"),
                high=err_params_value.get(P.COS_TILT_1 + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

        error_magazine.register(
            P.COS_TILT_2,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.COS_TILT_2 + "_scale"],
                low=err_params_value.get(P.COS_TILT_2 + "_low"),
                high=err_params_value.get(P.COS_TILT_2 + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_phi_1:
        parameters_name += (P.PHI_1,)

        all_params.extend(
            [
                (P.PHI_1 + "_high_g", N_g),
                (P.PHI_1 + "_high_pl", N_pl),
                (P.PHI_1 + "_low_g", N_g),
                (P.PHI_1 + "_low_pl", N_pl),
            ]
        )

        def phi_1_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.PHI_1 + "_scale"],
                low=err_params_value.get(P.PHI_1 + "_low", 0.0),
                high=err_params_value.get(P.PHI_1 + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

        error_magazine.register(P.PHI_1, phi_1_error)

    if has_phi_2:
        parameters_name += (P.PHI_2,)

        all_params.extend(
            [
                (P.PHI_2 + "_high_g", N_g),
                (P.PHI_2 + "_high_pl", N_pl),
                (P.PHI_2 + "_low_g", N_g),
                (P.PHI_2 + "_low_pl", N_pl),
            ]
        )

        def phi_2_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.PHI_2 + "_scale"],
                low=err_params_value.get(P.PHI_2 + "_low", 0.0),
                high=err_params_value.get(P.PHI_2 + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

        error_magazine.register(P.PHI_2, phi_2_error)

    if has_phi_12:
        parameters_name += (P.PHI_12,)

        all_params.extend(
            [
                (P.PHI_12 + "_high_g", N_g),
                (P.PHI_12 + "_high_pl", N_pl),
                (P.PHI_12 + "_low_g", N_g),
                (P.PHI_12 + "_low_pl", N_pl),
            ]
        )

        def phi_12_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.PHI_12 + "_scale"],
                low=err_params_value.get(P.PHI_12 + "_low", 0.0),
                high=err_params_value.get(P.PHI_12 + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

        error_magazine.register(P.PHI_12, phi_12_error)

    if use_eccentricity_mixture:
        parameters_name += (P.ECCENTRICITY,)
        all_params.extend(
            [
                (P.ECCENTRICITY + "_comp1_high_g", N_g),
                (P.ECCENTRICITY + "_comp1_high_pl", N_pl),
                (P.ECCENTRICITY + "_comp1_loc_g", N_g),
                (P.ECCENTRICITY + "_comp1_loc_pl", N_pl),
                (P.ECCENTRICITY + "_comp1_low_g", N_g),
                (P.ECCENTRICITY + "_comp1_low_pl", N_pl),
                (P.ECCENTRICITY + "_comp1_scale_g", N_g),
                (P.ECCENTRICITY + "_comp1_scale_pl", N_pl),
                (P.ECCENTRICITY + "_comp2_high_g", N_g),
                (P.ECCENTRICITY + "_comp2_high_pl", N_pl),
                (P.ECCENTRICITY + "_comp2_loc_g", N_g),
                (P.ECCENTRICITY + "_comp2_loc_pl", N_pl),
                (P.ECCENTRICITY + "_comp2_low_g", N_g),
                (P.ECCENTRICITY + "_comp2_low_pl", N_pl),
                (P.ECCENTRICITY + "_comp2_scale_g", N_g),
                (P.ECCENTRICITY + "_comp2_scale_pl", N_pl),
                (P.ECCENTRICITY + "_zeta_g", N_g),
                (P.ECCENTRICITY + "_zeta_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.ECCENTRICITY,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.ECCENTRICITY + "_scale"],
                low=err_params_value.get(P.ECCENTRICITY + "_low"),
                high=err_params_value.get(P.ECCENTRICITY + "_high"),
                cut_low=0.0,
                cut_high=1.0,
            ),
        )

    if has_mean_anomaly:
        parameters_name += (P.MEAN_ANOMALY,)

        all_params.extend(
            [
                (P.MEAN_ANOMALY + "_high_g", N_g),
                (P.MEAN_ANOMALY + "_high_pl", N_pl),
                (P.MEAN_ANOMALY + "_low_g", N_g),
                (P.MEAN_ANOMALY + "_low_pl", N_pl),
            ]
        )

        def mean_anomaly_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.MEAN_ANOMALY + "_scale"],
                low=err_params_value.get(P.MEAN_ANOMALY + "_low", 0.0),
                high=err_params_value.get(P.MEAN_ANOMALY + "_high", 2.0 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))
            err_x = jnp.mod(err_x, 2.0 * jnp.pi)
            return err_x

        error_magazine.register(P.MEAN_ANOMALY, mean_anomaly_error)

    if has_redshift:
        parameters_name += (P.REDSHIFT,)

        all_params.extend(
            [
                (P.REDSHIFT + "_kappa_g", N_g),
                (P.REDSHIFT + "_kappa_pl", N_pl),
                (P.REDSHIFT + "_z_max_g", N_g),
                (P.REDSHIFT + "_z_max_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.REDSHIFT,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.REDSHIFT + "_scale"],
                low=err_params_value.get(P.REDSHIFT + "_low"),
                high=err_params_value.get(P.REDSHIFT + "_high"),
                cut_low=1e-3,
                cut_high=None,
            ),
        )

    if has_right_ascension:
        parameters_name += (P.RIGHT_ASCENSION,)

        all_params.extend(
            [
                (P.RIGHT_ASCENSION + "_high_g", N_g),
                (P.RIGHT_ASCENSION + "_high_pl", N_pl),
                (P.RIGHT_ASCENSION + "_low_g", N_g),
                (P.RIGHT_ASCENSION + "_low_pl", N_pl),
            ]
        )

        def right_ascension_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.RIGHT_ASCENSION + "_scale"],
                low=err_params_value.get(P.RIGHT_ASCENSION + "_low", 0.0),
                high=err_params_value.get(P.RIGHT_ASCENSION + "_high", 2.0 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))
            err_x = jnp.mod(err_x, 2.0 * jnp.pi)
            return err_x

        error_magazine.register(P.RIGHT_ASCENSION, right_ascension_error)

    if has_sin_declination:
        parameters_name += (P.SIN_DECLINATION,)

        all_params.extend(
            [
                (P.SIN_DECLINATION + "_high_g", N_g),
                (P.SIN_DECLINATION + "_high_pl", N_pl),
                (P.SIN_DECLINATION + "_low_g", N_g),
                (P.SIN_DECLINATION + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.SIN_DECLINATION,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.SIN_DECLINATION + "_scale"],
                low=err_params_value.get(P.SIN_DECLINATION + "_low"),
                high=err_params_value.get(P.SIN_DECLINATION + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_detection_time:
        parameters_name += (P.DETECTION_TIME,)

        all_params.extend(
            [
                (P.DETECTION_TIME + "_high_g", N_g),
                (P.DETECTION_TIME + "_high_pl", N_pl),
                (P.DETECTION_TIME + "_low_g", N_g),
                (P.DETECTION_TIME + "_low_pl", N_pl),
            ]
        )

        def detection_time_error(x, size, key):
            eps = 1e-6  # To avoid log(0) or log of negative
            safe_x = jnp.maximum(x, eps)
            err_x = dist.LogNormal(
                loc=jnp.log(safe_x),
                scale=err_params_value[P.DETECTION_TIME + "_scale"],
            ).sample(key=key, sample_shape=(size,))

            return err_x

        error_magazine.register(P.DETECTION_TIME, detection_time_error)

    if has_cos_iota:
        parameters_name += (P.COS_IOTA,)

        all_params.extend(
            [
                (P.COS_IOTA + "_high_g", N_g),
                (P.COS_IOTA + "_high_pl", N_pl),
                (P.COS_IOTA + "_low_g", N_g),
                (P.COS_IOTA + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.COS_IOTA,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.COS_IOTA + "_scale"],
                low=err_params_value.get(P.COS_IOTA + "_low"),
                high=err_params_value.get(P.COS_IOTA + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_polarization_angle:
        parameters_name += (P.POLARIZATION_ANGLE,)

        all_params.extend(
            [
                (P.POLARIZATION_ANGLE + "_high_g", N_g),
                (P.POLARIZATION_ANGLE + "_high_pl", N_pl),
                (P.POLARIZATION_ANGLE + "_low_g", N_g),
                (P.POLARIZATION_ANGLE + "_low_pl", N_pl),
            ]
        )

        def polarization_angle_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.POLARIZATION_ANGLE + "_scale"],
                low=err_params_value.get(P.POLARIZATION_ANGLE + "_low", 0.0),
                high=err_params_value.get(P.POLARIZATION_ANGLE + "_high", jnp.pi),
            ).sample(key=key, sample_shape=(size,))
            err_x = jnp.mod(err_x, jnp.pi)
            return err_x

        error_magazine.register(P.POLARIZATION_ANGLE, polarization_angle_error)

    if has_phi_orb:
        parameters_name += (P.PHI_ORB,)

        all_params.extend(
            [
                (P.PHI_ORB + "_high_g", N_g),
                (P.PHI_ORB + "_high_pl", N_pl),
                (P.PHI_ORB + "_low_g", N_g),
                (P.PHI_ORB + "_low_pl", N_pl),
            ]
        )

        def phi_orb_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.PHI_ORB + "_scale"],
                low=err_params_value.get(P.PHI_ORB + "_low", 0.0),
                high=err_params_value.get(P.PHI_ORB + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

        error_magazine.register(P.PHI_ORB, phi_orb_error)

    extended_params = []
    for params in all_params:
        extended_params.extend(expand_arguments(*params))

    model_param = match_all(extended_params, model_json)

    model_param.update(
        {
            "N_pl": N_pl,
            "N_g": N_g,
            "use_beta_spin_magnitude": has_beta_spin_magnitude,
            "use_spin_magnitude_mixture": has_spin_magnitude_mixture,
            "use_truncated_normal_spin_x": has_truncated_normal_spin_x,
            "use_truncated_normal_spin_y": has_truncated_normal_spin_y,
            "use_truncated_normal_spin_z": has_truncated_normal_spin_z,
            "use_chi_eff_mixture": has_chi_eff_mixture,
            "use_skew_normal_chi_eff": use_skew_normal_chi_eff,
            "use_tilt": has_tilt,
            "use_eccentricity_mixture": use_eccentricity_mixture,
            "use_mean_anomaly": has_mean_anomaly,
            "use_redshift": has_redshift,
            "use_cos_iota": has_cos_iota,
            "use_phi_12": has_phi_12,
            "use_polarization_angle": has_polarization_angle,
            "use_right_ascension": has_right_ascension,
            "use_sin_declination": has_sin_declination,
            "use_detection_time": has_detection_time,
            "use_phi_1": has_phi_1,
            "use_phi_2": has_phi_2,
            "use_phi_orb": has_phi_orb,
        }
    )

    logger.info(f"Setting the random number generator key with seed {args.seed}.")
    pmean_key, factory_key = jrd.split(jrd.PRNGKey(args.seed), 2)

    pmean_config = read_json(args.pmean_json)
    log_selection_fn, erate_estimator, _, _ = (
        get_selection_fn_and_poisson_mean_estimator(
            key=pmean_key, parameters=parameters_name, **pmean_config
        )
    )

    popfactory = PopulationFactory(
        model_fn=NPowerlawMGaussian,
        model_params=model_param,
        parameters=parameters_name,
        log_selection_fn=log_selection_fn,
        poisson_mean_estimator=erate_estimator,
        num_realizations=args.num_realizations,
        error_size=args.error_size,
        tile_covariance=args.tile_covariance,
    )
    popfactory.produce(factory_key)
