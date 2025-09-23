# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from typing import List, Tuple

from jax import numpy as jnp, random as jrd
from numpyro import distributions as dist

import gwkokab
from gwkokab.errors import banana_error_m1_m2, truncated_normal_error
from gwkokab.models import NPowerlawMGaussian
from gwkokab.models.npowerlawmgaussian._ncombination import (
    create_truncated_normal_distributions,
)
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

    spin_group = model_group.add_mutually_exclusive_group()
    spin_group.add_argument(
        "--add-beta-spin",
        action="store_true",
        help="Include beta spin parameters in the model.",
    )
    spin_group.add_argument(
        "--add-truncated-normal-spin",
        action="store_true",
        help="Include truncated normal spin parameters in the model.",
    )

    model_group.add_argument(
        "--add-tilt",
        action="store_true",
        help="Include tilt parameters in the model.",
    )
    model_group.add_argument(
        "--add-truncated-normal-eccentricity",
        action="store_true",
        help="Include truncated normal eccentricity parameters in the model.",
    )
    model_group.add_argument(
        "--add-mean-anomaly",
        action="store_true",
        help="Include mean_anomaly parameter in the model",
    )
    model_group.add_argument(
        "--add-redshift",
        action="store_true",
        help="Include redshift parameter in the model",
    )
    model_group.add_argument(
        "--add-cos-iota",
        action="store_true",
        help="Include cos_iota parameter in the model",
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
        "--add-phi-12",
        action="store_true",
        help="Include phi_12 parameter in the model",
    )
    model_group.add_argument(
        "--add-phi-orb",
        action="store_true",
        help="Include phi_orb parameter in the model",
    )

    err_group = parser.add_argument_group("Error Options")
    err_group.add_argument(
        "--err-json",
        help="Path to the JSON file containing the error parameters",
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

    has_spin = args.add_beta_spin or args.add_truncated_normal_spin
    has_tilt = args.add_tilt
    has_eccentricity = args.add_truncated_normal_eccentricity
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
    if has_spin:
        err_params_name.extend(
            [
                "chi1_high",
                "chi1_low",
                "chi1_scale",
                "chi2_high",
                "chi2_low",
                "chi2_scale",
            ]
        )
    if has_tilt:
        err_params_name.extend(
            [
                "cos_tilt_1_high",
                "cos_tilt_1_low",
                "cos_tilt_1_scale",
                "cos_tilt_2_high",
                "cos_tilt_2_low",
                "cos_tilt_2_scale",
            ]
        )
    if has_eccentricity:
        err_params_name.extend(["ecc_high", "ecc_low", "ecc_scale"])
    if has_mean_anomaly:
        err_params_name.extend(
            [
                P.MEAN_ANOMALY.value + "_high",
                P.MEAN_ANOMALY.value + "_low",
                P.MEAN_ANOMALY.value + "_scale",
            ]
        )
    if has_redshift:
        err_params_name.extend(["redshift_high", "redshift_low", "redshift_scale"])
    if has_cos_iota:
        err_params_name.extend(
            [
                P.COS_IOTA.value + "_high",
                P.COS_IOTA.value + "_low",
                P.COS_IOTA.value + "_scale",
            ]
        )
    if has_polarization_angle:
        err_params_name.extend(
            [
                P.POLARIZATION_ANGLE.value + "_high",
                P.POLARIZATION_ANGLE.value + "_low",
                P.POLARIZATION_ANGLE.value + "_scale",
            ]
        )
    if has_right_ascension:
        err_params_name.extend(
            [
                P.RIGHT_ASCENSION.value + "_high",
                P.RIGHT_ASCENSION.value + "_low",
                P.RIGHT_ASCENSION.value + "_scale",
            ]
        )
    if has_sin_declination:
        err_params_name.extend(
            [
                P.SIN_DECLINATION.value + "_high",
                P.SIN_DECLINATION.value + "_low",
                P.SIN_DECLINATION.value + "_scale",
            ]
        )
    if has_detection_time:
        err_params_name.append(P.DETECTION_TIME.value + "_scale")
    if has_phi_1:
        err_params_name.extend(
            [
                P.PHI_1.value + "_high",
                P.PHI_1.value + "_low",
                P.PHI_1.value + "_scale",
            ]
        )
    if has_phi_2:
        err_params_name.extend(
            [
                P.PHI_2.value + "_high",
                P.PHI_2.value + "_low",
                P.PHI_2.value + "_scale",
            ]
        )
    if has_phi_12:
        err_params_name.extend(
            [
                P.PHI_12.value + "_high",
                P.PHI_12.value + "_low",
                P.PHI_12.value + "_scale",
            ]
        )
    if has_phi_orb:
        err_params_name.extend(
            [
                P.PHI_ORB.value + "_high",
                P.PHI_ORB.value + "_low",
                P.PHI_ORB.value + "_scale",
            ]
        )

    err_params_value = match_all(err_params_name, err_json)

    all_params: List[Tuple[str, int]] = [
        ("alpha_pl", N_pl),
        ("beta_pl", N_pl),
        ("log_rate", N_pl + N_g),
        ("m1_high_g", N_g),
        ("m1_loc_g", N_g),
        ("m1_low_g", N_g),
        ("m1_scale_g", N_g),
        ("m2_high_g", N_g),
        ("m2_loc_g", N_g),
        ("m2_low_g", N_g),
        ("m2_scale_g", N_g),
        ("mmax_pl", N_pl),
        ("mmin_pl", N_pl),
    ]

    parameters_name: Tuple[str, ...] = (
        P.PRIMARY_MASS_SOURCE.value,
        P.SECONDARY_MASS_SOURCE.value,
    )

    error_magazine.register(
        (P.PRIMARY_MASS_SOURCE.value, P.SECONDARY_MASS_SOURCE.value),
        lambda x, size, key: banana_error_m1_m2(
            x,
            size,
            key,
            scale_Mc=err_params_value["scale_Mc"],
            scale_eta=err_params_value["scale_eta"],
        ),
    )

    if has_spin:
        parameters_name += (
            P.PRIMARY_SPIN_MAGNITUDE.value,
            P.SECONDARY_SPIN_MAGNITUDE.value,
        )
        if args.add_truncated_normal_spin:
            gwkokab.models.npowerlawmgaussian._model.build_spin_distributions = (
                create_truncated_normal_distributions
            )
            all_params.extend(
                [
                    ("chi1_high_g", N_g),
                    ("chi1_high_pl", N_pl),
                    ("chi1_loc_g", N_g),
                    ("chi1_loc_pl", N_pl),
                    ("chi1_low_g", N_g),
                    ("chi1_low_pl", N_pl),
                    ("chi1_scale_g", N_g),
                    ("chi1_scale_pl", N_pl),
                    ("chi2_high_g", N_g),
                    ("chi2_high_pl", N_pl),
                    ("chi2_loc_g", N_g),
                    ("chi2_loc_pl", N_pl),
                    ("chi2_low_g", N_g),
                    ("chi2_low_pl", N_pl),
                    ("chi2_scale_g", N_g),
                    ("chi2_scale_pl", N_pl),
                ]
            )
        if args.add_beta_spin:
            all_params.extend(
                [
                    ("chi1_mean_g", N_g),
                    ("chi1_mean_pl", N_pl),
                    ("chi1_variance_g", N_g),
                    ("chi1_variance_pl", N_pl),
                    ("chi2_mean_g", N_g),
                    ("chi2_mean_pl", N_pl),
                    ("chi2_variance_g", N_g),
                    ("chi2_variance_pl", N_pl),
                ]
            )

        error_magazine.register(
            P.PRIMARY_SPIN_MAGNITUDE.value,
            partial(
                truncated_normal_error,
                scale=err_params_value["chi1_scale"],
                low=err_params_value.get("chi1_low"),
                high=err_params_value.get("chi1_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

        error_magazine.register(
            P.SECONDARY_SPIN_MAGNITUDE.value,
            partial(
                truncated_normal_error,
                scale=err_params_value["chi2_scale"],
                low=err_params_value.get("chi2_low"),
                high=err_params_value.get("chi2_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_tilt:
        parameters_name += (P.COS_TILT_1.value, P.COS_TILT_2.value)
        all_params.extend(
            [
                ("cos_tilt_zeta_g", N_g),
                ("cos_tilt_zeta_pl", N_pl),
                ("cos_tilt1_scale_g", N_g),
                ("cos_tilt1_scale_pl", N_pl),
                ("cos_tilt2_scale_g", N_g),
                ("cos_tilt2_scale_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.COS_TILT_1.value,
            partial(
                truncated_normal_error,
                scale=err_params_value["cos_tilt_1_scale"],
                low=err_params_value.get("cos_tilt_1_low"),
                high=err_params_value.get("cos_tilt_1_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

        error_magazine.register(
            P.COS_TILT_2.value,
            partial(
                truncated_normal_error,
                scale=err_params_value["cos_tilt_2_scale"],
                low=err_params_value.get("cos_tilt_2_low"),
                high=err_params_value.get("cos_tilt_2_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_phi_1:
        parameters_name += (P.PHI_1.value,)

        all_params.extend(
            [
                (P.PHI_1.value + "_high_g", N_g),
                (P.PHI_1.value + "_high_pl", N_pl),
                (P.PHI_1.value + "_low_g", N_g),
                (P.PHI_1.value + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(P.PHI_1.value)
        def phi_1_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.PHI_1.value + "_scale"],
                low=err_params_value.get(P.PHI_1.value + "_low", 0.0),
                high=err_params_value.get(P.PHI_1.value + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

    if has_phi_2:
        parameters_name += (P.PHI_2.value,)

        all_params.extend(
            [
                (P.PHI_2.value + "_high_g", N_g),
                (P.PHI_2.value + "_high_pl", N_pl),
                (P.PHI_2.value + "_low_g", N_g),
                (P.PHI_2.value + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(P.PHI_2.value)
        def phi_2_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.PHI_2.value + "_scale"],
                low=err_params_value.get(P.PHI_2.value + "_low", 0.0),
                high=err_params_value.get(P.PHI_2.value + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

    if has_phi_12:
        parameters_name += (P.PHI_12.value,)

        all_params.extend(
            [
                (P.PHI_12.value + "_high_g", N_g),
                (P.PHI_12.value + "_high_pl", N_pl),
                (P.PHI_12.value + "_low_g", N_g),
                (P.PHI_12.value + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(P.PHI_12.value)
        def phi_12_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.PHI_12.value + "_scale"],
                low=err_params_value.get(P.PHI_12.value + "_low", 0.0),
                high=err_params_value.get(P.PHI_12.value + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

    if has_eccentricity:
        parameters_name += (P.ECCENTRICITY.value,)
        all_params.extend(
            [
                ("ecc_high_g", N_g),
                ("ecc_high_pl", N_pl),
                ("ecc_loc_g", N_g),
                ("ecc_loc_pl", N_pl),
                ("ecc_low_g", N_g),
                ("ecc_low_pl", N_pl),
                ("ecc_scale_g", N_g),
                ("ecc_scale_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.ECCENTRICITY.value,
            partial(
                truncated_normal_error,
                scale=err_params_value["ecc_scale"],
                low=err_params_value.get("ecc_low"),
                high=err_params_value.get("ecc_high"),
                cut_low=0.0,
                cut_high=1.0,
            ),
        )

    if has_mean_anomaly:
        parameters_name += (P.MEAN_ANOMALY.value,)

        all_params.extend(
            [
                (P.MEAN_ANOMALY.value + "_high_g", N_g),
                (P.MEAN_ANOMALY.value + "_high_pl", N_pl),
                (P.MEAN_ANOMALY.value + "_low_g", N_g),
                (P.MEAN_ANOMALY.value + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.MEAN_ANOMALY.value,
            partial(
                truncated_normal_error,
                scale=err_params_value["mean_anomaly_scale"],
                low=err_params_value.get("mean_anomaly_low"),
                high=err_params_value.get("mean_anomaly_high"),
                cut_low=0.0,
                cut_high=1.0,
            ),
        )

    if has_redshift:
        parameters_name += (P.REDSHIFT.value,)

        all_params.extend(
            [
                ("redshift_kappa_g", N_g),
                ("redshift_kappa_pl", N_pl),
                ("redshift_z_max_g", N_g),
                ("redshift_z_max_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.REDSHIFT.value,
            partial(
                truncated_normal_error,
                scale=err_params_value["redshift_scale"],
                low=err_params_value.get("redshift_low"),
                high=err_params_value.get("redshift_high"),
                cut_low=1e-3,
                cut_high=None,
            ),
        )

    if has_right_ascension:
        parameters_name += (P.RIGHT_ASCENSION.value,)

        all_params.extend(
            [
                (P.RIGHT_ASCENSION.value + "_high_g", N_g),
                (P.RIGHT_ASCENSION.value + "_high_pl", N_pl),
                (P.RIGHT_ASCENSION.value + "_low_g", N_g),
                (P.RIGHT_ASCENSION.value + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.RIGHT_ASCENSION.value,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.RIGHT_ASCENSION.value + "_scale"],
                low=err_params_value.get(P.RIGHT_ASCENSION.value + "_low"),
                high=err_params_value.get(P.RIGHT_ASCENSION.value + "_high"),
                cut_low=0.0,
                cut_high=2.0 * jnp.pi,
            ),
        )

    if has_sin_declination:
        parameters_name += (P.SIN_DECLINATION.value,)

        all_params.extend(
            [
                (P.SIN_DECLINATION.value + "_high_g", N_g),
                (P.SIN_DECLINATION.value + "_high_pl", N_pl),
                (P.SIN_DECLINATION.value + "_low_g", N_g),
                (P.SIN_DECLINATION.value + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.SIN_DECLINATION.value,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.SIN_DECLINATION.value + "_scale"],
                low=err_params_value.get(P.SIN_DECLINATION.value + "_low"),
                high=err_params_value.get(P.SIN_DECLINATION.value + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_detection_time:
        parameters_name += (P.DETECTION_TIME.value,)

        all_params.extend(
            [
                (P.DETECTION_TIME.value + "_high_g", N_g),
                (P.DETECTION_TIME.value + "_high_pl", N_pl),
                (P.DETECTION_TIME.value + "_low_g", N_g),
                (P.DETECTION_TIME.value + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(P.DETECTION_TIME.value)
        def detection_time_error(x, size, key):
            eps = 1e-6  # To avoid log(0) or log of negative
            safe_x = jnp.maximum(x, eps)
            err_x = dist.LogNormal(
                loc=jnp.log(safe_x),
                scale=err_params_value[P.DETECTION_TIME.value + "_scale"],
            ).sample(key=key, sample_shape=(size,))

            return err_x

    if has_cos_iota:
        parameters_name += (P.COS_IOTA.value,)

        all_params.extend(
            [
                (P.COS_IOTA.value + "_high_g", N_g),
                (P.COS_IOTA.value + "_high_pl", N_pl),
                (P.COS_IOTA.value + "_low_g", N_g),
                (P.COS_IOTA.value + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.COS_IOTA.value,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.COS_IOTA.value + "_scale"],
                low=err_params_value.get(P.COS_IOTA.value + "_low"),
                high=err_params_value.get(P.COS_IOTA.value + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_polarization_angle:
        parameters_name += (P.POLARIZATION_ANGLE.value,)

        all_params.extend(
            [
                (P.POLARIZATION_ANGLE.value + "_high_g", N_g),
                (P.POLARIZATION_ANGLE.value + "_high_pl", N_pl),
                (P.POLARIZATION_ANGLE.value + "_low_g", N_g),
                (P.POLARIZATION_ANGLE.value + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            P.POLARIZATION_ANGLE.value,
            partial(
                truncated_normal_error,
                scale=err_params_value[P.POLARIZATION_ANGLE.value + "_scale"],
                low=err_params_value.get(P.POLARIZATION_ANGLE.value + "_low"),
                high=err_params_value.get(P.POLARIZATION_ANGLE.value + "_high"),
                cut_low=0.0,
                cut_high=jnp.pi,
            ),
        )

    if has_phi_orb:
        parameters_name += (P.PHI_ORB.value,)

        all_params.extend(
            [
                (P.PHI_ORB.value + "_high_g", N_g),
                (P.PHI_ORB.value + "_high_pl", N_pl),
                (P.PHI_ORB.value + "_low_g", N_g),
                (P.PHI_ORB.value + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(P.PHI_ORB.value)
        def phi_orb_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[P.PHI_ORB.value + "_scale"],
                low=err_params_value.get(P.PHI_ORB.value + "_low", 0.0),
                high=err_params_value.get(P.PHI_ORB.value + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

    extended_params = []
    for params in all_params:
        extended_params.extend(expand_arguments(*params))

    model_param = match_all(extended_params, model_json)

    model_param.update(
        {
            "N_pl": N_pl,
            "N_g": N_g,
            "use_spin": has_spin,
            "use_tilt": has_tilt,
            "use_eccentricity": has_eccentricity,
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

    pmean_key, factory_key = jrd.split(jrd.PRNGKey(args.seed), 2)

    pmean_config = read_json(args.pmean_json)
    log_selection_fn, erate_estimator, _ = get_selection_fn_and_poisson_mean_estimator(
        key=pmean_key, parameters=parameters_name, **pmean_config
    )

    popfactory = PopulationFactory(
        model_fn=NPowerlawMGaussian,
        model_params=model_param,
        parameters=parameters_name,
        log_selection_fn=log_selection_fn,
        poisson_mean_estimator=erate_estimator,
        num_realizations=args.num_realizations,
        error_size=args.error_size,
    )
    popfactory.produce(factory_key)
