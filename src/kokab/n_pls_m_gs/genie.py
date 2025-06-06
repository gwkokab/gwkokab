# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from typing import List, Tuple

import numpy as np
from jax import numpy as jnp, random as jrd
from numpyro import distributions as dist

import gwkokab
from gwkokab.errors import banana_error_m1_m2, truncated_normal_error
from gwkokab.models import NPowerlawMGaussian
from gwkokab.models.utils import create_truncated_normal_distributions
from gwkokab.parameters import (
    COS_IOTA,
    COS_TILT_1,
    COS_TILT_2,
    DETECTION_TIME,
    ECCENTRICITY,
    MEAN_ANOMALY,
    PHI_1,
    PHI_2,
    PHI_12,
    PHI_ORB,
    POLARIZATION_ANGLE,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    REDSHIFT,
    RIGHT_ASCENSION,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
    SIN_DECLINATION,
)
from gwkokab.poisson_mean import PoissonMean
from gwkokab.population import error_magazine, PopulationFactory
from kokab.utils import genie_parser, poisson_mean_parser
from kokab.utils.common import expand_arguments, vt_json_read_and_process
from kokab.utils.regex import match_all


m1_source_name = PRIMARY_MASS_SOURCE.name
m2_source_name = SECONDARY_MASS_SOURCE.name
chi1_name = PRIMARY_SPIN_MAGNITUDE.name
chi2_name = SECONDARY_SPIN_MAGNITUDE.name
cos_tilt_1_name = COS_TILT_1.name
cos_tilt_2_name = COS_TILT_2.name
ecc_name = ECCENTRICITY.name
mean_anomaly_name = MEAN_ANOMALY.name
redshift_name = REDSHIFT.name
cos_iota_name = COS_IOTA.name
phi_12_name = PHI_12.name
polarization_angle_name = POLARIZATION_ANGLE.name
right_ascension_name = RIGHT_ASCENSION.name
sin_declination_name = SIN_DECLINATION.name
detection_time_name = DETECTION_TIME.name
phi_1_name = PHI_1.name
phi_2_name = PHI_2.name
phi_orb_name = PHI_ORB.name


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
        help="Include redshift parameters in the model.",
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
                mean_anomaly_name + "_high",
                mean_anomaly_name + "_low",
                mean_anomaly_name + "_scale",
            ]
        )
    if has_redshift:
        err_params_name.extend(["redshift_high", "redshift_low", "redshift_scale"])
    if has_cos_iota:
        err_params_name.extend(
            [cos_iota_name + "_high", cos_iota_name + "_low", cos_iota_name + "_scale"]
        )
    if has_polarization_angle:
        err_params_name.extend(
            [
                polarization_angle_name + "_high",
                polarization_angle_name + "_low",
                polarization_angle_name + "_scale",
            ]
        )
    if has_right_ascension:
        err_params_name.extend(
            [
                right_ascension_name + "_high",
                right_ascension_name + "_low",
                right_ascension_name + "_scale",
            ]
        )
    if has_sin_declination:
        err_params_name.extend(
            [
                sin_declination_name + "_high",
                sin_declination_name + "_low",
                sin_declination_name + "_scale",
            ]
        )
    if has_detection_time:
        err_params_name.append(detection_time_name + "_scale")
    if has_phi_1:
        err_params_name.extend(
            [phi_1_name + "_high", phi_1_name + "_low", phi_1_name + "_scale"]
        )
    if has_phi_2:
        err_params_name.extend(
            [phi_2_name + "_high", phi_2_name + "_low", phi_2_name + "_scale"]
        )
    if has_phi_12:
        err_params_name.extend(
            [phi_12_name + "_high", phi_12_name + "_low", phi_12_name + "_scale"]
        )
    if has_phi_orb:
        err_params_name.extend(
            [phi_orb_name + "_high", phi_orb_name + "_low", phi_orb_name + "_scale"]
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

    parameters_name: Tuple[str, ...] = (m1_source_name, m2_source_name)

    error_magazine.register(
        (m1_source_name, m2_source_name),
        lambda x, size, key: banana_error_m1_m2(
            x,
            size,
            key,
            scale_Mc=err_params_value["scale_Mc"],
            scale_eta=err_params_value["scale_eta"],
        ),
    )

    if has_spin:
        parameters_name += (chi1_name, chi2_name)
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
            chi1_name,
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
            chi2_name,
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
        parameters_name += (cos_tilt_1_name, cos_tilt_2_name)
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
            cos_tilt_1_name,
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
            cos_tilt_2_name,
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
        parameters_name += (phi_1_name,)

        all_params.extend(
            [
                (phi_1_name + "_high_g", N_g),
                (phi_1_name + "_high_pl", N_pl),
                (phi_1_name + "_low_g", N_g),
                (phi_1_name + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(phi_1_name)
        def phi_1_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[phi_1_name + "_scale"],
                low=err_params_value.get(phi_1_name + "_low", 0.0),
                high=err_params_value.get(phi_1_name + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

    if has_phi_2:
        parameters_name += (phi_2_name,)

        all_params.extend(
            [
                (phi_2_name + "_high_g", N_g),
                (phi_2_name + "_high_pl", N_pl),
                (phi_2_name + "_low_g", N_g),
                (phi_2_name + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(phi_2_name)
        def phi_2_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[phi_2_name + "_scale"],
                low=err_params_value.get(phi_2_name + "_low", 0.0),
                high=err_params_value.get(phi_2_name + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

    if has_phi_12:
        parameters_name += (phi_12_name,)

        all_params.extend(
            [
                (phi_12_name + "_high_g", N_g),
                (phi_12_name + "_high_pl", N_pl),
                (phi_12_name + "_low_g", N_g),
                (phi_12_name + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(phi_12_name)
        def phi_12_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[phi_12_name + "_scale"],
                low=err_params_value.get(phi_12_name + "_low", 0.0),
                high=err_params_value.get(phi_12_name + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

    if has_eccentricity:
        parameters_name += (ecc_name,)
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
            ecc_name,
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
        parameters_name += (mean_anomaly_name,)

        all_params.extend(
            [
                (mean_anomaly_name + "_high_g", N_g),
                (mean_anomaly_name + "_high_pl", N_pl),
                (mean_anomaly_name + "_low_g", N_g),
                (mean_anomaly_name + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            mean_anomaly_name,
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
        parameters_name += (redshift_name,)

        all_params.extend(
            [
                ("redshift_lamb_g", N_g),
                ("redshift_lamb_pl", N_pl),
                ("redshift_z_max_g", N_g),
                ("redshift_z_max_pl", N_pl),
            ]
        )

        error_magazine.register(
            redshift_name,
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
        parameters_name += (right_ascension_name,)

        all_params.extend(
            [
                (right_ascension_name + "_high_g", N_g),
                (right_ascension_name + "_high_pl", N_pl),
                (right_ascension_name + "_low_g", N_g),
                (right_ascension_name + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            right_ascension_name,
            partial(
                truncated_normal_error,
                scale=err_params_value[right_ascension_name + "_scale"],
                low=err_params_value.get(right_ascension_name + "_low"),
                high=err_params_value.get(right_ascension_name + "_high"),
                cut_low=0.0,
                cut_high=2.0 * jnp.pi,
            ),
        )

    if has_sin_declination:
        parameters_name += (sin_declination_name,)

        all_params.extend(
            [
                (sin_declination_name + "_high_g", N_g),
                (sin_declination_name + "_high_pl", N_pl),
                (sin_declination_name + "_low_g", N_g),
                (sin_declination_name + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            sin_declination_name,
            partial(
                truncated_normal_error,
                scale=err_params_value[sin_declination_name + "_scale"],
                low=err_params_value.get(sin_declination_name + "_low"),
                high=err_params_value.get(sin_declination_name + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_detection_time:
        parameters_name += (detection_time_name,)

        all_params.extend(
            [
                (detection_time_name + "_high_g", N_g),
                (detection_time_name + "_high_pl", N_pl),
                (detection_time_name + "_low_g", N_g),
                (detection_time_name + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(detection_time_name)
        def detection_time_error(x, size, key):
            eps = 1e-6  # To avoid log(0) or log of negative
            safe_x = jnp.maximum(x, eps)
            err_x = dist.LogNormal(
                loc=jnp.log(safe_x),
                scale=err_params_value[detection_time_name + "_scale"],
            ).sample(key=key, sample_shape=(size,))

            return err_x

    if has_cos_iota:
        parameters_name += (cos_iota_name,)

        all_params.extend(
            [
                (cos_iota_name + "_high_g", N_g),
                (cos_iota_name + "_high_pl", N_pl),
                (cos_iota_name + "_low_g", N_g),
                (cos_iota_name + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            cos_iota_name,
            partial(
                truncated_normal_error,
                scale=err_params_value[cos_iota_name + "_scale"],
                low=err_params_value.get(cos_iota_name + "_low"),
                high=err_params_value.get(cos_iota_name + "_high"),
                cut_low=-1.0,
                cut_high=1.0,
            ),
        )

    if has_polarization_angle:
        parameters_name += (polarization_angle_name,)

        all_params.extend(
            [
                (polarization_angle_name + "_high_g", N_g),
                (polarization_angle_name + "_high_pl", N_pl),
                (polarization_angle_name + "_low_g", N_g),
                (polarization_angle_name + "_low_pl", N_pl),
            ]
        )

        error_magazine.register(
            polarization_angle_name,
            partial(
                truncated_normal_error,
                scale=err_params_value[polarization_angle_name + "_scale"],
                low=err_params_value.get(polarization_angle_name + "_low"),
                high=err_params_value.get(polarization_angle_name + "_high"),
                cut_low=0.0,
                cut_high=jnp.pi,
            ),
        )

    if has_phi_orb:
        parameters_name += (phi_orb_name,)

        all_params.extend(
            [
                (phi_orb_name + "_high_g", N_g),
                (phi_orb_name + "_high_pl", N_pl),
                (phi_orb_name + "_low_g", N_g),
                (phi_orb_name + "_low_pl", N_pl),
            ]
        )

        @error_magazine.register(phi_orb_name)
        def phi_orb_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_params_value[phi_orb_name + "_scale"],
                low=err_params_value.get(phi_orb_name + "_low", 0.0),
                high=err_params_value.get(phi_orb_name + "_high", 2 * jnp.pi),
            ).sample(key=key, sample_shape=(size,))

            err_x = jnp.mod(err_x, 2 * jnp.pi)
            return err_x

    extended_params = []
    for params in all_params:
        extended_params.extend(expand_arguments(*params))

    model_param = match_all(extended_params, model_json)

    model = NPowerlawMGaussian(
        N_pl=N_pl,
        N_g=N_g,
        use_spin=has_spin,
        use_tilt=has_tilt,
        use_eccentricity=has_eccentricity,
        use_mean_anomaly=has_mean_anomaly,
        use_redshift=has_redshift,
        use_cos_iota=has_cos_iota,
        use_phi_12=has_phi_12,
        use_polarization_angle=has_polarization_angle,
        use_right_ascension=has_right_ascension,
        use_sin_declination=has_sin_declination,
        use_detection_time=has_detection_time,
        use_phi_1=has_phi_1,
        use_phi_2=has_phi_2,
        use_phi_orb=has_phi_orb,
        **model_param,
    )

    nvt = vt_json_read_and_process(parameters_name, args.vt_json)
    logVT = nvt.get_mapped_logVT()

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(
        nvt,
        key=jrd.PRNGKey(np.random.randint(0, 2**32, dtype=np.uint32)),
        **pmean_kwargs,
    )

    popfactory = PopulationFactory(
        model=model,
        parameters=parameters_name,
        logVT_fn=logVT,
        ERate_fn=erate_estimator.__call__,
        num_realizations=args.num_realizations,
        error_size=args.error_size,
    )
    popfactory.produce()
