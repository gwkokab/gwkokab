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
from gwkokab.errors import banana_error_m1_m2
from gwkokab.models import NPowerlawMGaussian
from gwkokab.models.utils import create_truncated_normal_distributions
from gwkokab.parameters import (
    COS_INCLINATION,
    COS_TILT_1,
    COS_TILT_2,
    ECCENTRICITY,
    PHI_12,
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
from kokab.n_pls_m_gs.common import constraint
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
redshift_name = REDSHIFT.name
cos_inclination_name = COS_INCLINATION.name
phi_12_name = PHI_12.name
polarization_angle_name = POLARIZATION_ANGLE.name
right_ascension_name = RIGHT_ASCENSION.name
sin_declination_name = SIN_DECLINATION.name


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
    model_group.add_argument(
        "--no-spin",
        action="store_true",
        help="Do not include spin parameters in the model.",
    )
    model_group.add_argument(
        "--no-tilt",
        action="store_true",
        help="Do not include tilt parameters in the model.",
    )
    model_group.add_argument(
        "--no-eccentricity",
        action="store_true",
        help="Do not include eccentricity parameters in the model.",
    )
    model_group.add_argument(
        "--no-redshift",
        action="store_true",
        help="Do not include redshift parameters in the model.",
    )
    model_group.add_argument(
        "--no-cos-inclination",
        action="store_true",
        help="Do not include cos_inclination parameter in the model",
    )
    model_group.add_argument(
        "--no-phi-12",
        action="store_true",
        help="Do not include phi_12 parameter in the model",
    )
    model_group.add_argument(
        "--no-polarization-angle",
        action="store_true",
        help="Do not include polarization_angle parameter in the model",
    )
    model_group.add_argument(
        "--no-right-ascension",
        action="store_true",
        help="Do not include right_ascension parameter in the model",
    )
    model_group.add_argument(
        "--no-sin-declination",
        action="store_true",
        help="Do not include sin_declination parameter in the model",
    )
    model_group.add_argument(
        "--spin-truncated-normal",
        action="store_true",
        help="Use truncated normal distributions for spin parameters.",
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

    has_spin = not args.no_spin
    has_tilt = not args.no_tilt
    has_eccentricity = not args.no_eccentricity
    has_redshift = not args.no_redshift
    has_cos_inclination = not args.no_cos_inclination
    has_phi_12 = not args.no_phi_12
    has_polarization_angle = not args.no_polarization_angle
    has_right_ascension = not args.no_right_ascension
    has_sin_declination = not args.no_sin_declination

    err_param = match_all(
        [
            "chi1_high",
            "chi1_low",
            "chi1_scale",
            "chi2_high",
            "chi2_low",
            "chi2_scale",
            "cos_tilt_1_high",
            "cos_tilt_1_low",
            "cos_tilt_1_scale",
            "cos_tilt_2_high",
            "cos_tilt_2_low",
            "cos_tilt_2_scale",
            "ecc_err_high",
            "ecc_err_low",
            "ecc_err_scale",
            "redshift_high",
            "redshift_low",
            "redshift_scale",
            "scale_eta",
            "scale_Mc",
            cos_inclination_name + "_high",
            cos_inclination_name + "_low",
            cos_inclination_name + "_scale",
            phi_12_name + "_high",
            phi_12_name + "_low",
            phi_12_name + "_scale",
            polarization_angle_name + "_high",
            polarization_angle_name + "_low",
            polarization_angle_name + "_scale",
            right_ascension_name + "_high",
            right_ascension_name + "_low",
            right_ascension_name + "_scale",
            sin_declination_name + "_high",
            sin_declination_name + "_low",
            sin_declination_name + "_scale",
        ],
        err_json,
    )

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
            scale_Mc=err_param["scale_Mc"],
            scale_eta=err_param["scale_eta"],
        ),
    )

    if has_spin:
        parameters_name += (chi1_name, chi2_name)
        if args.spin_truncated_normal:
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
        else:
            all_params.extend(
                [
                    ("chi1_mean_g", N_g),
                    ("chi1_mean_pl", N_pl),
                    ("chi1_scale_g", N_g),
                    ("chi1_scale_pl", N_pl),
                    ("chi1_variance_g", N_g),
                    ("chi1_variance_pl", N_pl),
                    ("chi2_mean_g", N_g),
                    ("chi2_mean_pl", N_pl),
                    ("chi2_scale_g", N_g),
                    ("chi2_scale_pl", N_pl),
                    ("chi2_variance_g", N_g),
                    ("chi2_variance_pl", N_pl),
                ]
            )

        @error_magazine.register(chi1_name)
        def chi1_error_fn(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param["chi1_scale"],
                low=err_param.get("chi1_low"),
                high=err_param.get("chi1_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < 0.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

        @error_magazine.register(chi2_name)
        def chi2_error_fn(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param["chi2_scale"],
                low=err_param.get("chi2_low"),
                high=err_param.get("chi2_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < 0.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

    if has_tilt:
        parameters_name += (cos_tilt_1_name, cos_tilt_2_name)
        all_params.extend(
            [
                ("cos_tilt1_scale_g", N_g),
                ("cos_tilt1_scale_pl", N_pl),
                ("cos_tilt2_scale_g", N_g),
                ("cos_tilt2_scale_pl", N_pl),
            ]
        )

        @error_magazine.register(cos_tilt_1_name)
        def cos_tilt_1_error_fn(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param["cos_tilt_1_scale"],
                low=err_param.get("cos_tilt_1_low"),
                high=err_param.get("cos_tilt_1_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < -1.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

        @error_magazine.register(cos_tilt_2_name)
        def cos_tilt_2_error_fn(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param["cos_tilt_2_scale"],
                low=err_param.get("cos_tilt_2_low"),
                high=err_param.get("cos_tilt_2_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < -1.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
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

        @error_magazine.register(ecc_name)
        def ecc_error_fn(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param["ecc_err_scale"],
                low=err_param.get("ecc_err_low"),
                high=err_param.get("ecc_err_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < 0.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

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

        @error_magazine.register(redshift_name)
        def redshift_error_fn(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param["redshift_scale"],
                low=err_param.get("redshift_low"),
                high=err_param.get("redshift_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < 1e-3
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

    if has_cos_inclination:
        parameters_name += (cos_inclination_name,)

        all_params.extend(
            [
                (cos_inclination_name + "_high_g", N_g),
                (cos_inclination_name + "_high_pl", N_pl),
                (cos_inclination_name + "_loc_g", N_g),
                (cos_inclination_name + "_loc_pl", N_pl),
                (cos_inclination_name + "_low_g", N_g),
                (cos_inclination_name + "_low_pl", N_pl),
                (cos_inclination_name + "_scale_g", N_g),
                (cos_inclination_name + "_scale_pl", N_pl),
            ]
        )

        @error_magazine.register(cos_inclination_name)
        def cos_inclination_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param[cos_inclination_name + "_scale"],
                low=err_param.get(cos_inclination_name + "_low"),
                high=err_param.get(cos_inclination_name + "_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < -1.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

    if has_phi_12:
        parameters_name += (phi_12_name,)

        all_params.extend(
            [
                (phi_12_name + "_high_g", N_g),
                (phi_12_name + "_high_pl", N_pl),
                (phi_12_name + "_loc_g", N_g),
                (phi_12_name + "_loc_pl", N_pl),
                (phi_12_name + "_low_g", N_g),
                (phi_12_name + "_low_pl", N_pl),
                (phi_12_name + "_scale_g", N_g),
                (phi_12_name + "_scale_pl", N_pl),
            ]
        )

        @error_magazine.register(phi_12_name)
        def phi_12_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param[phi_12_name + "_scale"],
                low=err_param.get(phi_12_name + "_low"),
                high=err_param.get(phi_12_name + "_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < 0.0
            mask |= err_x > 2, 0 * jnp.pi
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

    if has_polarization_angle:
        parameters_name += (polarization_angle_name,)

        all_params.extend(
            [
                (polarization_angle_name + "_high_g", N_g),
                (polarization_angle_name + "_high_pl", N_pl),
                (polarization_angle_name + "_loc_g", N_g),
                (polarization_angle_name + "_loc_pl", N_pl),
                (polarization_angle_name + "_low_g", N_g),
                (polarization_angle_name + "_low_pl", N_pl),
                (polarization_angle_name + "_scale_g", N_g),
                (polarization_angle_name + "_scale_pl", N_pl),
            ]
        )

        @error_magazine.register(polarization_angle_name)
        def polarization_angle_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param[polarization_angle_name + "_scale"],
                low=err_param.get(polarization_angle_name + "_low"),
                high=err_param.get(polarization_angle_name + "_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < 0.0
            mask |= err_x > jnp.pi
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

    if has_right_ascension:
        parameters_name += (right_ascension_name,)

        all_params.extend(
            [
                (right_ascension_name + "_high_g", N_g),
                (right_ascension_name + "_high_pl", N_pl),
                (right_ascension_name + "_loc_g", N_g),
                (right_ascension_name + "_loc_pl", N_pl),
                (right_ascension_name + "_low_g", N_g),
                (right_ascension_name + "_low_pl", N_pl),
                (right_ascension_name + "_scale_g", N_g),
                (right_ascension_name + "_scale_pl", N_pl),
            ]
        )

        @error_magazine.register(right_ascension_name)
        def right_ascension_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param[right_ascension_name + "_scale"],
                low=err_param.get(right_ascension_name + "_low"),
                high=err_param.get(right_ascension_name + "_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < 0.0
            mask |= err_x > 2.0 * jnp.pi
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
            return err_x

    if has_sin_declination:
        parameters_name += (sin_declination_name,)

        all_params.extend(
            [
                (sin_declination_name + "_high_g", N_g),
                (sin_declination_name + "_high_pl", N_pl),
                (sin_declination_name + "_loc_g", N_g),
                (sin_declination_name + "_loc_pl", N_pl),
                (sin_declination_name + "_low_g", N_g),
                (sin_declination_name + "_low_pl", N_pl),
                (sin_declination_name + "_scale_g", N_g),
                (sin_declination_name + "_scale_pl", N_pl),
            ]
        )

        @error_magazine.register(sin_declination_name)
        def sin_declination_error(x, size, key):
            err_x = dist.TruncatedNormal(
                loc=x,
                scale=err_param[sin_declination_name + "_scale"],
                low=err_param.get(sin_declination_name + "_low"),
                high=err_param.get(sin_declination_name + "_high"),
            ).sample(key=key, sample_shape=(size,))
            mask = err_x < -1.0
            mask |= err_x > 1.0
            err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
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
        use_redshift=has_redshift,
        use_cos_inclination=has_cos_inclination,
        use_phi_12=has_phi_12,
        use_polarization_angle=has_polarization_angle,
        use_right_ascension=has_right_ascension,
        use_sin_declination=has_sin_declination,
        **model_param,
    )
    _constraint = partial(
        constraint,
        has_spin=has_spin,
        has_tilt=has_tilt,
        has_eccentricity=has_eccentricity,
        has_redshift=has_redshift,
        has_cos_inclination=has_cos_inclination,
        has_phi_12=has_phi_12,
        has_polarization_angle=has_polarization_angle,
        has_right_ascension=has_right_ascension,
        has_sin_declination=has_sin_declination,
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
        constraint=_constraint,
    )
    popfactory.produce()
