# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from typing import List, Tuple

import jax
import numpy as np
from jax import random as jrd
from loguru import logger

import gwkokab
from gwkokab.inference import Bake, poisson_likelihood
from gwkokab.models import NPowerlawMGaussian
from gwkokab.models.utils import create_truncated_normal_distributions
from gwkokab.parameters import (
    COS_IOTA,
    COS_TILT_1,
    COS_TILT_2,
    DETECTION_TIME,
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
from gwkokab.utils.tools import error_if
from kokab.utils import poisson_mean_parser, sage_parser
from kokab.utils.common import (
    expand_arguments,
    flowMC_default_parameters,
    get_posterior_data,
    get_processed_priors,
    read_json,
    vt_json_read_and_process,
)
from kokab.utils.flowMC_helper import flowMChandler


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser.get_parser(parser)

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
    model_group.add_argument(
        "--add-spin",
        action="store_true",
        help="Include spin parameters in the model.",
    )
    model_group.add_argument(
        "--add-tilt",
        action="store_true",
        help="Include tilt parameters in the model.",
    )
    model_group.add_argument(
        "--add-redshift",
        action="store_true",
        help="Include redshift parameters in the model.",
    )
    model_group.add_argument(
        "--add-eccentricity",
        action="store_true",
        help="Include eccentricity in the model.",
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
        "--spin-truncated-normal",
        action="store_true",
        help="Use truncated normal distributions for spin parameters.",
    )

    return parser


def main() -> None:
    r"""Main function of the script."""
    logger.warning(
        "If you have made any changes to any parameters, please make sure"
        " that the changes are reflected in scripts that generate plots.",
    )

    parser = make_parser()
    args = parser.parse_args()

    SEED = args.seed
    KEY = jrd.PRNGKey(SEED)
    KEY1, KEY2, KEY3, KEY4 = jrd.split(KEY, 4)
    POSTERIOR_REGEX = args.posterior_regex
    POSTERIOR_COLUMNS = args.posterior_columns

    N_pl = args.n_pl
    N_g = args.n_g

    has_spin = args.add_spin
    has_tilt = args.add_tilt
    has_eccentricity = args.add_eccentricity
    has_redshift = args.add_redshift
    has_cos_iota = args.add_cos_iota
    has_phi_12 = args.add_phi_12
    has_polarization_angle = args.add_polarization_angle
    has_right_ascension = args.add_right_ascension
    has_sin_declination = args.add_sin_declination
    has_detection_time = args.add_detection_time

    prior_dict = read_json(args.prior_json)

    all_params: List[Tuple[str, int]] = [
        ("alpha_pl", N_pl),
        ("beta_pl", N_pl),
        ("m1_loc_g", N_g),
        ("m2_loc_g", N_g),
        ("m1_scale_g", N_g),
        ("m2_scale_g", N_g),
        ("m1_low_g", N_g),
        ("m2_low_g", N_g),
        ("m1_high_g", N_g),
        ("m2_high_g", N_g),
        ("mmax_pl", N_pl),
        ("mmin_pl", N_pl),
    ]

    parameters = [PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE]

    if has_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE, SECONDARY_SPIN_MAGNITUDE])
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

    if has_tilt:
        parameters.extend([COS_TILT_1, COS_TILT_2])
        all_params.extend(
            [
                ("cos_tilt1_scale_g", N_g),
                ("cos_tilt1_scale_pl", N_pl),
                ("cos_tilt2_scale_g", N_g),
                ("cos_tilt2_scale_pl", N_pl),
            ]
        )

    if has_phi_12:
        parameters.append(PHI_12)

        all_params.extend(
            [
                (PHI_12.name + "_high_g", N_g),
                (PHI_12.name + "_high_pl", N_pl),
                (PHI_12.name + "_loc_g", N_g),
                (PHI_12.name + "_loc_pl", N_pl),
                (PHI_12.name + "_low_g", N_g),
                (PHI_12.name + "_low_pl", N_pl),
                (PHI_12.name + "_scale_g", N_g),
                (PHI_12.name + "_scale_pl", N_pl),
            ]
        )

    if has_eccentricity:
        parameters.append(ECCENTRICITY)
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

    if has_redshift:
        parameters.append(REDSHIFT)
        all_params.extend(
            [
                ("redshift_lamb_g", N_g),
                ("redshift_lamb_pl", N_pl),
                ("redshift_z_max_g", N_g),
                ("redshift_z_max_pl", N_pl),
            ]
        )

    if has_right_ascension:
        parameters.append(RIGHT_ASCENSION)

        all_params.extend(
            [
                (RIGHT_ASCENSION.name + "_high_g", N_g),
                (RIGHT_ASCENSION.name + "_high_pl", N_pl),
                (RIGHT_ASCENSION.name + "_loc_g", N_g),
                (RIGHT_ASCENSION.name + "_loc_pl", N_pl),
                (RIGHT_ASCENSION.name + "_low_g", N_g),
                (RIGHT_ASCENSION.name + "_low_pl", N_pl),
                (RIGHT_ASCENSION.name + "_scale_g", N_g),
                (RIGHT_ASCENSION.name + "_scale_pl", N_pl),
            ]
        )

    if has_sin_declination:
        parameters.append(SIN_DECLINATION)

        all_params.extend(
            [
                (SIN_DECLINATION.name + "_high_g", N_g),
                (SIN_DECLINATION.name + "_high_pl", N_pl),
                (SIN_DECLINATION.name + "_loc_g", N_g),
                (SIN_DECLINATION.name + "_loc_pl", N_pl),
                (SIN_DECLINATION.name + "_low_g", N_g),
                (SIN_DECLINATION.name + "_low_pl", N_pl),
                (SIN_DECLINATION.name + "_scale_g", N_g),
                (SIN_DECLINATION.name + "_scale_pl", N_pl),
            ]
        )

    if has_detection_time:
        parameters.append(DETECTION_TIME)

        all_params.extend(
            [
                (DETECTION_TIME.name + "_high_g", N_g),
                (DETECTION_TIME.name + "_high_pl", N_pl),
                (DETECTION_TIME.name + "_low_g", N_g),
                (DETECTION_TIME.name + "_low_pl", N_pl),
            ]
        )

    if has_cos_iota:
        parameters.append(COS_IOTA)

        all_params.extend(
            [
                (COS_IOTA.name + "_high_g", N_g),
                (COS_IOTA.name + "_high_pl", N_pl),
                (COS_IOTA.name + "_loc_g", N_g),
                (COS_IOTA.name + "_loc_pl", N_pl),
                (COS_IOTA.name + "_low_g", N_g),
                (COS_IOTA.name + "_low_pl", N_pl),
                (COS_IOTA.name + "_scale_g", N_g),
                (COS_IOTA.name + "_scale_pl", N_pl),
            ]
        )

    if has_polarization_angle:
        parameters.append(POLARIZATION_ANGLE)

        all_params.extend(
            [
                (POLARIZATION_ANGLE.name + "_high_g", N_g),
                (POLARIZATION_ANGLE.name + "_high_pl", N_pl),
                (POLARIZATION_ANGLE.name + "_loc_g", N_g),
                (POLARIZATION_ANGLE.name + "_loc_pl", N_pl),
                (POLARIZATION_ANGLE.name + "_low_g", N_g),
                (POLARIZATION_ANGLE.name + "_low_pl", N_pl),
                (POLARIZATION_ANGLE.name + "_scale_g", N_g),
                (POLARIZATION_ANGLE.name + "_scale_pl", N_pl),
            ]
        )

    all_params.append(("log_rate", N_pl + N_g))

    error_if(
        set(POSTERIOR_COLUMNS) != set(map(lambda p: p.name, parameters)),
        msg="The parameters in the posterior data do not match the parameters in the model.",
    )

    extended_params = []
    for params in all_params:
        extended_params.extend(expand_arguments(*params))

    model_prior_param = get_processed_priors(extended_params, prior_dict)

    model = Bake(NPowerlawMGaussian)(
        N_pl=N_pl,
        N_g=N_g,
        use_spin=has_spin,
        use_tilt=has_tilt,
        use_eccentricity=has_eccentricity,
        use_redshift=has_redshift,
        use_cos_iota=has_cos_iota,
        use_phi_12=has_phi_12,
        use_polarization_angle=has_polarization_angle,
        use_right_ascension=has_right_ascension,
        use_sin_declination=has_sin_declination,
        use_detection_time=has_detection_time,
        **model_prior_param,
    )

    nvt = vt_json_read_and_process([param.name for param in parameters], args.vt_json)

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(nvt, key=KEY4, **pmean_kwargs)

    data = get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS)
    data_shapes = [d.shape[0] for d in data]
    data = jax.device_put(np.concatenate(data, axis=0), may_alias=True)
    log_ref_priors = jax.device_put(np.zeros(data.shape[0]), may_alias=True)

    variables_index, priors, poisson_likelihood_fn = poisson_likelihood(
        model=model,
        stacked_data=data,
        stacked_log_ref_priors=log_ref_priors,
        ERate_fn=erate_estimator.__call__,
        data_shapes=data_shapes,
    )

    constants = model.constants

    constants["N_pl"] = N_pl
    constants["N_g"] = N_g
    constants["use_spin"] = int(has_spin)
    constants["use_tilt"] = int(has_tilt)
    constants["use_eccentricity"] = int(has_eccentricity)
    constants["use_redshift"] = int(has_redshift)
    constants["use_cos_iota"] = int(has_cos_iota)
    constants["use_phi_12"] = int(has_phi_12)
    constants["use_polarization_angle"] = int(has_polarization_angle)
    constants["use_right_ascension"] = int(has_right_ascension)
    constants["use_sin_declination"] = int(has_sin_declination)
    constants["use_detection_time"] = int(has_detection_time)

    with open("constants.json", "w") as f:
        json.dump(constants, f)

    with open("nf_samples_mapping.json", "w") as f:
        json.dump(variables_index, f)

    FLOWMC_HANDLER_KWARGS = read_json(args.flowMC_json)

    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["rng_key"] = KEY1
    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["key"] = KEY2

    N_CHAINS = FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_chains"]
    initial_position = priors.sample(KEY3, (N_CHAINS,))

    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["n_features"] = initial_position.shape[1]
    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_dim"] = initial_position.shape[1]

    FLOWMC_HANDLER_KWARGS["data_dump_kwargs"]["labels"] = list(model.variables.keys())

    FLOWMC_HANDLER_KWARGS = flowMC_default_parameters(**FLOWMC_HANDLER_KWARGS)

    if args.adam_optimizer:
        from flowMC.strategy.optimization import optimization_Adam

        adam_kwargs = read_json(args.adam_json)
        Adam_opt = optimization_Adam(**adam_kwargs)

        FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["strategies"] = [Adam_opt, "default"]

    handler = flowMChandler(
        logpdf=poisson_likelihood_fn,
        initial_position=initial_position,
        **FLOWMC_HANDLER_KWARGS,
    )

    handler.run(
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        file_prefix="n_pls_m_gs",
    )
