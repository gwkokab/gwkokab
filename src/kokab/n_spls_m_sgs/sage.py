# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from typing import List, Tuple

import jax
import numpy as np
from jax import random as jrd

import gwkokab
from gwkokab.inference import Bake, PoissonLikelihood
from gwkokab.models import NSmoothedPowerlawMSmoothedGaussian
from gwkokab.models.utils import (
    create_smoothed_gaussians_raw,
    create_smoothed_powerlaws_raw,
    create_truncated_normal_distributions,
)
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    ECCENTRICITY,
    MASS_RATIO,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    REDSHIFT,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)
from gwkokab.poisson_mean import PoissonMean
from gwkokab.utils.tools import error_if
from kokab.utils import flowmc_driver, poisson_mean_parser, sage_parser
from kokab.utils.common import (
    expand_arguments,
    get_posterior_data,
    get_processed_priors,
    read_json,
    vt_json_read_and_process,
)


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
        "--spin-truncated-normal",
        action="store_true",
        help="Use truncated normal distributions for spin parameters.",
    )
    model_group.add_argument(
        "--raw",
        action="store_true",
        help="The raw parameters for this model are primary mass and mass ratio. To"
        "align with the rest of the codebase, we transform primary mass and mass ratio"
        "to primary and secondary mass. This flag will use the raw parameters i.e."
        "primary mass and mass ratio.",
    )

    return parser


def main() -> None:
    r"""Main function of the script."""
    warnings.warn(
        "If you have made any changes to any parameters, please make sure"
        " that the changes are reflected in scripts that generate plots.",
        Warning,
    )

    parser = make_parser()
    args = parser.parse_args()

    rng_key, pmean_key = jrd.split(jrd.PRNGKey(args.seed))

    POSTERIOR_REGEX = args.posterior_regex
    POSTERIOR_COLUMNS = args.posterior_columns

    N_pl = args.n_pl
    N_g = args.n_g

    has_spin = args.add_spin
    has_tilt = args.add_tilt
    has_eccentricity = args.add_eccentricity
    has_redshift = args.add_redshift

    prior_dict = read_json(args.prior_json)

    all_params: List[Tuple[str, int]] = [
        ("alpha_pl", N_pl),
        ("beta_pl", N_pl),
        ("mmax_pl", N_pl),
        ("mmin_pl", N_pl),
        ("delta_pl", N_pl),
        ("log_rate", N_pl + N_g),
        ("lamb_scale_g", N_g),
        ("lamb_scale_pl", N_pl),
        ("loc_g", N_g),
        ("scale_g", N_g),
        ("beta_g", N_g),
        ("mmin_g", N_g),
        ("mmax_g", N_g),
        ("delta_g", N_g),
    ]

    parameters = [PRIMARY_MASS_SOURCE]

    if args.raw:
        gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_powerlaw_distributions = create_smoothed_powerlaws_raw
        gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_gaussian_distributions = create_smoothed_gaussians_raw
        parameters.append(MASS_RATIO)
    else:
        parameters.append(SECONDARY_MASS_SOURCE)

    if has_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE, SECONDARY_SPIN_MAGNITUDE])
        if args.spin_truncated_normal:
            gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_spin_distributions = create_truncated_normal_distributions
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

    error_if(
        set(POSTERIOR_COLUMNS) != set(map(lambda p: p.name, parameters)),
        msg="The parameters in the posterior data do not match the parameters in the model.",
    )

    extended_params = []
    for params in all_params:
        extended_params.extend(expand_arguments(*params))

    model_prior_param = get_processed_priors(extended_params, prior_dict)

    model = Bake(NSmoothedPowerlawMSmoothedGaussian)(
        N_pl=N_pl,
        N_g=N_g,
        use_spin=has_spin,
        use_tilt=has_tilt,
        use_eccentricity=has_eccentricity,
        use_redshift=has_redshift,
        **model_prior_param,
    )

    nvt = vt_json_read_and_process([param.name for param in parameters], args.vt_json)

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(nvt, key=pmean_key, **pmean_kwargs)

    data = get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS)
    log_ref_priors = jax.device_put(
        [np.zeros(d.shape[:-1]) for d in data], may_alias=True
    )

    poisson_likelihood = PoissonLikelihood(
        model=model,
        log_ref_priors=log_ref_priors,
        data=data,
        ERate_fn=erate_estimator.__call__,
    )

    constants = model.constants

    constants["N_pl"] = N_pl
    constants["N_g"] = N_g
    constants["use_spin"] = int(has_spin)
    constants["use_tilt"] = int(has_tilt)
    constants["use_eccentricity"] = int(has_eccentricity)
    constants["use_redshift"] = int(has_redshift)

    with open("constants.json", "w") as f:
        json.dump(constants, f)

    with open("nf_samples_mapping.json", "w") as f:
        json.dump(poisson_likelihood.variables_index, f)

    flowmc_driver.run_flowMC(
        rng_key=rng_key,
        flowMC_config_path=args.flowMC_json,
        likelihood=poisson_likelihood,
        labels=list(model.variables.keys()),
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        file_prefix="n_spls_m_sgs",
        verbose=args.verbose,
    )
