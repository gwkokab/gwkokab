# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import jax
import numpy as np
from jax import random as jrd
from loguru import logger

from gwkokab.inference import Bake, PoissonLikelihood
from gwkokab.models import SmoothedPowerlawAndPeak
from gwkokab.parameters import (
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
        "--add-spin",
        action="store_true",
        help="Include spin parameters in the model.",
    )
    model_group.add_argument(
        "--add-redshift",
        action="store_true",
        help="Include redshift parameters in the model.",
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

    rng_key, pmean_key = jrd.split(jrd.PRNGKey(args.seed))

    POSTERIOR_REGEX = args.posterior_regex
    POSTERIOR_COLUMNS = args.posterior_columns

    prior_dict = read_json(args.prior_json)

    has_spin = args.add_spin
    has_redshift = args.add_redshift

    model_parameters = [
        "alpha",
        "beta",
        "delta",
        "lambda_peak",
        "loc",
        "log_rate",
        "mmax",
        "mmin",
        "scale",
    ]

    parameters = [PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE]

    if has_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE, SECONDARY_SPIN_MAGNITUDE])
        model_parameters.extend(
            [
                "chi1_high_g",
                "chi1_high_pl",
                "chi1_loc_g",
                "chi1_loc_pl",
                "chi1_low_g",
                "chi1_low_pl",
                "chi1_scale_g",
                "chi1_scale_pl",
                "chi2_high_g",
                "chi2_high_pl",
                "chi2_loc_g",
                "chi2_loc_pl",
                "chi2_low_g",
                "chi2_low_pl",
                "chi2_scale_g",
                "chi2_scale_pl",
            ]
        )

    if has_redshift:
        parameters.append(REDSHIFT)
        model_parameters.extend(["lamb", "z_max"])

    error_if(
        set(POSTERIOR_COLUMNS) != set(map(lambda p: p.name, parameters)),
        msg="The parameters in the posterior data do not match the parameters in the model.",
    )

    model_prior_param = get_processed_priors(model_parameters, prior_dict)

    model = Bake(SmoothedPowerlawAndPeak)(
        use_spin=has_spin,
        use_redshift=has_redshift,
        **model_prior_param,
    )

    nvt = vt_json_read_and_process([param.name for param in parameters], args.vt_json)

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(nvt, key=pmean_key, **pmean_kwargs)
    del nvt

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
    constants["use_spin"] = int(has_spin)
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
        file_prefix="one_powerlaw_one_peak",
        verbose=args.verbose,
    )
