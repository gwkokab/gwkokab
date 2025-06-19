# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import numpy as np
from jax import random as jrd
from loguru import logger

from gwkokab.inference import Bake, poisson_likelihood
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
from kokab.utils import poisson_mean_parser, sage_parser
from kokab.utils.common import (
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

    SEED = args.seed
    KEY = jrd.PRNGKey(SEED)
    KEY1, KEY2, KEY3, KEY4 = jrd.split(KEY, 4)
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
        model_parameters.extend(["kappa", "z_max"])

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

    parameters_name = [param.name for param in parameters]
    nvt = vt_json_read_and_process(parameters_name, args.vt_json)

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(nvt, key=KEY4, **pmean_kwargs)  # type: ignore[arg-type]
    del nvt

    data = get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS)
    log_ref_priors = [np.zeros(d.shape[:-1]) for d in data]

    variables_index, priors, poisson_likelihood_fn = poisson_likelihood(
        parameters_name=parameters_name,
        dist_builder=model,
        data=data,
        log_ref_priors=log_ref_priors,
        ERate_fn=erate_estimator.__call__,
    )

    constants = model.constants
    constants["use_spin"] = int(has_spin)
    constants["use_redshift"] = int(has_redshift)

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
        file_prefix="one_powerlaw_one_peak",
    )
