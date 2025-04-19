# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import jax
import numpy as np
from jax import numpy as jnp, random as jrd
from loguru import logger
from numpyro.distributions import Uniform

from gwkokab.inference import Bake, poisson_likelihood
from gwkokab.models import ChiEffMassRatioCorrelated
from gwkokab.parameters import (
    Parameter,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
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


class RedshiftReferencePrior(Uniform):
    def __init__(self, low=0.001, high=3.0 + 1e-6, *, validate_args=None):
        super(RedshiftReferencePrior, self).__init__(
            low, high, validate_args=validate_args
        )

    def log_prob(self, value):
        return (2.7 - 1) * jnp.log1p(value)


REDSHIFT = Parameter(name="redshift", prior=RedshiftReferencePrior())


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser.get_parser(parser)

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

    model_parameters = [
        "alpha",
        "beta",
        "gamma",
        "kappa",
        "lamb",
        "lambda_peak",
        "loc_m",
        "log10_sigma_eff_0",
        "mmax",
        "mmin",
        "mu_eff_0",
        "scale_m",
    ]

    model_prior_param = get_processed_priors(
        model_parameters,
        prior_dict,
    )

    model = Bake(ChiEffMassRatioCorrelated)(**model_prior_param)

    parameters = [
        PRIMARY_MASS_SOURCE,
        SECONDARY_MASS_SOURCE,
        PRIMARY_SPIN_MAGNITUDE,
        SECONDARY_SPIN_MAGNITUDE,
        REDSHIFT,
    ]
    error_if(
        set(POSTERIOR_COLUMNS) != set(map(lambda p: p.name, parameters)),
        msg="The parameters in the posterior data do not match the parameters in the model.",
    )

    nvt = vt_json_read_and_process([param.name for param in parameters], args.vt_json)

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(nvt, key=KEY4, **pmean_kwargs)

    data = get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS)
    data_shapes = [d.shape[0] for d in data]
    log_ref_priors = jax.device_put(
        np.vstack([REDSHIFT.prior.log_prob(d[..., 4]) for d in data]), may_alias=True
    )
    data = jax.device_put(np.concatenate(data, axis=0), may_alias=True)

    variables_index, priors, poisson_likelihood_fn = poisson_likelihood(
        model=model,
        stacked_data=data,
        stacked_log_ref_priors=log_ref_priors,
        ERate_fn=erate_estimator.__call__,
        data_shapes=data_shapes,
    )

    constants = model.constants

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
        file_prefix="chi_eff_q_relation",
    )
