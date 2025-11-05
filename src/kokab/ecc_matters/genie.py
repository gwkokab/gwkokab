# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

from jax import numpy as jnp, random as jrd
from loguru import logger
from numpyro import distributions as dist

from gwkokab.errors import banana_error_m1_m2
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean import get_selection_fn_and_poisson_mean_estimator
from kokab.core.population import error_magazine, PopulationFactory
from kokab.ecc_matters.common import EccentricityMattersModel
from kokab.utils import genie_parser
from kokab.utils.common import read_json
from kokab.utils.logger import log_info
from kokab.utils.regex import match_all


def make_parser() -> ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = genie_parser.get_parser(parser)
    parser.description = "Generate a population of CBCs"
    parser.epilog = "This script generates a population of CBCs"

    model_group = parser.add_argument_group("Model Options")

    model_group.add_argument(
        "--model-json",
        help="Path to the JSON file containing the model parameters",
        type=str,
        required=True,
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

    model_params_name: List[str] = [
        "alpha_m",
        "high",
        "loc",
        "log_rate",
        "low",
        "mmax",
        "mmin",
        "scale",
    ]

    model_param = match_all(model_params_name, model_json)
    err_param = match_all(
        ["scale_Mc", "scale_eta", "loc", "scale", "low", "high"], err_json
    )

    error_magazine.register(
        (P.PRIMARY_MASS_SOURCE.value, P.SECONDARY_MASS_SOURCE.value),
        lambda x, size, key: banana_error_m1_m2(
            x,
            size,
            key,
            scale_Mc=err_param["scale_Mc"],
            scale_eta=err_param["scale_eta"],
        ),
    )

    @error_magazine.register(P.ECCENTRICITY.value)
    def ecc_error_fn(x, size, key):
        err_x = dist.TruncatedNormal(
            loc=x,
            scale=err_param["scale"],
            low=err_param["low"],
            high=err_param["high"],
        ).sample(key=key, sample_shape=(size,))
        mask = err_x < 0.0
        mask |= err_x > 1.0
        err_x = jnp.where(mask, jnp.full_like(mask, jnp.nan), err_x)
        return err_x

    parameters_name = [
        P.PRIMARY_MASS_SOURCE.value,
        P.SECONDARY_MASS_SOURCE.value,
        P.ECCENTRICITY.value,
    ]

    logger.info(f"Setting the random number generator key with seed {args.seed}.")
    pmean_key, factory_key = jrd.split(jrd.PRNGKey(args.seed), 2)

    pmean_config = read_json(args.pmean_json)
    log_selection_fn, erate_estimator, _, _ = (
        get_selection_fn_and_poisson_mean_estimator(
            key=pmean_key, parameters=parameters_name, **pmean_config
        )
    )

    popfactory = PopulationFactory(
        model_fn=EccentricityMattersModel,
        model_params=model_param,
        parameters=parameters_name,
        log_selection_fn=log_selection_fn,
        poisson_mean_estimator=erate_estimator,
        num_realizations=args.num_realizations,
        error_size=args.error_size,
        tile_covariance=args.tile_covariance,
    )

    popfactory.produce(factory_key)
