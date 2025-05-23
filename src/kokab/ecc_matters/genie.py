# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

import numpy as np
from jax import numpy as jnp, random as jrd
from numpyro import distributions as dist

from gwkokab.errors import banana_error_m1_m2
from gwkokab.parameters import ECCENTRICITY, PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE
from gwkokab.poisson_mean import PoissonMean
from gwkokab.population import error_magazine, PopulationFactory
from kokab.ecc_matters.common import constraint, EccentricityMattersModel
from kokab.utils import genie_parser, poisson_mean_parser
from kokab.utils.common import vt_json_read_and_process
from kokab.utils.regex import match_all


m1_source = PRIMARY_MASS_SOURCE.name
m2_source = SECONDARY_MASS_SOURCE.name
ecc = ECCENTRICITY.name


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
        "maximum_mass",
        "minimum_mass",
    ]

    model_param = match_all(model_params_name, model_json)
    err_param = match_all(
        ["scale_Mc", "scale_eta", "loc", "scale", "low", "high"], err_json
    )

    model = EccentricityMattersModel(**model_param)

    error_magazine.register(
        (m1_source, m2_source),
        lambda x, size, key: banana_error_m1_m2(
            x,
            size,
            key,
            scale_Mc=err_param["scale_Mc"],
            scale_eta=err_param["scale_eta"],
            maximum_mass=err_param.get("maximum_mass"),
            minimum_mass=err_param.get("minimum_mass"),
        ),
    )

    @error_magazine.register(ecc)
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

    model_parameters = [m1_source, m2_source, ecc]

    nvt = vt_json_read_and_process(model_parameters, args.vt_json)
    logVT = nvt.get_mapped_logVT()

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(
        nvt,
        key=jrd.PRNGKey(np.random.randint(0, 2**32, dtype=np.uint32)),
        **pmean_kwargs,
    )

    popfactory = PopulationFactory(
        model=model,
        parameters=model_parameters,
        logVT_fn=logVT,
        ERate_fn=erate_estimator.__call__,
        num_realizations=args.num_realizations,
        error_size=args.error_size,
        constraint=constraint,
    )

    popfactory.produce()
