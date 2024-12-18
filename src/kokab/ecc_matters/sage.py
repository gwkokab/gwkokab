# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import numpy as np
from jax import random as jrd

from gwkokab.debug import enable_debugging
from gwkokab.inference import Bake, flowMChandler, PoissonLikelihood
from gwkokab.parameters import ECCENTRICITY, PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE
from gwkokab.poisson_mean import (
    ImportanceSamplingPoissonMean,
    InverseTransformSamplingPoissonMean,
)

from ..utils import sage_parser
from ..utils.common import (
    flowMC_json_read_and_process,
    get_posterior_data,
    get_processed_priors,
    vt_json_read_and_process,
)
from .common import EccentricityMattersModel


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser.get_parser(parser)

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

    if args.verbose:
        enable_debugging()

    SEED = args.seed
    KEY = jrd.PRNGKey(SEED)
    KEY1, KEY2, KEY3, KEY4 = jrd.split(KEY, 4)
    POSTERIOR_REGEX = args.posterior_regex
    POSTERIOR_COLUMNS = args.posterior_columns

    FLOWMC_HANDLER_KWARGS = flowMC_json_read_and_process(args.flowMC_json)

    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["rng_key"] = KEY1
    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["key"] = KEY2

    FLOWMC_HANDLER_KWARGS["data"] = {
        **get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS)
    }

    with open(args.prior_json, "r") as f:
        prior_dict = json.load(f)

    model_prior_param = get_processed_priors(
        ["log_rate", "alpha_m", "mmin", "mmax", "loc", "scale", "low", "high"],
        prior_dict,
    )

    model = Bake(EccentricityMattersModel)(**model_prior_param)

    parameters = [PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE, ECCENTRICITY]

    nvt = vt_json_read_and_process(
        [param.name for param in parameters], args.vt_path, args.vt_json
    )
    logVT = nvt.get_vmapped_logVT()

    if args.erate_estimator == "IS":
        erate_estimator = ImportanceSamplingPoissonMean(
            logVT,
            [PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE, ECCENTRICITY],
            jrd.PRNGKey(np.random.randint(0, 2**32, dtype=np.uint32)),
            args.n_samples,
            args.analysis_time,
        )
    elif args.erate_estimator == "ITS":
        erate_estimator = InverseTransformSamplingPoissonMean(
            logVT,
            jrd.PRNGKey(np.random.randint(0, 2**32, dtype=np.uint32)),
            args.n_samples,
            args.analysis_time,
        )
    else:
        raise ValueError("Invalid estimator for expected rate.")

    poisson_likelihood = PoissonLikelihood(
        model=model,
        parameters=parameters,
        data=get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS),
        ERate_fn=erate_estimator.__call__,
    )

    N_CHAINS = FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_chains"]
    initial_position = poisson_likelihood.priors.sample(KEY3, (N_CHAINS,))

    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["n_features"] = initial_position.shape[1]
    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_dim"] = initial_position.shape[1]

    FLOWMC_HANDLER_KWARGS["data_dump_kwargs"]["labels"] = list(model.variables.keys())

    handler = flowMChandler(
        logpdf=poisson_likelihood.log_posterior,
        initial_position=initial_position,
        **FLOWMC_HANDLER_KWARGS,
    )

    handler.run()
