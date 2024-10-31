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


from __future__ import annotations

import json
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

import jax.numpy as jnp
from jax import random as jrd
from numpyro import distributions as dist
from numpyro.distributions import constraints

from gwkokab.debug import enable_debugging
from gwkokab.inference import Bake, flowMChandler, poisson_likelihood
from gwkokab.models import Wysocki2019MassModel
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import ECCENTRICITY, PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE

from ..utils import sage_parser
from ..utils.common import (
    flowMC_json_read_and_process,
    get_logVT,
    get_posterior_data,
    get_processed_priors,
)


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser.get_parser(parser)

    return parser


def EccentricityMattersModel(log_rate, alpha_m, mmin, mmax, scale) -> JointDistribution:
    model = JointDistribution(
        Wysocki2019MassModel(alpha_m=alpha_m, mmin=mmin, mmax=mmax),
        dist.HalfNormal(scale=scale, validate_args=True),
    )
    return ScaledMixture(
        log_scales=jnp.array([log_rate]),
        component_distributions=[model],
        support=constraints.real_vector,
        validate_args=True,
    )


def main() -> None:
    r"""Main function of the script."""
    # raise DeprecationWarning("This script is deprecated. Use `n_pls_m_gs` instead.")
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
    VT_FILENAME = args.vt_path
    VT_PARAMS = [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name]
    ANALYSIS_TIME = args.analysis_time

    FLOWMC_HANDLER_KWARGS = flowMC_json_read_and_process(args.flowMC_json)

    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["rng_key"] = KEY1
    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["key"] = KEY2

    FLOWMC_HANDLER_KWARGS["data"] = get_posterior_data(
        glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS
    )

    with open(args.prior_json, "r") as f:
        prior_dict = json.load(f)

    model_prior_param = get_processed_priors(
        ["log_rate", "alpha_m", "mmin", "mmax", "scale"],
        prior_dict,
    )

    model = Bake(EccentricityMattersModel)(**model_prior_param)

    poisson_likelihood.logVT = get_logVT(
        vt_path=VT_FILENAME,
        vt_params=VT_PARAMS,
        model_params=[
            PRIMARY_MASS_SOURCE.name,
            SECONDARY_MASS_SOURCE.name,
            ECCENTRICITY.name,
        ],
        key=KEY4,
        use_pdet=args.use_pdet,
    )
    poisson_likelihood.time = ANALYSIS_TIME
    poisson_likelihood.vt_method = "model"
    poisson_likelihood.vt_params = VT_PARAMS

    poisson_likelihood.set_model(
        (PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE, ECCENTRICITY),
        model=model,
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
