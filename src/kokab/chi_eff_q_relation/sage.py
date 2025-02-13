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

from jax import numpy as jnp, random as jrd
from numpyro.distributions import Uniform

from gwkokab.inference import Bake, flowMChandler, PoissonLikelihood
from gwkokab.logger import enable_logging
from gwkokab.models import ChiEffMassRatioCorrelated
from gwkokab.parameters import (
    Parameter,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)
from gwkokab.poisson_mean import PoissonMean
from kokab.utils import poisson_mean_parser, sage_parser
from kokab.utils.common import (
    flowMC_default_parameters,
    get_posterior_data,
    get_processed_priors,
    read_json,
    vt_json_read_and_process,
)


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
    warnings.warn(
        "If you have made any changes to any parameters, please make sure"
        " that the changes are reflected in scripts that generate plots.",
        Warning,
    )

    parser = make_parser()
    args = parser.parse_args()

    if args.verbose:
        enable_logging()

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

    nvt = vt_json_read_and_process(
        [param.name for param in parameters], args.vt_path, args.vt_json
    )
    logVT = nvt.get_mapped_logVT()

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(logVT, KEY4, **pmean_kwargs)

    data = get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS)
    log_ref_priors = [REDSHIFT.prior.log_prob(d[..., 4]) for d in data]

    poisson_likelihood = PoissonLikelihood(
        model=model,
        log_ref_priors=log_ref_priors,
        data=data,
        ERate_fn=erate_estimator.__call__,
    )

    constants = model.constants

    with open("constants.json", "w") as f:
        json.dump(constants, f)

    with open("nf_samples_mapping.json", "w") as f:
        json.dump(poisson_likelihood.variables_index, f)

    FLOWMC_HANDLER_KWARGS = read_json(args.flowMC_json)

    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["rng_key"] = KEY1
    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["key"] = KEY2

    N_CHAINS = FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_chains"]
    initial_position = poisson_likelihood.priors.sample(KEY3, (N_CHAINS,))

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
        logpdf=poisson_likelihood.log_posterior,
        initial_position=initial_position,
        **FLOWMC_HANDLER_KWARGS,
    )

    handler.run(args.debug_nans)
