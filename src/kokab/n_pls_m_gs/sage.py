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

from jax import random as jrd

from gwkokab.debug import enable_debugging
from gwkokab.inference import Bake, flowMChandler, poisson_likelihood
from gwkokab.models import (
    NPowerLawMGaussian,
    NPowerLawMGaussianWithDefaultSpinMagnitude,
    NPowerLawMGaussianWithDefaultSpinMagnitudeAndSpinMisalignment,
    NPowerLawMGaussianWithSpinMisalignment,
)
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)

from ..utils import sage_parser
from ..utils.common import (
    expand_arguments,
    flowMC_json_read_and_process,
    get_posterior_data,
    get_processed_priors,
)
from .common import get_logVT


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
        "--no-spin",
        action="store_true",
        help="Do not include spin parameters in the model.",
    )
    model_group.add_argument(
        "--no-tilt",
        action="store_true",
        help="Do not include tilt parameters in the model.",
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

    if args.verbose:
        enable_debugging()

    SEED = args.seed
    KEY = jrd.PRNGKey(SEED)
    KEY1, KEY2, KEY3 = jrd.split(KEY, 3)
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

    N_pl = args.n_pl
    N_g = args.n_g

    has_spin = not args.no_spin
    has_tilt = not args.no_tilt

    with open(args.prior_json, "r") as f:
        prior_dict = json.load(f)

    if has_spin and has_tilt:
        parameters = (
            PRIMARY_MASS_SOURCE,
            SECONDARY_MASS_SOURCE,
            PRIMARY_SPIN_MAGNITUDE,
            SECONDARY_SPIN_MAGNITUDE,
            COS_TILT_1,
            COS_TILT_2,
        )
        model_prior_param = get_processed_priors(
            expand_arguments("alpha", N_pl)
            + expand_arguments("beta", N_pl)
            + expand_arguments("mmin", N_pl)
            + expand_arguments("mmax", N_pl)
            + expand_arguments("mean_chi1_pl", N_pl)
            + expand_arguments("mean_chi2_pl", N_pl)
            + expand_arguments("std_dev_tilt1_pl", N_pl)
            + expand_arguments("std_dev_tilt2_pl", N_pl)
            + expand_arguments("variance_chi1_pl", N_pl)
            + expand_arguments("variance_chi2_pl", N_pl)
            + expand_arguments("loc_m1", N_g)
            + expand_arguments("loc_m2", N_g)
            + expand_arguments("scale_m1", N_g)
            + expand_arguments("scale_m2", N_g)
            + expand_arguments("mean_chi1_g", N_g)
            + expand_arguments("mean_chi2_g", N_g)
            + expand_arguments("std_dev_tilt1_g", N_g)
            + expand_arguments("std_dev_tilt2_g", N_g)
            + expand_arguments("variance_chi1_g", N_g)
            + expand_arguments("variance_chi2_g", N_g),
            prior_dict,
        )

        model = Bake(NPowerLawMGaussianWithDefaultSpinMagnitudeAndSpinMisalignment)(
            N_pl=N_pl,
            N_g=N_g,
            **model_prior_param,
        )
    elif has_spin:
        parameters = (
            PRIMARY_MASS_SOURCE,
            SECONDARY_MASS_SOURCE,
            PRIMARY_SPIN_MAGNITUDE,
            SECONDARY_SPIN_MAGNITUDE,
        )
        model_prior_param = get_processed_priors(
            expand_arguments("alpha", N_pl)
            + expand_arguments("beta", N_pl)
            + expand_arguments("mmin", N_pl)
            + expand_arguments("mmax", N_pl)
            + expand_arguments("mean_chi1_pl", N_pl)
            + expand_arguments("mean_chi2_pl", N_pl)
            + expand_arguments("variance_chi1_pl", N_pl)
            + expand_arguments("variance_chi2_pl", N_pl)
            + expand_arguments("loc_m1", N_g)
            + expand_arguments("loc_m2", N_g)
            + expand_arguments("scale_m1", N_g)
            + expand_arguments("scale_m2", N_g)
            + expand_arguments("mean_chi1_g", N_g)
            + expand_arguments("mean_chi2_g", N_g)
            + expand_arguments("variance_chi1_g", N_g)
            + expand_arguments("variance_chi2_g", N_g),
            prior_dict,
        )

        model = Bake(NPowerLawMGaussianWithDefaultSpinMagnitude)(
            N_pl=N_pl,
            N_g=N_g,
            **model_prior_param,
        )
    elif has_tilt:
        parameters = (
            PRIMARY_MASS_SOURCE,
            SECONDARY_MASS_SOURCE,
            COS_TILT_1,
            COS_TILT_2,
        )
        model_prior_param = get_processed_priors(
            expand_arguments("alpha", N_pl)
            + expand_arguments("beta", N_pl)
            + expand_arguments("mmin", N_pl)
            + expand_arguments("mmax", N_pl)
            + expand_arguments("std_dev_tilt1_pl", N_pl)
            + expand_arguments("std_dev_tilt2_pl", N_pl)
            + expand_arguments("loc_m1", N_g)
            + expand_arguments("loc_m2", N_g)
            + expand_arguments("scale_m1", N_g)
            + expand_arguments("scale_m2", N_g)
            + expand_arguments("std_dev_tilt1_g", N_g)
            + expand_arguments("std_dev_tilt2_g", N_g),
            prior_dict,
        )

        model = Bake(NPowerLawMGaussianWithSpinMisalignment)(
            N_pl=N_pl,
            N_g=N_g,
            **model_prior_param,
        )
    else:
        parameters = (PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE)
        model_prior_param = get_processed_priors(
            expand_arguments("alpha", N_pl)
            + expand_arguments("beta", N_pl)
            + expand_arguments("mmin", N_pl)
            + expand_arguments("mmax", N_pl)
            + expand_arguments("loc_m1", N_g)
            + expand_arguments("loc_m2", N_g)
            + expand_arguments("scale_m1", N_g)
            + expand_arguments("scale_m2", N_g),
            prior_dict,
        )

        model = Bake(NPowerLawMGaussian)(
            N_pl=N_pl,
            N_g=N_g,
            **model_prior_param,
        )

    log_rate_prior_param = get_processed_priors(
        expand_arguments("log_rate", N_pl + N_g), prior_dict
    )

    poisson_likelihood.is_multi_rate_model = True
    poisson_likelihood.logVT = get_logVT(VT_FILENAME)
    poisson_likelihood.time = ANALYSIS_TIME
    poisson_likelihood.vt_method = "model"
    poisson_likelihood.vt_params = VT_PARAMS

    poisson_likelihood.set_model(
        parameters,
        [log_rate_prior_param[r] for r in expand_arguments("log_rate", N_pl + N_g)],
        model=model,
    )

    N_CHAINS = FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_chains"]
    initial_position = poisson_likelihood.priors.sample(KEY3, (N_CHAINS,))

    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["n_features"] = initial_position.shape[1]
    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_dim"] = initial_position.shape[1]

    FLOWMC_HANDLER_KWARGS["data_dump_kwargs"]["labels"] = expand_arguments(
        "log_rate", N_pl + N_g
    ) + list(model.variables.keys())

    handler = flowMChandler(
        logpdf=poisson_likelihood.log_posterior,
        initial_position=initial_position,
        **FLOWMC_HANDLER_KWARGS,
    )

    handler.run()
