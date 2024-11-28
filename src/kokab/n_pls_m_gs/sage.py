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
from typing_extensions import List, Tuple

from jax import random as jrd
from jaxtyping import Int

from gwkokab.debug import enable_debugging
from gwkokab.inference import Bake, flowMChandler, PoissonLikelihood
from gwkokab.models import NPowerlawMGaussian
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    ECCENTRICITY,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    REDSHIFT,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)

from ..utils import sage_parser
from ..utils.common import (
    expand_arguments,
    flowMC_json_read_and_process,
    get_posterior_data,
    get_processed_priors,
    log_weights_and_samples,
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
        "--no-spin",
        action="store_true",
        help="Do not include spin parameters in the model.",
    )
    model_group.add_argument(
        "--no-tilt",
        action="store_true",
        help="Do not include tilt parameters in the model.",
    )
    model_group.add_argument(
        "--no-redshift",
        action="store_true",
        help="Do not include redshift parameters in the model.",
    )
    model_group.add_argument(
        "--no-eccentricity",
        action="store_true",
        help="Do not include eccentricity in the model.",
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
    KEY1, KEY2, KEY3, KEY4 = jrd.split(KEY, 4)
    POSTERIOR_REGEX = args.posterior_regex
    POSTERIOR_COLUMNS = args.posterior_columns
    VT_FILENAME = args.vt_path
    ANALYSIS_TIME = args.analysis_time

    FLOWMC_HANDLER_KWARGS = flowMC_json_read_and_process(args.flowMC_json)

    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["rng_key"] = KEY1
    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["key"] = KEY2

    N_pl = args.n_pl
    N_g = args.n_g

    has_spin = not args.no_spin
    has_tilt = not args.no_tilt
    has_eccentricity = not args.no_eccentricity
    has_redshift = not args.no_redshift

    with open(args.prior_json, "r") as f:
        prior_dict = json.load(f)

    all_params: List[Tuple[str, Int[int, "N_pl", "N_g"]]] = [
        ("alpha_pl", N_pl),
        ("beta_pl", N_pl),
        ("m1_loc_g", N_g),
        ("m2_loc_g", N_g),
        ("m1_scale_g", N_g),
        ("m2_scale_g", N_g),
        ("mmax_pl", N_pl),
        ("mmin_pl", N_pl),
    ]

    parameters = [PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE]

    if has_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE, SECONDARY_SPIN_MAGNITUDE])
        all_params.extend(
            [
                ("chi1_alpha_g", N_g),
                ("chi1_alpha_pl", N_pl),
                ("chi1_beta_g", N_g),
                ("chi1_beta_pl", N_pl),
                ("chi2_alpha_g", N_g),
                ("chi2_alpha_pl", N_pl),
                ("chi2_beta_g", N_g),
                ("chi2_beta_pl", N_pl),
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

    all_params.append(("log_rate", N_pl + N_g))

    extended_params = []
    for params in all_params:
        extended_params.extend(expand_arguments(*params))

    model_prior_param = get_processed_priors(extended_params, prior_dict)

    model = Bake(NPowerlawMGaussian)(
        N_pl=N_pl,
        N_g=N_g,
        use_spin=has_spin,
        use_tilt=has_tilt,
        use_eccentricity=has_eccentricity,
        use_redshift=has_redshift,
        **model_prior_param,
    )

    log_weight, samples = log_weights_and_samples(
        KEY4, parameters, VT_FILENAME, args.vt_n_samples
    )

    poisson_likelihood = PoissonLikelihood(
        model=model,
        parameters=parameters,
        data=get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS),
        log_weights=log_weight,
        samples=samples,
        time=ANALYSIS_TIME,
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
