# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

from gwkokab.inference import analytical_likelihood
from gwkokab.parameters import Parameters
from kokab.ecc_matters.common import EccentricityMattersModel
from kokab.utils.flowMC_based import flowMC_arg_parser, FlowMCBased
from kokab.utils.monk import Monk, monk_arg_parser


class EccentricityMattersCore(Monk):
    @property
    def parameters(self) -> List[str]:
        return [
            Parameters.PRIMARY_MASS_SOURCE.value,
            Parameters.SECONDARY_MASS_SOURCE.value,
            Parameters.ECCENTRICITY.value,
        ]

    @property
    def model_parameters(self) -> List[str]:
        return ["log_rate", "alpha_m", "mmin", "mmax", "loc", "scale", "low", "high"]


class EccentricityMattersFMonk(EccentricityMattersCore, FlowMCBased):
    pass


def f_main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = monk_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    EccentricityMattersFMonk(
        likelihood_fn=analytical_likelihood,
        model=EccentricityMattersModel,
        data_filename=args.data_filename,
        seed=args.seed,
        prior_filename=args.prior_json,
        selection_fn_filename=args.vt_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        analysis_name="f_monk_ecc_matters",
        n_samples=args.n_samples,
        max_iter_mean=args.max_iter_mean,
        max_iter_cov=args.max_iter_cov,
        n_vi_steps=args.n_vi_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        minimum_mc_error=args.minimum_mc_error,
        n_checkpoints=args.n_checkpoints,
        n_max_steps=args.n_max_steps,
    ).run()
