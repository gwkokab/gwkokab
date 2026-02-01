# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

from gwkanal.core.monk import Monk, monk_arg_parser
from gwkanal.ecc_matters.common import EccentricityMattersModel
from gwkanal.utils.logger import log_info
from gwkokab.parameters import Parameters as P


class EccentricityMattersMonk(Monk):
    @property
    def parameters(self) -> List[str]:
        return [P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.ECCENTRICITY]

    @property
    def model_parameters(self) -> List[str]:
        return ["log_rate", "alpha_m", "mmin", "mmax", "loc", "scale", "low", "high"]


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    EccentricityMattersMonk(
        model=EccentricityMattersModel,
        data_filename=args.data_filename,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_config,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        analysis_name="ecc_matters",
        n_samples=args.n_samples,
        minimum_mc_error=args.minimum_mc_error,
        n_checkpoints=args.n_checkpoints,
        n_max_steps=args.n_max_steps,
    ).run()
