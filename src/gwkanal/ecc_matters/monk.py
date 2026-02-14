# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from gwkanal.core.inference_io import AnalyticalPELoader as DataLoader
from gwkanal.core.monk import Monk, monk_arg_parser
from gwkanal.ecc_matters.common import EccentricityMattersCore, EccentricityMattersModel
from gwkanal.utils.logger import log_info


class EccentricityMattersMonk(EccentricityMattersCore, Monk):
    pass


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    EccentricityMattersMonk.init_rng_seed(seed=args.seed)

    EccentricityMattersMonk(
        model=EccentricityMattersModel,
        data_loader=data_loader,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_config,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        analysis_name="ecc_matters",
        n_samples=args.n_samples,
        n_mom_samples=args.n_mom_samples,
        max_iter_mean=args.max_iter_mean,
        max_iter_cov=args.max_iter_cov,
    ).run()
