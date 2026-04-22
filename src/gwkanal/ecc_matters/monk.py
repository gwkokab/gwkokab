# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from numpyro.distributions.distribution import enable_validation

from gwkanal.core.flowMC_based import FlowMCBased
from gwkanal.core.inference_io import AnalyticalPELoader as DataLoader
from gwkanal.core.monk import Monk, monk_arg_parser
from gwkanal.core.numpyro_based import NumpyroBased
from gwkanal.ecc_matters.common import EccentricityMattersCore, EccentricityMattersModel
from gwkanal.utils.logger import log_info
from gwkokab.inference import (
    flowMC_analytical_poisson_likelihood,
    numpyro_analytical_poisson_likelihood,
)


class EccentricityMattersFMonk(EccentricityMattersCore, Monk, FlowMCBased):
    pass


class EccentricityMattersNMonk(EccentricityMattersCore, Monk, NumpyroBased):
    pass


def f_main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    EccentricityMattersFMonk.init_rng_seed(seed=args.seed)

    EccentricityMattersFMonk(
        likelihood_fn=flowMC_analytical_poisson_likelihood,
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
        variance_cut_threshold=args.variance_cut_threshold,
    ).run()


def n_main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    EccentricityMattersNMonk.init_rng_seed(seed=args.seed)

    EccentricityMattersNMonk(
        likelihood_fn=numpyro_analytical_poisson_likelihood,
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
        variance_cut_threshold=args.variance_cut_threshold,
    ).run()
