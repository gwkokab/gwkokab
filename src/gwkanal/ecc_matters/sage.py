# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from numpyro.distributions.distribution import enable_validation

from gwkanal.core.flowMC_based import flowMC_arg_parser, FlowMCBased
from gwkanal.core.inference_io import DiscretePELoader as DataLoader
from gwkanal.core.numpyro_based import numpyro_arg_parser, NumpyroBased
from gwkanal.core.sage import Sage, sage_arg_parser as sage_parser
from gwkanal.ecc_matters.common import EccentricityMattersCore, EccentricityMattersModel
from gwkanal.utils.logger import log_info
from gwkokab.inference import flowMC_poisson_likelihood, numpyro_poisson_likelihood


class EccentricityMattersFSage(EccentricityMattersCore, Sage, FlowMCBased):
    pass


class EccentricityMattersNSage(EccentricityMattersCore, Sage, NumpyroBased):
    pass


def f_main() -> None:
    enable_validation()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    EccentricityMattersFSage.init_rng_seed(seed=args.seed)

    EccentricityMattersFSage(
        likelihood_fn=flowMC_poisson_likelihood,
        model=EccentricityMattersModel,
        data_loader=data_loader,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_config,
        variance_cut_threshold=args.variance_cut_threshold,
        analysis_name="f_sage_ecc_matters",
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        where_fns=None,
    ).run()


def n_main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser(parser)
    parser = numpyro_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)
    EccentricityMattersNSage.init_rng_seed(seed=args.seed)

    EccentricityMattersNSage(
        likelihood_fn=numpyro_poisson_likelihood,
        model=EccentricityMattersModel,
        data_loader=data_loader,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_config,
        variance_cut_threshold=args.variance_cut_threshold,
        analysis_name="n_sage_ecc_matters",
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        where_fns=None,
    ).run()
