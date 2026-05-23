# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from numpyro.distributions.distribution import enable_validation

from gwkokab.analysis.core.flowMC_based import flowMC_arg_parser, FlowMCBased
from gwkokab.analysis.core.inference_io import (
    DiscretePELoader as DataLoader,
    SamplerConfig,
)
from gwkokab.analysis.core.numpyro_based import NumpyroBased
from gwkokab.analysis.core.sage import Sage, sage_arg_parser as sage_parser
from gwkokab.analysis.ecc_matters.common import (
    EccentricityMattersCore,
    EccentricityMattersModel,
)
from gwkokab.analysis.utils.logger import log_info
from gwkokab.inference.factory import get_likelihood_fn


class EccentricityMattersFSage(EccentricityMattersCore, Sage, FlowMCBased):
    pass


class EccentricityMattersNSage(EccentricityMattersCore, Sage, NumpyroBased):
    pass


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    sampler_cfg = SamplerConfig.from_json(args.sampler_cfg)
    data_loader = DataLoader.from_json(args.data_loader_cfg)

    likelihood_fn = get_likelihood_fn(
        sampler_name=sampler_cfg.sampler_name,
        analysis_type="discrete",
    )

    AnalysisClass = (
        EccentricityMattersFSage
        if sampler_cfg.sampler_name == "flowMC"
        else EccentricityMattersNSage
    )

    AnalysisClass.init_rng_seed(seed=args.seed)

    AnalysisClass(
        likelihood_fn=likelihood_fn,
        model=EccentricityMattersModel,
        data_loader=data_loader,
        prior_filename=args.prior_cfg,
        poisson_mean_filename=args.pmean_cfg,
        sampler_cfg=sampler_cfg,
        variance_cut_threshold=args.variance_cut_threshold,
        analysis_name="f_sage_ecc_matters",
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        where_fns=None,
    ).run()
