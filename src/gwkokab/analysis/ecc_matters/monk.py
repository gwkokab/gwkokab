# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from numpyro.distributions.distribution import enable_validation

from gwkokab.analysis.core.flowMC_based import FlowMCBased
from gwkokab.analysis.core.inference_io import (
    AnalyticalPELoader as DataLoader,
    SamplerConfig,
)
from gwkokab.analysis.core.monk import Monk, monk_arg_parser
from gwkokab.analysis.core.numpyro_based import NumpyroBased
from gwkokab.analysis.ecc_matters.common import (
    EccentricityMattersCore,
    EccentricityMattersModel,
)
from gwkokab.analysis.utils.logger import log_info
from gwkokab.inference.factory import get_likelihood_fn


class EccentricityMattersFMonk(EccentricityMattersCore, Monk, FlowMCBased):
    pass


class EccentricityMattersNMonk(EccentricityMattersCore, Monk, NumpyroBased):
    pass


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    sampler_cfg = SamplerConfig.from_json(args.sampler_cfg)
    data_loader = DataLoader.from_json(args.data_loader_cfg)

    likelihood_fn = get_likelihood_fn(
        sampler_name=sampler_cfg.sampler_name,
        analysis_type="analytical",
    )

    AnalysisClass = (
        EccentricityMattersFMonk
        if sampler_cfg.sampler_name == "flowMC"
        else EccentricityMattersNMonk
    )

    AnalysisClass.init_rng_seed(seed=args.seed)

    AnalysisClass(
        likelihood_fn=likelihood_fn,
        model=EccentricityMattersModel,
        data_loader=data_loader,
        prior_filename=args.prior_cfg,
        poisson_mean_filename=args.pmean_cfg,
        sampler_cfg=sampler_cfg,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        analysis_name="ecc_matters",
        n_samples=args.n_samples,
        variance_cut_threshold=args.variance_cut_threshold,
    ).run()
