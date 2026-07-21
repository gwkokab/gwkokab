# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from numpyro.distributions.distribution import enable_validation

from gwkokab.analysis.core.discrete_base import discrete_arg_parser, DiscreteBase
from gwkokab.analysis.core.flowMC_base import flowMC_arg_parser, FlowMCBase
from gwkokab.analysis.core.inference_io import (
    DiscretePELoader as DataLoader,
    SamplerConfig,
)
from gwkokab.analysis.core.numpyro_base import NumpyroBase
from gwkokab.analysis.ecc_matters.common import (
    EccentricityMattersCore,
    EccentricityMattersModel,
)
from gwkokab.analysis.utils.logger import log_info
from gwkokab.inference.factory import get_likelihood_fn


class EccentricityMattersFDiscreteAnalysis(
    EccentricityMattersCore, DiscreteBase, FlowMCBase
):
    pass


class EccentricityMattersNDiscreteAnalysis(
    EccentricityMattersCore, DiscreteBase, NumpyroBase
):
    pass


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = discrete_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    sampler_cfg = SamplerConfig.read_from_json(args.sampler_cfg)
    data_loader = DataLoader.read_from_json(args.data_loader_cfg)

    likelihood_fn = get_likelihood_fn(
        sampler_name=sampler_cfg.sampler_name,
        analysis_type="discrete",
    )

    AnalysisClass = (
        EccentricityMattersFDiscreteAnalysis
        if sampler_cfg.sampler_name == "flowMC"
        else EccentricityMattersNDiscreteAnalysis
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
        analysis_name="ecc_matters",
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        where_fns=None,
    ).run()
