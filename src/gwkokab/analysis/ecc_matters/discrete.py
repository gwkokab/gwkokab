# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``discrete_ecc_matters`` console script: eccentricity-matters inference over per-
event posterior samples.

One cell of the analysis matrix. The family half is :class:`~gwkokab.analysis.ecc_matters.common.EccentricityMattersCore`, which names the
parameters and hyper-parameters; the data-representation half is :class:`~gwkokab.analysis.core.discrete_base.DiscreteBase`, which
supplies ``read_data`` and ``run``. The sampler half is chosen at *runtime* by
``sampler_cfg.json``, which is why there are two trivial subclasses -- one mixing in
flowMC, one NumPyro -- and :func:`main` picks between them after reading that file.
"""

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
    """Eccentricity-matters analysis over per-event posterior samples, sampled with
    flowMC.

    Selected by :func:`main` when ``sampler_cfg.json`` names ``flowMC``.
    """

    pass


class EccentricityMattersNDiscreteAnalysis(
    EccentricityMattersCore, DiscreteBase, NumpyroBase
):
    """Eccentricity-matters analysis over per-event posterior samples, sampled with
    NumPyro.

    Selected by :func:`main` when ``sampler_cfg.json`` names ``numpyro``.
    """

    pass


def main() -> None:
    """Console script entry point for ``discrete_ecc_matters``.

    Parses the command line, reads the sampler and data loader configurations, picks the
    likelihood and the analysis class matching the configured sampler, seeds the PRNG,
    and runs the analysis.
    """
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
