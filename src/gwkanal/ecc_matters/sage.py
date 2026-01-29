# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

from numpyro.distributions.distribution import enable_validation

from gwkanal.core.flowMC_based import flowMC_arg_parser, FlowMCBased
from gwkanal.core.inference_io import DiscreteParameterEstimationLoader as DataLoader
from gwkanal.core.numpyro_based import numpyro_arg_parser, NumpyroBased
from gwkanal.core.sage import Sage, sage_arg_parser as sage_parser
from gwkanal.ecc_matters.common import EccentricityMattersModel
from gwkanal.utils.logger import log_info
from gwkokab.inference import flowMC_poisson_likelihood, numpyro_poisson_likelihood
from gwkokab.parameters import Parameters as P


class EccentricityMattersCore(Sage):
    @property
    def parameters(self) -> List[str]:
        return [P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE, P.ECCENTRICITY]

    @property
    def model_parameters(self) -> List[str]:
        return ["log_rate", "alpha_m", "mmin", "mmax", "loc", "scale", "low", "high"]


class EccentricityMattersFSage(EccentricityMattersCore, FlowMCBased):
    pass


class EccentricityMattersNSage(EccentricityMattersCore, NumpyroBased):
    pass


def f_main() -> None:
    enable_validation()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    EccentricityMattersFSage(
        likelihood_fn=flowMC_poisson_likelihood,
        model=EccentricityMattersModel,
        data_loader=data_loader,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_json,
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

    EccentricityMattersNSage(
        likelihood_fn=numpyro_poisson_likelihood,
        model=EccentricityMattersModel,
        data_loader=data_loader,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_json,
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
