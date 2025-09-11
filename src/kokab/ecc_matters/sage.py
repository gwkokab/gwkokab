# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import numpyro_poisson_likelihood, poisson_likelihood
from gwkokab.parameters import Parameters
from kokab.ecc_matters.common import EccentricityMattersModel
from kokab.utils.flowMC_based import flowMC_arg_parser, FlowMCBased
from kokab.utils.numpyro_based import numpyro_arg_parser, NumpyroBased
from kokab.utils.sage import Sage, sage_arg_parser as sage_parser


class EccentricityMattersCore(Sage):
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

    EccentricityMattersFSage(
        likelihood_fn=poisson_likelihood,
        model=EccentricityMattersModel,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        selection_fn_filename=args.vt_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
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

    EccentricityMattersNSage(
        likelihood_fn=numpyro_poisson_likelihood,
        model=EccentricityMattersModel,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        selection_fn_filename=args.vt_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        analysis_name="n_sage_ecc_matters",
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        where_fns=None,
    ).run()
