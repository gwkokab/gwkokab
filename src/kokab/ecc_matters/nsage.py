# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

from loguru import logger

from gwkokab.parameters import ECCENTRICITY, PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE
from kokab.ecc_matters.common import EccentricityMattersModel
from kokab.utils.nsage import get_parser, NSage


class EccentricityMattersNSage(NSage):
    @property
    def parameters(self) -> List[str]:
        return [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name, ECCENTRICITY.name]

    @property
    def model_parameters(self) -> List[str]:
        return ["alpha_m", "high", "loc", "log_rate", "low", "mmax", "mmin", "scale"]


def main() -> None:
    r"""Main function of the script."""
    logger.warning(
        "If you have made any changes to any parameters, please make sure"
        " that the changes are reflected in scripts that generate plots.",
    )

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = get_parser(parser)

    args = parser.parse_args()

    EccentricityMattersNSage(
        model=EccentricityMattersModel,
        posterior_regex=args.posterior_regex,
        seed=args.seed,
        prior_filename=args.prior_json,
        selection_fn_filename=args.vt_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        analysis_name="ecc_matters",
    ).run(
        has_log_ref_prior=args.has_log_ref_prior,
        n_buckets=args.n_buckets,
        threshold=args.threshold,
    )
