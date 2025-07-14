# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List

from gwkokab.parameters import ECCENTRICITY, PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE
from kokab.ecc_matters.common import EccentricityMattersModel
from kokab.utils.monk import get_parser, Monk


class EccentricityMattersMonk(Monk):
    @property
    def parameters(self) -> List[str]:
        return [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name, ECCENTRICITY.name]

    @property
    def model_parameters(self) -> List[str]:
        return ["log_rate", "alpha_m", "mmin", "mmax", "loc", "scale", "low", "high"]


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = get_parser(parser)

    args = parser.parse_args()

    EccentricityMattersMonk(
        model=EccentricityMattersModel,
        data_filename=args.data_filename,
        seed=args.seed,
        prior_filename=args.prior_json,
        selection_fn_filename=args.vt_json,
        poisson_mean_filename=args.pmean_json,
        flowMC_settings_filename=args.flowMC_json,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        analysis_name="ecc_matters",
    ).run(
        n_samples=args.n_samples,
        max_iter=args.max_iter,
    )
