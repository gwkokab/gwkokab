# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Dict, List, Union

from gwkokab.models import BrokenPowerlawTwoPeakFull
from gwkokab.parameters import Parameters as P
from kokab.core.monk import Monk, monk_arg_parser
from kokab.utils.logger import log_info


class BrokenPowerlawTwoPeakFullMonk(Monk):
    def __init__(
        self,
        has_spin: bool,
        has_tilt: bool,
        has_eccentricity: bool,
        has_redshift: bool,
        data_filename: str,
        seed: int,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_samples: int,
        minimum_mc_error: float,
        n_checkpoints: int,
        n_max_steps: int,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        self.has_spin = has_spin
        self.has_tilt = has_tilt
        self.has_eccentricity = has_eccentricity
        self.has_redshift = has_redshift

        super().__init__(
            BrokenPowerlawTwoPeakFull,
            data_filename,
            seed,
            prior_filename,
            poisson_mean_filename,
            sampler_settings_filename,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            analysis_name="bp2pfull",
            n_samples=n_samples,
            minimum_mc_error=minimum_mc_error,
            n_checkpoints=n_checkpoints,
            n_max_steps=n_max_steps,
        )

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        return {
            "use_spin": self.has_spin,
            "use_tilt": self.has_tilt,
            "use_eccentricity": self.has_eccentricity,
            "use_redshift": self.has_redshift,
        }

    @property
    def parameters(self) -> List[str]:
        names = [P.PRIMARY_MASS_SOURCE.value, P.SECONDARY_MASS_SOURCE.value]
        if self.has_spin:
            names.append(P.PRIMARY_SPIN_MAGNITUDE.value)
            names.append(P.SECONDARY_SPIN_MAGNITUDE.value)
        if self.has_tilt:
            names.extend([P.COS_TILT_1.value, P.COS_TILT_2.value])
        if self.has_eccentricity:
            names.append(P.ECCENTRICITY.value)
        if self.has_redshift:
            names.append(P.REDSHIFT.value)
        return names

    @property
    def model_parameters(self) -> List[str]:
        all_params: List[str] = [
            "alpha1",
            "alpha2",
            "beta",
            "delta_m1",
            "delta_m2",
            "lambda_0",
            "lambda_1",
            "loc1",
            "loc2",
            "log_rate",
            "m1min",
            "m2min",
            "mbreak",
            "mmax",
            "scale1",
            "scale2",
        ]

        if self.has_spin:
            all_params.extend(["chi_loc", "chi_scale"])

        if self.has_tilt:
            all_params.extend(["cos_tilt_zeta", "cos_tilt_loc", "cos_tilt_scale"])

        if self.has_eccentricity:
            all_params.append("eccentricity_scale")

        if self.has_redshift:
            all_params.extend(["z_max", "kappa"])

        return all_params


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--add-spin",
        action="store_true",
        help="Include beta spin parameters in the model.",
    )
    model_group.add_argument(
        "--add-tilt",
        action="store_true",
        help="Include tilt parameters in the model.",
    )
    model_group.add_argument(
        "--add-eccentricity",
        action="store_true",
        help="Include eccentricity parameter in the model",
    )
    model_group.add_argument(
        "--add-redshift",
        action="store_true",
        help="Include redshift parameter in the model",
    )
    return parser


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    BrokenPowerlawTwoPeakFullMonk(
        has_spin=args.add_spin,
        has_tilt=args.add_tilt,
        has_eccentricity=args.add_eccentricity,
        has_redshift=args.add_redshift,
        data_filename=args.data_filename,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        n_samples=args.n_samples,
        minimum_mc_error=args.minimum_mc_error,
        n_checkpoints=args.n_checkpoints,
        n_max_steps=args.n_max_steps,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
