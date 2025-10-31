# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Dict, List, Union

from gwkokab.models import BrokenPowerlawTwoPeakMultiSpinMultiTiltFull
from gwkokab.parameters import Parameters as P
from kokab.core.monk import Monk, monk_arg_parser
from kokab.utils.logger import log_info


class BrokenPowerlawTwoPeakMultiSpinFullMonk(Monk):
    def __init__(
        self,
        has_eccentricity: bool,
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
        self.has_eccentricity = has_eccentricity

        super().__init__(
            BrokenPowerlawTwoPeakMultiSpinMultiTiltFull,
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
        return {"use_eccentricity": self.has_eccentricity}

    @property
    def parameters(self) -> List[str]:
        names = [
            P.PRIMARY_MASS_SOURCE.value,
            P.SECONDARY_MASS_SOURCE.value,
            P.PRIMARY_SPIN_MAGNITUDE.value,
            P.SECONDARY_SPIN_MAGNITUDE.value,
            P.COS_TILT_1.value,
            P.COS_TILT_2.value,
        ]
        if self.has_eccentricity:
            names.append(P.ECCENTRICITY.value)
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
            "a_1_loc_bpl1",
            "a_1_loc_bpl2",
            "a_1_loc_n1",
            "a_1_loc_n2",
            "a_2_loc_bpl1",
            "a_2_loc_bpl2",
            "a_2_loc_n1",
            "a_2_loc_n2",
            "a_1_scale_bpl1",
            "a_1_scale_bpl2",
            "a_1_scale_n1",
            "a_1_scale_n2",
            "a_2_scale_bpl1",
            "a_2_scale_bpl2",
            "a_2_scale_n1",
            "a_2_scale_n2",
            "cos_tilt_1_loc_bpl1",
            "cos_tilt_1_loc_bpl2",
            "cos_tilt_1_loc_n1",
            "cos_tilt_1_loc_n2",
            "cos_tilt_1_scale_bpl1",
            "cos_tilt_1_scale_bpl2",
            "cos_tilt_1_scale_n1",
            "cos_tilt_1_scale_n2",
            "cos_tilt_2_loc_bpl1",
            "cos_tilt_2_loc_bpl2",
            "cos_tilt_2_loc_n1",
            "cos_tilt_2_loc_n2",
            "cos_tilt_2_scale_bpl1",
            "cos_tilt_2_scale_bpl2",
            "cos_tilt_2_scale_n1",
            "cos_tilt_2_scale_n2",
            "cos_tilt_zeta_bpl1",
            "cos_tilt_zeta_bpl2",
            "cos_tilt_zeta_n1",
            "cos_tilt_zeta_n2",
            "z_max",
            "kappa",
        ]

        if self.has_eccentricity:
            all_params.append("eccentricity_scale")

        return all_params


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--add-eccentricity",
        action="store_true",
        help="Include eccentricity parameter in the model",
    )
    return parser


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    BrokenPowerlawTwoPeakMultiSpinFullMonk(
        has_eccentricity=args.add_eccentricity,
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
