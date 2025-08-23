# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Dict, List, Union

from gwkokab.models import SmoothedPowerlawAndPeak
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    REDSHIFT,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)
from kokab.utils.nsage import get_parser, NSage


class SmoothedPowerlawAndPeakNSage(NSage):
    def __init__(
        self,
        has_spin: bool,
        has_tilt: bool,
        has_redshift: bool,
        posterior_regex: str,
        posterior_columns: List[str],
        seed: int,
        prior_filename: str,
        selection_fn_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        self.has_spin = has_spin
        self.has_tilt = has_tilt
        self.has_redshift = has_redshift

        super().__init__(
            SmoothedPowerlawAndPeak,
            posterior_regex,
            posterior_columns,
            seed,
            prior_filename,
            selection_fn_filename,
            poisson_mean_filename,
            sampler_settings_filename,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            analysis_name="n_pls_m_gs",
        )

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        return {
            "use_spin": self.has_spin,
            "use_tilt": self.has_tilt,
            "use_redshift": self.has_redshift,
        }

    @property
    def parameters(self) -> List[str]:
        names = [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name]
        if self.has_spin:
            names.append(PRIMARY_SPIN_MAGNITUDE.name)
            names.append(SECONDARY_SPIN_MAGNITUDE.name)
        if self.has_tilt:
            names.extend([COS_TILT_1.name, COS_TILT_2.name])
        if self.has_redshift:
            names.append(REDSHIFT.name)
        return names

    @property
    def model_parameters(self) -> List[str]:
        model_parameters: List[str] = [
            "alpha",
            "beta",
            "delta",
            "lambda_peak",
            "loc",
            "log_rate",
            "mmax",
            "mmin",
            "scale",
        ]

        if self.has_spin:
            model_parameters.extend(
                [
                    "chi1_mean_g",
                    "chi1_mean_pl",
                    "chi1_variance_g",
                    "chi1_variance_pl",
                    "chi2_mean_g",
                    "chi2_mean_pl",
                    "chi2_variance_g",
                    "chi2_variance_pl",
                ]
            )

        if self.has_tilt:
            model_parameters.extend(
                [
                    "cos_tilt_zeta_g",
                    "cos_tilt_zeta_pl",
                    "cos_tilt1_scale_g",
                    "cos_tilt1_scale_pl",
                    "cos_tilt2_scale_g",
                    "cos_tilt2_scale_pl",
                ]
            )

        if self.has_redshift:
            model_parameters.extend(["kappa", "z_max"])

        return model_parameters


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = get_parser(parser)

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
        "--add-redshift",
        action="store_true",
        help="Include redshift parameter in the model",
    )

    args = parser.parse_args()

    SmoothedPowerlawAndPeakNSage(
        has_spin=args.add_spin,
        has_tilt=args.add_tilt,
        has_redshift=args.add_redshift,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        selection_fn_filename=args.vt_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.flowMC_json,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run(
        n_buckets=args.n_buckets,
        threshold=args.threshold,
    )
