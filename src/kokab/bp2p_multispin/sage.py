# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional, Union

from jaxtyping import Array, ArrayLike
from numpyro._typing import DistributionLike
from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import numpyro_poisson_likelihood, poisson_likelihood
from gwkokab.models import BrokenPowerlawTwoPeakMultiSpinFull
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from kokab.core.flowMC_based import flowMC_arg_parser, FlowMCBased
from kokab.core.numpyro_based import numpyro_arg_parser, NumpyroBased
from kokab.core.sage import Sage, sage_arg_parser
from kokab.utils.logger import log_info


class BrokenPowerlawTwoPeakMultiSpinFullCore(Sage):
    def __init__(
        self,
        has_eccentricity: bool,
        likelihood_fn: Callable[
            [
                Callable[..., DistributionLike],
                JointDistribution,
                Dict[str, DistributionLike],
                Dict[str, int],
                ArrayLike,
                Callable[[ScaledMixture], Array],
                Optional[List[Callable[..., Array]]],
                Dict[str, Array],
            ],
            Callable,
        ],
        posterior_regex: str,
        posterior_columns: List[str],
        seed: int,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_buckets: int,
        threshold: float,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        self.has_eccentricity = has_eccentricity

        super().__init__(
            likelihood_fn=likelihood_fn,
            model=BrokenPowerlawTwoPeakMultiSpinFull,
            posterior_regex=posterior_regex,
            posterior_columns=posterior_columns,
            seed=seed,
            prior_filename=prior_filename,
            poisson_mean_filename=poisson_mean_filename,
            sampler_settings_filename=sampler_settings_filename,
            analysis_name="one_powerlaw_one_peak",
            n_buckets=n_buckets,
            threshold=threshold,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            where_fns=None,
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
        model_parameters = [
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
            "loc_a1_bpl",
            "loc_a1_n1",
            "loc_a1_n2",
            "loc_a2_bpl",
            "loc_a2_n1",
            "loc_a2_n2",
            "scale_a1_bpl",
            "scale_a1_n1",
            "scale_a1_n2",
            "scale_a2_bpl",
            "scale_a2_n1",
            "scale_a2_n2",
            "cos_tilt_zeta",
            "cos_tilt_loc",
            "cos_tilt_scale",
            "kappa",
            "z_max",
        ]

        if self.has_eccentricity:
            model_parameters.append("eccentricity_scale")

        return model_parameters


class BrokenPowerlawTwoPeakMultiSpinFullFSage(
    BrokenPowerlawTwoPeakMultiSpinFullCore, FlowMCBased
):
    pass


class BrokenPowerlawTwoPeakMultiSpinFullNSage(
    BrokenPowerlawTwoPeakMultiSpinFullCore, NumpyroBased
):
    pass


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--add-eccentricity",
        action="store_true",
        help="Include eccentricity parameter in the model",
    )

    return parser


def f_main() -> None:
    enable_validation()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = sage_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    BrokenPowerlawTwoPeakMultiSpinFullFSage(
        has_eccentricity=args.add_eccentricity,
        likelihood_fn=poisson_likelihood,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()


def n_main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = sage_arg_parser(parser)
    parser = numpyro_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    BrokenPowerlawTwoPeakMultiSpinFullNSage(
        has_eccentricity=args.add_eccentricity,
        likelihood_fn=numpyro_poisson_likelihood,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
