# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional, Tuple, Union

from jaxtyping import Array, ArrayLike
from numpyro._typing import DistributionLike
from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import flowMC_poisson_likelihood, numpyro_poisson_likelihood
from gwkokab.models import NSmoothedPowerlawMSmoothedGaussian
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from kokab.core.flowMC_based import flowMC_arg_parser, FlowMCBased
from kokab.core.numpyro_based import numpyro_arg_parser, NumpyroBased
from kokab.core.sage import Sage, sage_arg_parser
from kokab.utils.common import expand_arguments
from kokab.utils.logger import log_info


class NSmoothedPowerlawMSmoothedGaussianCore(Sage):
    def __init__(
        self,
        N_pl: int,
        N_g: int,
        use_beta_spin_magnitude: bool,
        use_truncated_normal_spin_magnitude: bool,
        use_tilt: bool,
        use_redshift: bool,
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
        variance_cut_threshold: Optional[float],
        n_buckets: int,
        threshold: float,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        self.N_pl = N_pl
        self.N_g = N_g
        self.use_beta_spin_magnitude = use_beta_spin_magnitude
        self.use_truncated_normal_spin_magnitude = use_truncated_normal_spin_magnitude
        self.use_tilt = use_tilt
        self.use_redshift = use_redshift

        super().__init__(
            likelihood_fn=likelihood_fn,
            model=NSmoothedPowerlawMSmoothedGaussian,
            posterior_regex=posterior_regex,
            posterior_columns=posterior_columns,
            seed=seed,
            prior_filename=prior_filename,
            poisson_mean_filename=poisson_mean_filename,
            sampler_settings_filename=sampler_settings_filename,
            variance_cut_threshold=variance_cut_threshold,
            analysis_name="othree_n_pls_m_gs",
            n_buckets=n_buckets,
            threshold=threshold,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            where_fns=None,
        )

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        return {
            "N_pl": self.N_pl,
            "N_g": self.N_g,
            "use_beta_spin_magnitude": self.use_beta_spin_magnitude,
            "use_truncated_normal_spin_magnitude": self.use_truncated_normal_spin_magnitude,
            "use_tilt": self.use_tilt,
            "use_redshift": self.use_redshift,
        }

    @property
    def parameters(self) -> List[str]:
        names = [P.PRIMARY_MASS_SOURCE.value]
        if self.use_beta_spin_magnitude or self.use_truncated_normal_spin_magnitude:
            names.append(P.PRIMARY_SPIN_MAGNITUDE.value)
            names.append(P.SECONDARY_SPIN_MAGNITUDE.value)
        if self.use_tilt:
            names.extend([P.COS_TILT_1.value, P.COS_TILT_2.value])
        if self.use_redshift:
            names.append(P.REDSHIFT.value)
        names.append(P.SECONDARY_MASS_SOURCE.value)
        return names

    @property
    def model_parameters(self) -> List[str]:
        all_params: List[Tuple[str, int]] = [
            ("alpha_pl", self.N_pl),
            ("lambda", self.N_pl + self.N_g - 1),
            ("m1_high_g", self.N_g),
            ("m1_loc_g", self.N_g),
            ("m1_low_g", self.N_g),
            ("m1_scale_g", self.N_g),
            ("mmax_pl", self.N_pl),
            ("mmin_pl", self.N_pl),
        ]

        if self.use_truncated_normal_spin_magnitude:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_high_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_high_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_loc_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_low_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_low_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_scale_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_high_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_high_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_loc_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_low_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_low_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_scale_pl", self.N_pl),
                ]
            )

        if self.use_beta_spin_magnitude:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_mean_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_mean_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_variance_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_variance_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_mean_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_mean_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_variance_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_variance_pl", self.N_pl),
                ]
            )

        if self.use_tilt:
            all_params.extend(
                [
                    ("cos_tilt_zeta_g", self.N_g),
                    ("cos_tilt_zeta_pl", self.N_pl),
                    (P.COS_TILT_1.value + "_loc_pl", self.N_pl),
                    (P.COS_TILT_1.value + "_loc_g", self.N_g),
                    (P.COS_TILT_1.value + "_minimum_pl", self.N_pl),
                    (P.COS_TILT_1.value + "_minimum_g", self.N_g),
                    (P.COS_TILT_1.value + "_scale_pl", self.N_pl),
                    (P.COS_TILT_1.value + "_scale_g", self.N_g),
                    (P.COS_TILT_2.value + "_loc_pl", self.N_pl),
                    (P.COS_TILT_2.value + "_loc_g", self.N_g),
                    (P.COS_TILT_2.value + "_minimum_pl", self.N_pl),
                    (P.COS_TILT_2.value + "_minimum_g", self.N_g),
                    (P.COS_TILT_2.value + "_scale_pl", self.N_pl),
                    (P.COS_TILT_2.value + "_scale_g", self.N_g),
                ]
            )

        if self.use_redshift:
            all_params.extend(
                [
                    (P.REDSHIFT.value + "_kappa_g", self.N_g),
                    (P.REDSHIFT.value + "_kappa_pl", self.N_pl),
                    (P.REDSHIFT.value + "_z_max_g", self.N_g),
                    (P.REDSHIFT.value + "_z_max_pl", self.N_pl),
                ]
            )

        extended_params = [
            "beta",
            "delta_m1",
            "delta_m2",
            "log_rate",
            "m1max",
            "m1min",
            "m2min",
        ]
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--n-pl",
        type=int,
        help="Number of power-law components in the mass model.",
    )
    model_group.add_argument(
        "--n-g",
        type=int,
        help="Number of Gaussian components in the mass model.",
    )

    spin_group = model_group.add_mutually_exclusive_group()
    spin_group.add_argument(
        "--add-beta-spin-magnitude",
        action="store_true",
        help="Include beta spin magnitude parameters in the model.",
    )
    spin_group.add_argument(
        "--add-truncated-normal-spin-magnitude",
        action="store_true",
        help="Include truncated normal spin magnitude parameters in the model.",
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

    return parser


def f_main() -> None:
    enable_validation()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = sage_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    class NSmoothedPowerlawMSmoothedGaussianFSage(
        NSmoothedPowerlawMSmoothedGaussianCore, FlowMCBased
    ):
        pass

    NSmoothedPowerlawMSmoothedGaussianFSage(
        N_pl=args.n_pl,
        N_g=args.n_g,
        use_beta_spin_magnitude=args.add_beta_spin_magnitude,
        use_truncated_normal_spin_magnitude=args.add_truncated_normal_spin_magnitude,
        use_tilt=args.add_tilt,
        use_redshift=args.add_redshift,
        likelihood_fn=flowMC_poisson_likelihood,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        variance_cut_threshold=args.variance_cut_threshold,
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

    class NSmoothedPowerlawMSmoothedGaussianNSage(
        NSmoothedPowerlawMSmoothedGaussianCore, NumpyroBased
    ):
        pass

    NSmoothedPowerlawMSmoothedGaussianNSage(
        N_pl=args.n_pl,
        N_g=args.n_g,
        use_beta_spin_magnitude=args.add_beta_spin_magnitude,
        use_truncated_normal_spin_magnitude=args.add_truncated_normal_spin_magnitude,
        use_tilt=args.add_tilt,
        use_redshift=args.add_redshift,
        likelihood_fn=numpyro_poisson_likelihood,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        variance_cut_threshold=args.variance_cut_threshold,
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
