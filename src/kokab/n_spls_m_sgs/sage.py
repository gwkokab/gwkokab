# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional, Tuple, Union

from jaxtyping import Array, ArrayLike
from numpyro._typing import DistributionLike
from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import flowMC_poisson_likelihood, numpyro_poisson_likelihood
from gwkokab.models import NSmoothingPowerlawMGaussian
from gwkokab.models.utils import JointDistribution, ScaledMixture
from gwkokab.parameters import Parameters as P
from kokab.core.flowMC_based import flowMC_arg_parser, FlowMCBased
from kokab.core.numpyro_based import numpyro_arg_parser, NumpyroBased
from kokab.core.sage import Sage, sage_arg_parser
from kokab.utils.common import expand_arguments
from kokab.utils.logger import log_info


class NPowerlawMGaussianCore(Sage):
    def __init__(
        self,
        N_pl: int,
        N_g: int,
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

        super().__init__(
            likelihood_fn=likelihood_fn,
            model=NSmoothingPowerlawMGaussian,
            posterior_regex=posterior_regex,
            posterior_columns=posterior_columns,
            seed=seed,
            prior_filename=prior_filename,
            poisson_mean_filename=poisson_mean_filename,
            sampler_settings_filename=sampler_settings_filename,
            variance_cut_threshold=variance_cut_threshold,
            analysis_name="n_spls_m_sgs",
            n_buckets=n_buckets,
            threshold=threshold,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            where_fns=None,
        )

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        return {"N_pl": self.N_pl, "N_g": self.N_g}

    @property
    def parameters(self) -> List[str]:
        names = [
            P.PRIMARY_MASS_SOURCE.value,
            P.SECONDARY_MASS_SOURCE.value,
            P.PRIMARY_SPIN_MAGNITUDE.value,
            P.SECONDARY_SPIN_MAGNITUDE.value,
            P.COS_TILT_1.value,
            P.COS_TILT_2.value,
            P.REDSHIFT.value,
        ]

        return names

    @property
    def model_parameters(self) -> List[str]:
        all_params: List[Tuple[str, int]] = [
            ("a1_loc_g", self.N_g),
            ("a1_loc_pl", self.N_pl),
            ("a1_scale_g", self.N_g),
            ("a1_scale_pl", self.N_pl),
            ("a2_loc_g", self.N_g),
            ("a2_loc_pl", self.N_pl),
            ("a2_scale_g", self.N_g),
            ("a2_scale_pl", self.N_pl),
            ("alpha", self.N_pl),
            ("lambda", self.N_g + self.N_pl - 1),
            ("loc_g", self.N_g),
            ("mmax_g", self.N_g),
            ("mmax_pl", self.N_pl),
            ("mmin_g", self.N_g),
            ("mmin_pl", self.N_pl),
            ("scale_g", self.N_g),
            ("t1_loc_g", self.N_g),
            ("t1_loc_pl", self.N_pl),
            ("t1_scale_g", self.N_g),
            ("t1_scale_pl", self.N_pl),
            ("t2_loc_g", self.N_g),
            ("t2_loc_pl", self.N_pl),
            ("t2_scale_g", self.N_g),
            ("t2_scale_pl", self.N_pl),
            ("zeta_g", self.N_g),
            ("zeta_pl", self.N_pl),
        ]

        extended_params = [
            "beta",
            "delta_m1",
            "delta_m2",
            "kappa",
            "log_rate",
            "m2min",
            "z_max",
        ]
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


class NPowerlawMGaussianFSage(NPowerlawMGaussianCore, FlowMCBased):
    pass


class NPowerlawMGaussianNSage(NPowerlawMGaussianCore, NumpyroBased):
    pass


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

    return parser


def f_main() -> None:
    enable_validation()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = sage_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    NPowerlawMGaussianFSage(
        N_pl=args.n_pl,
        N_g=args.n_g,
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

    NPowerlawMGaussianNSage(
        N_pl=args.n_pl,
        N_g=args.n_g,
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
