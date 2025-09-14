# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional, Union

from jaxtyping import Array, ArrayLike
from numpyro._typing import DistributionLike
from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import numpyro_poisson_likelihood, poisson_likelihood
from gwkokab.models import PowerlawPeak
from gwkokab.models.utils import JointDistribution
from gwkokab.parameters import Parameters
from gwkokab.poisson_mean import PoissonMean
from kokab.utils.checks import check_min_concentration_for_beta_dist
from kokab.utils.flowMC_based import flowMC_arg_parser, FlowMCBased
from kokab.utils.numpyro_based import numpyro_arg_parser, NumpyroBased
from kokab.utils.sage import Sage, sage_arg_parser


def where_fns_list(has_spin: bool) -> Optional[List[Callable[..., Array]]]:
    where_fns = []

    if has_spin:
        where_fns.append(
            lambda chi_mean,
            chi_variance,
            **kwargs: check_min_concentration_for_beta_dist(chi_mean, chi_variance)
        )

    return where_fns if len(where_fns) > 0 else None


class PowerlawPeakCore(Sage):
    def __init__(
        self,
        has_spin: bool,
        has_tilt: bool,
        has_redshift: bool,
        likelihood_fn: Callable[
            [
                Callable[..., DistributionLike],
                JointDistribution,
                Dict[str, DistributionLike],
                Dict[str, int],
                ArrayLike,
                PoissonMean,
                Optional[List[Callable[..., Array]]],
                Dict[str, Array],
            ],
            Callable,
        ],
        posterior_regex: str,
        posterior_columns: List[str],
        seed: int,
        prior_filename: str,
        selection_fn_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_buckets: int,
        threshold: float,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        self.has_spin = has_spin
        self.has_tilt = has_tilt
        self.has_redshift = has_redshift

        super().__init__(
            likelihood_fn=likelihood_fn,
            model=PowerlawPeak,
            posterior_regex=posterior_regex,
            posterior_columns=posterior_columns,
            seed=seed,
            prior_filename=prior_filename,
            selection_fn_filename=selection_fn_filename,
            poisson_mean_filename=poisson_mean_filename,
            sampler_settings_filename=sampler_settings_filename,
            analysis_name="one_powerlaw_one_peak",
            n_buckets=n_buckets,
            threshold=threshold,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            where_fns=where_fns_list(has_spin=has_spin),
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
        names = [
            Parameters.PRIMARY_MASS_SOURCE.value,
            Parameters.SECONDARY_MASS_SOURCE.value,
        ]
        if self.has_spin:
            names.append(Parameters.PRIMARY_SPIN_MAGNITUDE.value)
            names.append(Parameters.SECONDARY_SPIN_MAGNITUDE.value)
        if self.has_tilt:
            names.extend([Parameters.COS_TILT_1.value, Parameters.COS_TILT_2.value])
        if self.has_redshift:
            names.append(Parameters.REDSHIFT.value)
        return names

    @property
    def model_parameters(self) -> List[str]:
        model_parameters = [
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
            model_parameters.extend(["chi_mean", "chi_variance"])

        if self.has_tilt:
            model_parameters.extend(["cos_tilt_zeta", "cos_tilt_scale"])

        if self.has_redshift:
            model_parameters.extend(["kappa", "z_max"])

        return model_parameters


class PowerlawPeakFSage(PowerlawPeakCore, FlowMCBased):
    pass


class PowerlawPeakNSage(PowerlawPeakCore, NumpyroBased):
    pass


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

    PowerlawPeakFSage(
        has_spin=args.add_spin,
        has_tilt=args.add_tilt,
        has_redshift=args.add_redshift,
        likelihood_fn=poisson_likelihood,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        selection_fn_filename=args.vt_json,
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

    PowerlawPeakNSage(
        has_spin=args.add_spin,
        has_tilt=args.add_tilt,
        has_redshift=args.add_redshift,
        likelihood_fn=numpyro_poisson_likelihood,
        posterior_regex=args.posterior_regex,
        posterior_columns=args.posterior_columns,
        seed=args.seed,
        prior_filename=args.prior_json,
        selection_fn_filename=args.vt_json,
        poisson_mean_filename=args.pmean_json,
        sampler_settings_filename=args.sampler_config,
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
