# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional, Tuple, Union

from jax import numpy as jnp
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
from kokab.utils.checks import check_min_concentration_for_beta_dist
from kokab.utils.common import expand_arguments
from kokab.utils.logger import log_info


def where_fns_list(
    use_beta_spin_magnitude: bool,
) -> Optional[List[Callable[..., Array]]]:
    where_fns = []

    if use_beta_spin_magnitude:

        def positive_concentration(**kwargs) -> Array:
            N_pl: int = kwargs.get("N_pl")  # type: ignore
            N_g: int = kwargs.get("N_g")  # type: ignore
            mask = jnp.ones((), dtype=bool)
            for n_pl in range(N_pl):
                chi_mean: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE.value + "_mean_pl_" + str(n_pl)
                )  # type: ignore
                chi_variance: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE.value + "_variance_pl_" + str(n_pl)
                )  # type: ignore
                mask &= check_min_concentration_for_beta_dist(chi_mean, chi_variance)
            for n_g in range(N_g):
                chi_mean: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE.value + "_mean_g_" + str(n_g)
                )  # type: ignore
                chi_variance: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE.value + "_variance_g_" + str(n_g)
                )  # type: ignore
                mask &= check_min_concentration_for_beta_dist(chi_mean, chi_variance)
            return mask

        where_fns.append(positive_concentration)

    return where_fns if len(where_fns) > 0 else None


class NSmoothedPowerlawMSmoothedGaussianCore(Sage):
    def __init__(
        self,
        N_pl: int,
        N_g: int,
        use_beta_spin_magnitude: bool,
        use_spin_magnitude_mixture: bool,
        use_chi_eff_mixture: bool,
        use_tilt: bool,
        use_eccentricity_mixture: bool,
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
        self.use_spin_magnitude_mixture = use_spin_magnitude_mixture
        self.use_chi_eff_mixture = use_chi_eff_mixture
        self.use_tilt = use_tilt
        self.use_eccentricity_mixture = use_eccentricity_mixture
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
            where_fns=where_fns_list(use_beta_spin_magnitude=use_beta_spin_magnitude),
        )

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        return {
            "N_pl": self.N_pl,
            "N_g": self.N_g,
            "use_beta_spin_magnitude": self.use_beta_spin_magnitude,
            "use_spin_magnitude_mixture": self.use_spin_magnitude_mixture,
            "use_chi_eff_mixture": self.use_chi_eff_mixture,
            "use_tilt": self.use_tilt,
            "use_eccentricity_mixture": self.use_eccentricity_mixture,
            "use_redshift": self.use_redshift,
        }

    @property
    def parameters(self) -> List[str]:
        names = [P.PRIMARY_MASS_SOURCE.value]
        if self.use_beta_spin_magnitude or self.use_spin_magnitude_mixture:
            names.append(P.PRIMARY_SPIN_MAGNITUDE.value)
            names.append(P.SECONDARY_SPIN_MAGNITUDE.value)
        if self.use_chi_eff_mixture:
            names.append(P.EFFECTIVE_SPIN_MAGNITUDE.value)
        if self.use_tilt:
            names.extend([P.COS_TILT_1.value, P.COS_TILT_2.value])
        if self.use_eccentricity_mixture:
            names.append(P.ECCENTRICITY.value)
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

        if self.use_spin_magnitude_mixture:
            # fmt: off
            all_params.extend(
                [
                    ("a_zeta_g", self.N_g),
                    ("a_zeta_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_gaussian_high_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_gaussian_high_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_gaussian_low_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_gaussian_low_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_isotropic_high_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_isotropic_high_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_isotropic_low_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_isotropic_low_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_loc_pl", self.N_pl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_scale_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_gaussian_high_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_gaussian_high_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_gaussian_low_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_gaussian_low_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_isotropic_high_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_isotropic_high_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_isotropic_low_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_isotropic_low_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_loc_pl", self.N_pl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_scale_pl", self.N_pl),
                ]
            )
            # fmt: on

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

        if self.use_chi_eff_mixture:
            all_params.extend(
                [
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_high1_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_high1_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_high2_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_high2_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_loc1_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_loc1_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_loc2_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_loc2_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_low1_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_low1_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_low2_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_low2_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_scale1_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_scale1_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_scale2_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_scale2_pl", self.N_pl),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_zeta_g", self.N_g),
                    (P.EFFECTIVE_SPIN_MAGNITUDE.value + "_zeta_pl", self.N_pl),
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

        if self.use_eccentricity_mixture:
            all_params.extend(
                [
                    (P.ECCENTRICITY.value + "_high1_g", self.N_g),
                    (P.ECCENTRICITY.value + "_high1_pl", self.N_pl),
                    (P.ECCENTRICITY.value + "_high2_g", self.N_g),
                    (P.ECCENTRICITY.value + "_high2_pl", self.N_pl),
                    (P.ECCENTRICITY.value + "_loc1_g", self.N_g),
                    (P.ECCENTRICITY.value + "_loc1_pl", self.N_pl),
                    (P.ECCENTRICITY.value + "_loc2_g", self.N_g),
                    (P.ECCENTRICITY.value + "_loc2_pl", self.N_pl),
                    (P.ECCENTRICITY.value + "_low1_g", self.N_g),
                    (P.ECCENTRICITY.value + "_low1_pl", self.N_pl),
                    (P.ECCENTRICITY.value + "_low2_g", self.N_g),
                    (P.ECCENTRICITY.value + "_low2_pl", self.N_pl),
                    (P.ECCENTRICITY.value + "_scale1_g", self.N_g),
                    (P.ECCENTRICITY.value + "_scale1_pl", self.N_pl),
                    (P.ECCENTRICITY.value + "_scale2_g", self.N_g),
                    (P.ECCENTRICITY.value + "_scale2_pl", self.N_pl),
                    (P.ECCENTRICITY.value + "_zeta_g", self.N_g),
                    (P.ECCENTRICITY.value + "_zeta_pl", self.N_pl),
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
            "delta_m",
            "log_rate",
            "mmax",
            "mmin",
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
        "--add-spin-magnitude-mixture",
        action="store_true",
        help="Include truncated normal spin magnitude parameters in the model.",
    )

    model_group.add_argument(
        "--add-chi-eff-mixture",
        action="store_true",
        help="Include chi_eff mixture parameters in the model.",
    )
    model_group.add_argument(
        "--add-tilt",
        action="store_true",
        help="Include tilt parameters in the model.",
    )
    model_group.add_argument(
        "--add-eccentricity-mixture",
        action="store_true",
        help="Include eccentricity mixture in the model.",
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
        use_spin_magnitude_mixture=args.add_spin_magnitude_mixture,
        use_chi_eff_mixture=args.add_chi_eff_mixture,
        use_tilt=args.add_tilt,
        use_eccentricity_mixture=args.add_eccentricity_mixture,
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
        use_spin_magnitude_mixture=args.add_spin_magnitude_mixture,
        use_chi_eff_mixture=args.add_chi_eff_mixture,
        use_tilt=args.add_tilt,
        use_eccentricity_mixture=args.add_eccentricity_mixture,
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
