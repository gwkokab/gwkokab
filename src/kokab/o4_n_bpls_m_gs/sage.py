# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional, Tuple, Union

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro._typing import DistributionLike
from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import flowMC_poisson_likelihood, numpyro_poisson_likelihood
from gwkokab.models import NBrokenPowerlawMGaussian
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
            N_bpl: int = kwargs.get("N_bpl")  # type: ignore
            N_g: int = kwargs.get("N_g")  # type: ignore
            mask = jnp.ones((), dtype=bool)
            for n_bpl in range(N_bpl):
                chi_mean: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE.value + "_mean_bpl_" + str(n_bpl)
                )  # type: ignore
                chi_variance: Array = kwargs.get(
                    P.PRIMARY_SPIN_MAGNITUDE.value + "_variance_bpl_" + str(n_bpl)
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


class NBrokenPowerlawMGaussianCore(Sage):
    def __init__(
        self,
        N_bpl: int,
        N_g: int,
        use_beta_spin_magnitude: bool,
        use_spin_magnitude_mixture: bool,
        use_chi_eff_mixture: bool,
        use_skew_normal_chi_eff: bool,
        use_truncated_normal_chi_p: bool,
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
        read_reference_prior: bool,
        n_pe_samples: Optional[int],
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
        self.N_bpl = N_bpl
        self.N_g = N_g
        self.use_beta_spin_magnitude = use_beta_spin_magnitude
        self.use_spin_magnitude_mixture = use_spin_magnitude_mixture
        self.use_chi_eff_mixture = use_chi_eff_mixture
        self.use_skew_normal_chi_eff = use_skew_normal_chi_eff
        self.use_truncated_normal_chi_p = use_truncated_normal_chi_p
        self.use_tilt = use_tilt
        self.use_eccentricity_mixture = use_eccentricity_mixture
        self.use_redshift = use_redshift

        super().__init__(
            likelihood_fn=likelihood_fn,
            model=NBrokenPowerlawMGaussian,
            posterior_regex=posterior_regex,
            read_reference_prior=read_reference_prior,
            n_pe_samples=n_pe_samples,
            seed=seed,
            prior_filename=prior_filename,
            poisson_mean_filename=poisson_mean_filename,
            sampler_settings_filename=sampler_settings_filename,
            variance_cut_threshold=variance_cut_threshold,
            analysis_name="ofour_n_bpls_m_gs",
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
            "N_bpl": self.N_bpl,
            "N_g": self.N_g,
            "use_beta_spin_magnitude": self.use_beta_spin_magnitude,
            "use_spin_magnitude_mixture": self.use_spin_magnitude_mixture,
            "use_chi_eff_mixture": self.use_chi_eff_mixture,
            "use_skew_normal_chi_eff": self.use_skew_normal_chi_eff,
            "use_truncated_normal_chi_p": self.use_truncated_normal_chi_p,
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
        if self.use_chi_eff_mixture or self.use_skew_normal_chi_eff:
            names.append(P.EFFECTIVE_SPIN.value)
        if self.use_truncated_normal_chi_p:
            names.append(P.PRECESSING_SPIN.value)
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
            ("alpha1_bpl", self.N_bpl),
            ("alpha2_bpl", self.N_bpl),
            ("lambda", self.N_bpl + self.N_g - 1),
            ("m1_high_g", self.N_g),
            ("m1_loc_g", self.N_g),
            ("m1_low_g", self.N_g),
            ("m1_scale_g", self.N_g),
            ("m1break_bpl", self.N_bpl),
            ("m1max_bpl", self.N_bpl),
            ("m1min_bpl", self.N_bpl),
        ]

        if self.use_spin_magnitude_mixture:
            all_params.extend(
                [
                    ("a_zeta_g", self.N_g),
                    ("a_zeta_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp1_high_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp1_high_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp1_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp1_loc_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp1_low_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp1_low_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp1_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp1_scale_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp2_high_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp2_high_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp2_loc_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp2_loc_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp2_low_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp2_low_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp2_scale_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_comp2_scale_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp1_high_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp1_high_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp1_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp1_loc_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp1_low_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp1_low_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp1_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp1_scale_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp2_high_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp2_high_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp2_loc_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp2_loc_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp2_low_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp2_low_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp2_scale_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_comp2_scale_bpl", self.N_bpl),
                ]
            )

        if self.use_beta_spin_magnitude:
            all_params.extend(
                [
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_mean_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_mean_bpl", self.N_bpl),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_variance_g", self.N_g),
                    (P.PRIMARY_SPIN_MAGNITUDE.value + "_variance_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_mean_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_mean_bpl", self.N_bpl),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_variance_g", self.N_g),
                    (P.SECONDARY_SPIN_MAGNITUDE.value + "_variance_bpl", self.N_bpl),
                ]
            )

        if self.use_chi_eff_mixture:
            all_params.extend(
                [
                    (P.EFFECTIVE_SPIN.value + "_comp1_high_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_comp1_high_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_comp1_loc_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_comp1_loc_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_comp1_low_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_comp1_low_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_comp1_scale_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_comp1_scale_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_comp2_high_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_comp2_high_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_comp2_loc_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_comp2_loc_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_comp2_low_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_comp2_low_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_comp2_scale_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_comp2_scale_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_zeta_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_zeta_g", self.N_g),
                ]
            )

        if self.use_skew_normal_chi_eff:
            all_params.extend(
                [
                    (P.EFFECTIVE_SPIN.value + "_epsilon_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_epsilon_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_loc_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_loc_g", self.N_g),
                    (P.EFFECTIVE_SPIN.value + "_scale_bpl", self.N_bpl),
                    (P.EFFECTIVE_SPIN.value + "_scale_g", self.N_g),
                ]
            )

        if self.use_truncated_normal_chi_p:
            all_params.extend(
                [
                    (P.PRECESSING_SPIN.value + "_high_g", self.N_g),
                    (P.PRECESSING_SPIN.value + "_high_bpl", self.N_bpl),
                    (P.PRECESSING_SPIN.value + "_loc_g", self.N_g),
                    (P.PRECESSING_SPIN.value + "_loc_bpl", self.N_bpl),
                    (P.PRECESSING_SPIN.value + "_low_g", self.N_g),
                    (P.PRECESSING_SPIN.value + "_low_bpl", self.N_bpl),
                    (P.PRECESSING_SPIN.value + "_scale_g", self.N_g),
                    (P.PRECESSING_SPIN.value + "_scale_bpl", self.N_bpl),
                ]
            )

        if self.use_tilt:
            all_params.extend(
                [
                    ("cos_tilt_zeta_g", self.N_g),
                    ("cos_tilt_zeta_bpl", self.N_bpl),
                    (P.COS_TILT_1.value + "_loc_bpl", self.N_bpl),
                    (P.COS_TILT_1.value + "_loc_g", self.N_g),
                    (P.COS_TILT_1.value + "_minimum_bpl", self.N_bpl),
                    (P.COS_TILT_1.value + "_minimum_g", self.N_g),
                    (P.COS_TILT_1.value + "_scale_bpl", self.N_bpl),
                    (P.COS_TILT_1.value + "_scale_g", self.N_g),
                    (P.COS_TILT_2.value + "_loc_bpl", self.N_bpl),
                    (P.COS_TILT_2.value + "_loc_g", self.N_g),
                    (P.COS_TILT_2.value + "_minimum_bpl", self.N_bpl),
                    (P.COS_TILT_2.value + "_minimum_g", self.N_g),
                    (P.COS_TILT_2.value + "_scale_bpl", self.N_bpl),
                    (P.COS_TILT_2.value + "_scale_g", self.N_g),
                ]
            )

        if self.use_eccentricity_mixture:
            all_params.extend(
                [
                    (P.ECCENTRICITY.value + "_comp1_high_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_comp1_high_g", self.N_g),
                    (P.ECCENTRICITY.value + "_comp1_loc_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_comp1_loc_g", self.N_g),
                    (P.ECCENTRICITY.value + "_comp1_low_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_comp1_low_g", self.N_g),
                    (P.ECCENTRICITY.value + "_comp1_scale_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_comp1_scale_g", self.N_g),
                    (P.ECCENTRICITY.value + "_comp2_high_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_comp2_high_g", self.N_g),
                    (P.ECCENTRICITY.value + "_comp2_loc_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_comp2_loc_g", self.N_g),
                    (P.ECCENTRICITY.value + "_comp2_low_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_comp2_low_g", self.N_g),
                    (P.ECCENTRICITY.value + "_comp2_scale_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_comp2_scale_g", self.N_g),
                    (P.ECCENTRICITY.value + "_zeta_bpl", self.N_bpl),
                    (P.ECCENTRICITY.value + "_zeta_g", self.N_g),
                ]
            )

        if self.use_redshift:
            all_params.extend(
                [
                    (P.REDSHIFT.value + "_kappa_g", self.N_g),
                    (P.REDSHIFT.value + "_kappa_bpl", self.N_bpl),
                    (P.REDSHIFT.value + "_z_max_g", self.N_g),
                    (P.REDSHIFT.value + "_z_max_bpl", self.N_bpl),
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
        "--n-bpl",
        type=int,
        help="Number of broken power-law components in the mass model.",
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
        "--add-skew-normal-chi-eff",
        action="store_true",
        help="Include skew normal chi_eff parameters in the model.",
    )
    model_group.add_argument(
        "--add-truncated-normal-chi-p",
        action="store_true",
        help="Include truncated normal chi_p parameters in the model.",
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

    class NBrokenPowerlawMGaussianFSage(NBrokenPowerlawMGaussianCore, FlowMCBased):
        pass

    NBrokenPowerlawMGaussianFSage(
        N_bpl=args.n_bpl,
        N_g=args.n_g,
        use_beta_spin_magnitude=args.add_beta_spin_magnitude,
        use_spin_magnitude_mixture=args.add_spin_magnitude_mixture,
        use_chi_eff_mixture=args.add_chi_eff_mixture,
        use_skew_normal_chi_eff=args.add_skew_normal_chi_eff,
        use_truncated_normal_chi_p=args.add_truncated_normal_chi_p,
        use_tilt=args.add_tilt,
        use_eccentricity_mixture=args.add_eccentricity_mixture,
        use_redshift=args.add_redshift,
        likelihood_fn=flowMC_poisson_likelihood,
        posterior_regex=args.posterior_regex,
        read_reference_prior=args.read_reference_prior,
        n_pe_samples=args.n_pe_samples,
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

    class NBrokenPowerlawMGaussianNSage(NBrokenPowerlawMGaussianCore, NumpyroBased):
        pass

    NBrokenPowerlawMGaussianNSage(
        N_bpl=args.n_bpl,
        N_g=args.n_g,
        use_beta_spin_magnitude=args.add_beta_spin_magnitude,
        use_spin_magnitude_mixture=args.add_spin_magnitude_mixture,
        use_chi_eff_mixture=args.add_chi_eff_mixture,
        use_skew_normal_chi_eff=args.add_skew_normal_chi_eff,
        use_truncated_normal_chi_p=args.add_truncated_normal_chi_p,
        use_tilt=args.add_tilt,
        use_eccentricity_mixture=args.add_eccentricity_mixture,
        use_redshift=args.add_redshift,
        likelihood_fn=numpyro_poisson_likelihood,
        posterior_regex=args.posterior_regex,
        read_reference_prior=args.read_reference_prior,
        n_pe_samples=args.n_pe_samples,
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
