# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional, Tuple, Union

from jax import numpy as jnp
from jaxtyping import Array, ArrayLike
from numpyro._typing import DistributionLike
from numpyro.distributions.distribution import enable_validation

import gwkokab
from gwkokab.inference import numpyro_poisson_likelihood, poisson_likelihood
from gwkokab.models import NPowerlawMGaussian
from gwkokab.models.npowerlawmgaussian._ncombination import (
    create_truncated_normal_distributions,
)
from gwkokab.models.utils import JointDistribution
from gwkokab.parameters import Parameters
from gwkokab.poisson_mean import PoissonMean
from kokab.utils.checks import check_min_concentration_for_beta_dist
from kokab.utils.common import expand_arguments
from kokab.utils.flowMC_based import flowMC_arg_parser, FlowMCBased
from kokab.utils.numpyro_based import numpyro_arg_parser, NumpyroBased
from kokab.utils.sage import Sage, sage_arg_parser


def where_fns_list(has_beta_spin: bool) -> Optional[List[Callable[..., Array]]]:
    where_fns = []

    if has_beta_spin:

        def mean_variance_check(N_pl: int, N_g: int, **kwargs) -> Array:
            if N_pl > 0:
                means_pl = jnp.stack(
                    [kwargs[f"chi{i}_mean_pl_{j}"] for j in range(N_pl) for i in (1, 2)]
                )
                vars_pl = jnp.stack(
                    [
                        kwargs[f"chi{i}_variance_pl_{j}"]
                        for j in range(N_pl)
                        for i in (1, 2)
                    ]
                )
            if N_g > 0:
                means_g = jnp.stack(
                    [kwargs[f"chi{i}_mean_g_{j}"] for j in range(N_g) for i in (1, 2)]
                )
                vars_g = jnp.stack(
                    [
                        kwargs[f"chi{i}_variance_g_{j}"]
                        for j in range(N_g)
                        for i in (1, 2)
                    ]
                )

            if N_pl > 0 and N_g > 0:
                means = jnp.concatenate([means_pl, means_g])
                variances = jnp.concatenate([vars_pl, vars_g])
            elif N_pl > 0:
                means = means_pl
                variances = vars_pl
            else:
                means = means_g
                variances = vars_g

            checks = check_min_concentration_for_beta_dist(means, variances)
            return jnp.all(checks)

        where_fns.append(mean_variance_check)
    return where_fns if len(where_fns) > 0 else None


class NPowerlawMGaussianCore(Sage):
    def __init__(
        self,
        N_pl: int,
        N_g: int,
        has_beta_spin: bool,
        has_truncated_normal_spin: bool,
        has_tilt: bool,
        has_eccentricity: bool,
        has_redshift: bool,
        has_cos_iota: bool,
        has_phi_12: bool,
        has_polarization_angle: bool,
        has_right_ascension: bool,
        has_sin_declination: bool,
        has_detection_time: bool,
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
        self.N_pl = N_pl
        self.N_g = N_g
        self.has_beta_spin = has_beta_spin
        self.has_truncated_normal_spin = has_truncated_normal_spin
        if self.has_truncated_normal_spin:
            gwkokab.models.npowerlawmgaussian._model.build_spin_distributions = (
                create_truncated_normal_distributions
            )
        self.has_tilt = has_tilt
        self.has_eccentricity = has_eccentricity
        self.has_redshift = has_redshift
        self.has_cos_iota = has_cos_iota
        self.has_phi_12 = has_phi_12
        self.has_polarization_angle = has_polarization_angle
        self.has_right_ascension = has_right_ascension
        self.has_sin_declination = has_sin_declination
        self.has_detection_time = has_detection_time

        super().__init__(
            likelihood_fn=likelihood_fn,
            model=NPowerlawMGaussian,
            posterior_regex=posterior_regex,
            posterior_columns=posterior_columns,
            seed=seed,
            prior_filename=prior_filename,
            selection_fn_filename=selection_fn_filename,
            poisson_mean_filename=poisson_mean_filename,
            sampler_settings_filename=sampler_settings_filename,
            analysis_name="n_pls_m_gs",
            n_buckets=n_buckets,
            threshold=threshold,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            where_fns=where_fns_list(has_beta_spin=has_beta_spin),
        )

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        return {
            "N_pl": self.N_pl,
            "N_g": self.N_g,
            "use_spin": self.has_beta_spin or self.has_truncated_normal_spin,
            "use_tilt": self.has_tilt,
            "use_eccentricity": self.has_eccentricity,
            "use_redshift": self.has_redshift,
            "use_cos_iota": self.has_cos_iota,
            "use_phi_12": self.has_phi_12,
            "use_polarization_angle": self.has_polarization_angle,
            "use_right_ascension": self.has_right_ascension,
            "use_sin_declination": self.has_sin_declination,
            "use_detection_time": self.has_detection_time,
        }

    @property
    def parameters(self) -> List[str]:
        names = [
            Parameters.PRIMARY_MASS_SOURCE.value,
            Parameters.SECONDARY_MASS_SOURCE.value,
        ]
        if self.has_beta_spin or self.has_truncated_normal_spin:
            names.append(Parameters.PRIMARY_SPIN_MAGNITUDE.value)
            names.append(Parameters.SECONDARY_SPIN_MAGNITUDE.value)
        if self.has_tilt:
            names.extend([Parameters.COS_TILT_1.value, Parameters.COS_TILT_2.value])
        if self.has_phi_12:
            names.append(Parameters.PHI_12.value)
        if self.has_eccentricity:
            names.append(Parameters.ECCENTRICITY.value)
        if self.has_redshift:
            names.append(Parameters.REDSHIFT.value)
        if self.has_right_ascension:
            names.append(Parameters.RIGHT_ASCENSION.value)
        if self.has_sin_declination:
            names.append(Parameters.SIN_DECLINATION.value)
        if self.has_detection_time:
            names.append(Parameters.DETECTION_TIME.value)
        if self.has_cos_iota:
            names.append(Parameters.COS_IOTA.value)
        if self.has_polarization_angle:
            names.append(Parameters.POLARIZATION_ANGLE.value)
        return names

    @property
    def model_parameters(self) -> List[str]:
        all_params: List[Tuple[str, int]] = [
            ("log_rate", self.N_pl + self.N_g),
            ("alpha_pl", self.N_pl),
            ("beta_pl", self.N_pl),
            ("m1_loc_g", self.N_g),
            ("m2_loc_g", self.N_g),
            ("m1_scale_g", self.N_g),
            ("m2_scale_g", self.N_g),
            ("m1_low_g", self.N_g),
            ("m2_low_g", self.N_g),
            ("m1_high_g", self.N_g),
            ("m2_high_g", self.N_g),
            ("mmax_pl", self.N_pl),
            ("mmin_pl", self.N_pl),
        ]

        if self.has_truncated_normal_spin:
            all_params.extend(
                [
                    ("chi1_high_g", self.N_g),
                    ("chi1_high_pl", self.N_pl),
                    ("chi1_loc_g", self.N_g),
                    ("chi1_loc_pl", self.N_pl),
                    ("chi1_low_g", self.N_g),
                    ("chi1_low_pl", self.N_pl),
                    ("chi1_scale_g", self.N_g),
                    ("chi1_scale_pl", self.N_pl),
                    ("chi2_high_g", self.N_g),
                    ("chi2_high_pl", self.N_pl),
                    ("chi2_loc_g", self.N_g),
                    ("chi2_loc_pl", self.N_pl),
                    ("chi2_low_g", self.N_g),
                    ("chi2_low_pl", self.N_pl),
                    ("chi2_scale_g", self.N_g),
                    ("chi2_scale_pl", self.N_pl),
                ]
            )
        if self.has_beta_spin:
            all_params.extend(
                [
                    ("chi1_mean_g", self.N_g),
                    ("chi1_mean_pl", self.N_pl),
                    ("chi1_variance_g", self.N_g),
                    ("chi1_variance_pl", self.N_pl),
                    ("chi2_mean_g", self.N_g),
                    ("chi2_mean_pl", self.N_pl),
                    ("chi2_variance_g", self.N_g),
                    ("chi2_variance_pl", self.N_pl),
                ]
            )

        if self.has_tilt:
            all_params.extend(
                [
                    ("cos_tilt_zeta_g", self.N_g),
                    ("cos_tilt_zeta_pl", self.N_pl),
                    ("cos_tilt1_scale_g", self.N_g),
                    ("cos_tilt1_scale_pl", self.N_pl),
                    ("cos_tilt2_scale_g", self.N_g),
                    ("cos_tilt2_scale_pl", self.N_pl),
                ]
            )

        if self.has_phi_12:
            all_params.extend(
                [
                    (Parameters.PHI_12.value + "_high_g", self.N_g),
                    (Parameters.PHI_12.value + "_high_pl", self.N_pl),
                    (Parameters.PHI_12.value + "_loc_g", self.N_g),
                    (Parameters.PHI_12.value + "_loc_pl", self.N_pl),
                    (Parameters.PHI_12.value + "_low_g", self.N_g),
                    (Parameters.PHI_12.value + "_low_pl", self.N_pl),
                    (Parameters.PHI_12.value + "_scale_g", self.N_g),
                    (Parameters.PHI_12.value + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_eccentricity:
            all_params.extend(
                [
                    ("ecc_high_g", self.N_g),
                    ("ecc_high_pl", self.N_pl),
                    ("ecc_loc_g", self.N_g),
                    ("ecc_loc_pl", self.N_pl),
                    ("ecc_low_g", self.N_g),
                    ("ecc_low_pl", self.N_pl),
                    ("ecc_scale_g", self.N_g),
                    ("ecc_scale_pl", self.N_pl),
                ]
            )

        if self.has_redshift:
            all_params.extend(
                [
                    (Parameters.REDSHIFT.value + "_kappa_g", self.N_g),
                    (Parameters.REDSHIFT.value + "_kappa_pl", self.N_pl),
                    (Parameters.REDSHIFT.value + "_z_max_g", self.N_g),
                    (Parameters.REDSHIFT.value + "_z_max_pl", self.N_pl),
                ]
            )

        if self.has_right_ascension:
            all_params.extend(
                [
                    (Parameters.RIGHT_ASCENSION.value + "_high_g", self.N_g),
                    (Parameters.RIGHT_ASCENSION.value + "_high_pl", self.N_pl),
                    (Parameters.RIGHT_ASCENSION.value + "_loc_g", self.N_g),
                    (Parameters.RIGHT_ASCENSION.value + "_loc_pl", self.N_pl),
                    (Parameters.RIGHT_ASCENSION.value + "_low_g", self.N_g),
                    (Parameters.RIGHT_ASCENSION.value + "_low_pl", self.N_pl),
                    (Parameters.RIGHT_ASCENSION.value + "_scale_g", self.N_g),
                    (Parameters.RIGHT_ASCENSION.value + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_sin_declination:
            all_params.extend(
                [
                    (Parameters.SIN_DECLINATION.value + "_high_g", self.N_g),
                    (Parameters.SIN_DECLINATION.value + "_high_pl", self.N_pl),
                    (Parameters.SIN_DECLINATION.value + "_loc_g", self.N_g),
                    (Parameters.SIN_DECLINATION.value + "_loc_pl", self.N_pl),
                    (Parameters.SIN_DECLINATION.value + "_low_g", self.N_g),
                    (Parameters.SIN_DECLINATION.value + "_low_pl", self.N_pl),
                    (Parameters.SIN_DECLINATION.value + "_scale_g", self.N_g),
                    (Parameters.SIN_DECLINATION.value + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_detection_time:
            all_params.extend(
                [
                    (Parameters.DETECTION_TIME.value + "_high_g", self.N_g),
                    (Parameters.DETECTION_TIME.value + "_high_pl", self.N_pl),
                    (Parameters.DETECTION_TIME.value + "_low_g", self.N_g),
                    (Parameters.DETECTION_TIME.value + "_low_pl", self.N_pl),
                ]
            )

        if self.has_cos_iota:
            all_params.extend(
                [
                    (Parameters.COS_IOTA.value + "_high_g", self.N_g),
                    (Parameters.COS_IOTA.value + "_high_pl", self.N_pl),
                    (Parameters.COS_IOTA.value + "_loc_g", self.N_g),
                    (Parameters.COS_IOTA.value + "_loc_pl", self.N_pl),
                    (Parameters.COS_IOTA.value + "_low_g", self.N_g),
                    (Parameters.COS_IOTA.value + "_low_pl", self.N_pl),
                    (Parameters.COS_IOTA.value + "_scale_g", self.N_g),
                    (Parameters.COS_IOTA.value + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_polarization_angle:
            all_params.extend(
                [
                    (Parameters.POLARIZATION_ANGLE.value + "_high_g", self.N_g),
                    (Parameters.POLARIZATION_ANGLE.value + "_high_pl", self.N_pl),
                    (Parameters.POLARIZATION_ANGLE.value + "_loc_g", self.N_g),
                    (Parameters.POLARIZATION_ANGLE.value + "_loc_pl", self.N_pl),
                    (Parameters.POLARIZATION_ANGLE.value + "_low_g", self.N_g),
                    (Parameters.POLARIZATION_ANGLE.value + "_low_pl", self.N_pl),
                    (Parameters.POLARIZATION_ANGLE.value + "_scale_g", self.N_g),
                    (Parameters.POLARIZATION_ANGLE.value + "_scale_pl", self.N_pl),
                ]
            )

        extended_params = []
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

    spin_group = model_group.add_mutually_exclusive_group()
    spin_group.add_argument(
        "--add-beta-spin",
        action="store_true",
        help="Include beta spin parameters in the model.",
    )
    spin_group.add_argument(
        "--add-truncated-normal-spin",
        action="store_true",
        help="Include truncated normal spin parameters in the model.",
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
    model_group.add_argument(
        "--add-truncated-normal-eccentricity",
        action="store_true",
        help="Include truncated normal eccentricity in the model.",
    )
    model_group.add_argument(
        "--add-cos-iota",
        action="store_true",
        help="Include cos_iota parameter in the model",
    )
    model_group.add_argument(
        "--add-phi-12",
        action="store_true",
        help="Include phi_12 parameter in the model",
    )
    model_group.add_argument(
        "--add-polarization-angle",
        action="store_true",
        help="Include polarization_angle parameter in the model",
    )
    model_group.add_argument(
        "--add-right-ascension",
        action="store_true",
        help="Include right_ascension parameter in the model",
    )
    model_group.add_argument(
        "--add-sin-declination",
        action="store_true",
        help="Include sin_declination parameter in the model",
    )
    model_group.add_argument(
        "--add-detection-time",
        action="store_true",
        help="Include detection_time parameter in the model",
    )
    return parser


def f_main() -> None:
    enable_validation()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = sage_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    NPowerlawMGaussianFSage(
        N_pl=args.n_pl,
        N_g=args.n_g,
        has_beta_spin=args.add_beta_spin,
        has_truncated_normal_spin=args.add_truncated_normal_spin,
        has_tilt=args.add_tilt,
        has_eccentricity=args.add_truncated_normal_eccentricity,
        has_redshift=args.add_redshift,
        has_cos_iota=args.add_cos_iota,
        has_phi_12=args.add_phi_12,
        has_polarization_angle=args.add_polarization_angle,
        has_right_ascension=args.add_right_ascension,
        has_sin_declination=args.add_sin_declination,
        has_detection_time=args.add_detection_time,
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

    NPowerlawMGaussianNSage(
        N_pl=args.n_pl,
        N_g=args.n_g,
        has_beta_spin=args.add_beta_spin,
        has_truncated_normal_spin=args.add_truncated_normal_spin,
        has_tilt=args.add_tilt,
        has_eccentricity=args.add_truncated_normal_eccentricity,
        has_redshift=args.add_redshift,
        has_cos_iota=args.add_cos_iota,
        has_phi_12=args.add_phi_12,
        has_polarization_angle=args.add_polarization_angle,
        has_right_ascension=args.add_right_ascension,
        has_sin_declination=args.add_sin_declination,
        has_detection_time=args.add_detection_time,
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
