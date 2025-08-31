# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional, Tuple, Union

from jaxtyping import Array

import gwkokab
from gwkokab.inference import numpyro_poisson_likelihood, poisson_likelihood
from gwkokab.models import NPowerlawMGaussian
from gwkokab.models.utils import create_truncated_normal_distributions
from gwkokab.parameters import (
    COS_IOTA,
    COS_TILT_1,
    COS_TILT_2,
    DETECTION_TIME,
    ECCENTRICITY,
    PHI_12,
    POLARIZATION_ANGLE,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    REDSHIFT,
    RIGHT_ASCENSION,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
    SIN_DECLINATION,
)
from kokab.utils.common import expand_arguments
from kokab.utils.flowMC_based import FlowMCBased
from kokab.utils.numpyro_based import NumpyroBased
from kokab.utils.sage import Sage, sage_arg_parser


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
        where_fns: Optional[List[Callable[..., Array]]],
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
            where_fns=where_fns,
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
        names = [PRIMARY_MASS_SOURCE.name, SECONDARY_MASS_SOURCE.name]
        if self.has_beta_spin or self.has_truncated_normal_spin:
            names.append(PRIMARY_SPIN_MAGNITUDE.name)
            names.append(SECONDARY_SPIN_MAGNITUDE.name)
        if self.has_tilt:
            names.extend([COS_TILT_1.name, COS_TILT_2.name])
        if self.has_phi_12:
            names.append(PHI_12.name)
        if self.has_eccentricity:
            names.append(ECCENTRICITY.name)
        if self.has_redshift:
            names.append(REDSHIFT.name)
        if self.has_right_ascension:
            names.append(RIGHT_ASCENSION.name)
        if self.has_sin_declination:
            names.append(SIN_DECLINATION.name)
        if self.has_detection_time:
            names.append(DETECTION_TIME.name)
        if self.has_cos_iota:
            names.append(COS_IOTA.name)
        if self.has_polarization_angle:
            names.append(POLARIZATION_ANGLE.name)
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
                    (PHI_12.name + "_high_g", self.N_g),
                    (PHI_12.name + "_high_pl", self.N_pl),
                    (PHI_12.name + "_loc_g", self.N_g),
                    (PHI_12.name + "_loc_pl", self.N_pl),
                    (PHI_12.name + "_low_g", self.N_g),
                    (PHI_12.name + "_low_pl", self.N_pl),
                    (PHI_12.name + "_scale_g", self.N_g),
                    (PHI_12.name + "_scale_pl", self.N_pl),
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
                    (REDSHIFT.name + "_kappa_g", self.N_g),
                    (REDSHIFT.name + "_kappa_pl", self.N_pl),
                    (REDSHIFT.name + "_z_max_g", self.N_g),
                    (REDSHIFT.name + "_z_max_pl", self.N_pl),
                ]
            )

        if self.has_right_ascension:
            all_params.extend(
                [
                    (RIGHT_ASCENSION.name + "_high_g", self.N_g),
                    (RIGHT_ASCENSION.name + "_high_pl", self.N_pl),
                    (RIGHT_ASCENSION.name + "_loc_g", self.N_g),
                    (RIGHT_ASCENSION.name + "_loc_pl", self.N_pl),
                    (RIGHT_ASCENSION.name + "_low_g", self.N_g),
                    (RIGHT_ASCENSION.name + "_low_pl", self.N_pl),
                    (RIGHT_ASCENSION.name + "_scale_g", self.N_g),
                    (RIGHT_ASCENSION.name + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_sin_declination:
            all_params.extend(
                [
                    (SIN_DECLINATION.name + "_high_g", self.N_g),
                    (SIN_DECLINATION.name + "_high_pl", self.N_pl),
                    (SIN_DECLINATION.name + "_loc_g", self.N_g),
                    (SIN_DECLINATION.name + "_loc_pl", self.N_pl),
                    (SIN_DECLINATION.name + "_low_g", self.N_g),
                    (SIN_DECLINATION.name + "_low_pl", self.N_pl),
                    (SIN_DECLINATION.name + "_scale_g", self.N_g),
                    (SIN_DECLINATION.name + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_detection_time:
            all_params.extend(
                [
                    (DETECTION_TIME.name + "_high_g", self.N_g),
                    (DETECTION_TIME.name + "_high_pl", self.N_pl),
                    (DETECTION_TIME.name + "_low_g", self.N_g),
                    (DETECTION_TIME.name + "_low_pl", self.N_pl),
                ]
            )

        if self.has_cos_iota:
            all_params.extend(
                [
                    (COS_IOTA.name + "_high_g", self.N_g),
                    (COS_IOTA.name + "_high_pl", self.N_pl),
                    (COS_IOTA.name + "_loc_g", self.N_g),
                    (COS_IOTA.name + "_loc_pl", self.N_pl),
                    (COS_IOTA.name + "_low_g", self.N_g),
                    (COS_IOTA.name + "_low_pl", self.N_pl),
                    (COS_IOTA.name + "_scale_g", self.N_g),
                    (COS_IOTA.name + "_scale_pl", self.N_pl),
                ]
            )

        if self.has_polarization_angle:
            all_params.extend(
                [
                    (POLARIZATION_ANGLE.name + "_high_g", self.N_g),
                    (POLARIZATION_ANGLE.name + "_high_pl", self.N_pl),
                    (POLARIZATION_ANGLE.name + "_loc_g", self.N_g),
                    (POLARIZATION_ANGLE.name + "_loc_pl", self.N_pl),
                    (POLARIZATION_ANGLE.name + "_low_g", self.N_g),
                    (POLARIZATION_ANGLE.name + "_low_pl", self.N_pl),
                    (POLARIZATION_ANGLE.name + "_scale_g", self.N_g),
                    (POLARIZATION_ANGLE.name + "_scale_pl", self.N_pl),
                ]
            )

        extended_params = []
        for params in all_params:
            extended_params.extend(expand_arguments(*params))
        return extended_params


class NPowerlawMGaussianFSage(NPowerlawMGaussianCore, FlowMCBased):
    likelihood_fn = poisson_likelihood


class NPowerlawMGaussianNSage(NPowerlawMGaussianCore, NumpyroBased):
    likelihood_fn = numpyro_poisson_likelihood


def parse(parser: ArgumentParser) -> ArgumentParser:
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
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_arg_parser(parser)
    parser = parse(parser)

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
        where_fns=None,  # TODO(Qazalbash): Add `where_fns`.
    ).run()


def n_main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_arg_parser(parser)
    parser = parse(parser)

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
        where_fns=None,  # TODO(Qazalbash): Add `where_fns`.
    ).run()
