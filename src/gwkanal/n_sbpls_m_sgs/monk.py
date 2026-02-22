# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from gwkanal.core.inference_io import AnalyticalPELoader as DataLoader
from gwkanal.core.monk import Monk, monk_arg_parser
from gwkanal.n_sbpls_m_sgs.common import (
    model_arg_parser,
    NSmoothedBrokenPowerlawMSmoothedGaussianCore,
)
from gwkanal.utils.logger import log_info
from gwkokab.models import NSmoothedBrokenPowerlawMSmoothedGaussian


class NSmoothedBrokenPowerlawMSmoothedGaussianMonk(
    NSmoothedBrokenPowerlawMSmoothedGaussianCore, Monk
):
    def __init__(
        self,
        N_sbpl: int,
        N_sgpl: int,
        N_gg: int,
        use_beta_spin_magnitude: bool,
        use_spin_magnitude_mixture: bool,
        use_truncated_normal_spin_x: bool,
        use_truncated_normal_spin_y: bool,
        use_truncated_normal_spin_z: bool,
        use_chi_eff_mixture: bool,
        use_skew_normal_chi_eff: bool,
        use_truncated_normal_chi_p: bool,
        use_tilt: bool,
        use_eccentricity_mixture: bool,
        use_redshift: bool,
        use_cos_iota: bool,
        use_phi_12: bool,
        use_polarization_angle: bool,
        use_right_ascension: bool,
        use_sin_declination: bool,
        use_detection_time: bool,
        data_loader: DataLoader,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_samples: int,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        NSmoothedBrokenPowerlawMSmoothedGaussianCore.__init__(
            self,
            N_sbpl=N_sbpl,
            N_sgpl=N_sgpl,
            N_gg=N_gg,
            use_beta_spin_magnitude=use_beta_spin_magnitude,
            use_spin_magnitude_mixture=use_spin_magnitude_mixture,
            use_truncated_normal_spin_x=use_truncated_normal_spin_x,
            use_truncated_normal_spin_y=use_truncated_normal_spin_y,
            use_truncated_normal_spin_z=use_truncated_normal_spin_z,
            use_chi_eff_mixture=use_chi_eff_mixture,
            use_skew_normal_chi_eff=use_skew_normal_chi_eff,
            use_truncated_normal_chi_p=use_truncated_normal_chi_p,
            use_tilt=use_tilt,
            use_eccentricity_mixture=use_eccentricity_mixture,
            use_redshift=use_redshift,
            use_cos_iota=use_cos_iota,
            use_phi_12=use_phi_12,
            use_polarization_angle=use_polarization_angle,
            use_right_ascension=use_right_ascension,
            use_sin_declination=use_sin_declination,
            use_detection_time=use_detection_time,
        )

        Monk.__init__(
            self,
            NSmoothedBrokenPowerlawMSmoothedGaussian,
            data_loader,
            prior_filename,
            poisson_mean_filename,
            sampler_settings_filename,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            analysis_name="n_sbpls_m_sgs",
            n_samples=n_samples,
        )


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    NSmoothedBrokenPowerlawMSmoothedGaussianMonk.init_rng_seed(seed=args.seed)

    NSmoothedBrokenPowerlawMSmoothedGaussianMonk(
        N_sbpl=args.n_sbpl,
        N_sgpl=args.n_sgpl,
        N_gg=args.n_gg,
        use_beta_spin_magnitude=args.add_beta_spin_magnitude,
        use_spin_magnitude_mixture=args.add_spin_magnitude_mixture,
        use_truncated_normal_spin_x=args.add_truncated_normal_spin_x,
        use_truncated_normal_spin_y=args.add_truncated_normal_spin_y,
        use_truncated_normal_spin_z=args.add_truncated_normal_spin_z,
        use_chi_eff_mixture=args.add_chi_eff_mixture,
        use_skew_normal_chi_eff=args.add_skew_normal_chi_eff,
        use_truncated_normal_chi_p=args.add_truncated_normal_chi_p,
        use_tilt=args.add_tilt,
        use_eccentricity_mixture=args.add_eccentricity_mixture,
        use_redshift=args.add_redshift,
        use_cos_iota=args.add_cos_iota,
        use_phi_12=args.add_phi_12,
        use_polarization_angle=args.add_polarization_angle,
        use_right_ascension=args.add_right_ascension,
        use_sin_declination=args.add_sin_declination,
        use_detection_time=args.add_detection_time,
        data_loader=data_loader,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_config,
        n_samples=args.n_samples,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
