# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable

from numpyro.distributions.distribution import enable_validation

from gwkanal.core.flowMC_based import FlowMCBased
from gwkanal.core.inference_io import AnalyticalPELoader as DataLoader
from gwkanal.core.monk import Monk, monk_arg_parser
from gwkanal.core.numpyro_based import NumpyroBased
from gwkanal.subpopulation.common import model_arg_parser, SubPopulationModelCore
from gwkanal.utils.logger import log_info
from gwkokab.inference import (
    flowMC_analytical_poisson_likelihood,
    numpyro_analytical_poisson_likelihood,
)
from gwkokab.models import SubPopulationModel


class SubPopulationModelMonk(SubPopulationModelCore, Monk):
    def __init__(
        self,
        N_spl: int,
        N_bpl: int,
        N_gpl: int,
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
        use_eccentricity_powerlaw: bool,
        use_mean_anomaly: bool,
        use_powerlaw_redshift: bool,
        use_madau_dickinson_redshift: bool,
        likelihood_fn: Callable[..., Callable],
        data_loader: DataLoader,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        variance_cut_threshold: float | None,
        n_samples: int,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        SubPopulationModelCore.__init__(
            self,
            N_spl=N_spl,
            N_bpl=N_bpl,
            N_gpl=N_gpl,
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
            use_eccentricity_powerlaw=use_eccentricity_powerlaw,
            use_mean_anomaly=use_mean_anomaly,
            use_powerlaw_redshift=use_powerlaw_redshift,
            use_madau_dickinson_redshift=use_madau_dickinson_redshift,
        )

        Monk.__init__(
            self,
            likelihood_fn,
            SubPopulationModel,
            data_loader,
            prior_filename,
            poisson_mean_filename,
            sampler_settings_filename,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            analysis_name="subpopulation",
            n_samples=n_samples,
            variance_cut_threshold=variance_cut_threshold,
        )


class SubPopulationModelFMonk(SubPopulationModelMonk, FlowMCBased):
    pass


class SubPopulationModelNMonk(SubPopulationModelMonk, NumpyroBased):
    pass


def f_main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    SubPopulationModelFMonk.init_rng_seed(seed=args.seed)

    SubPopulationModelFMonk(
        N_spl=args.n_spl,
        N_bpl=args.n_bpl,
        N_gpl=args.n_gpl,
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
        use_eccentricity_powerlaw=args.add_eccentricity_powerlaw,
        use_mean_anomaly=args.add_mean_anomaly,
        use_powerlaw_redshift=args.add_powerlaw_redshift,
        use_madau_dickinson_redshift=args.add_madau_dickinson_redshift,
        likelihood_fn=flowMC_analytical_poisson_likelihood,
        data_loader=data_loader,
        prior_filename=args.prior_cfg,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_cfg,
        variance_cut_threshold=args.variance_cut_threshold,
        n_samples=args.n_samples,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()


def n_main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    SubPopulationModelNMonk.init_rng_seed(seed=args.seed)

    SubPopulationModelNMonk(
        N_spl=args.n_spl,
        N_bpl=args.n_bpl,
        N_gpl=args.n_gpl,
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
        use_eccentricity_powerlaw=args.add_eccentricity_powerlaw,
        use_mean_anomaly=args.add_mean_anomaly,
        use_powerlaw_redshift=args.add_powerlaw_redshift,
        use_madau_dickinson_redshift=args.add_madau_dickinson_redshift,
        likelihood_fn=numpyro_analytical_poisson_likelihood,
        data_loader=data_loader,
        prior_filename=args.prior_cfg,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_cfg,
        variance_cut_threshold=args.variance_cut_threshold,
        n_samples=args.n_samples,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
