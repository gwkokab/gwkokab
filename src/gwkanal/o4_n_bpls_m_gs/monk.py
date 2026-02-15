# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from gwkanal.core.inference_io import AnalyticalPELoader as DataLoader
from gwkanal.core.monk import Monk, monk_arg_parser
from gwkanal.o4_n_bpls_m_gs.common import (
    model_arg_parser,
    NBrokenPowerlawMGaussianCore,
)
from gwkanal.utils.logger import log_info
from gwkokab.models import NBrokenPowerlawMGaussian


class NBrokenPowerlawMGaussianMonk(NBrokenPowerlawMGaussianCore, Monk):
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
        data_loader: DataLoader,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        n_samples: int,
        n_mom_samples: int,
        max_iter_mean: int,
        max_iter_cov: int,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        NBrokenPowerlawMGaussianCore.__init__(
            self,
            N_bpl=N_bpl,
            N_g=N_g,
            use_beta_spin_magnitude=use_beta_spin_magnitude,
            use_spin_magnitude_mixture=use_spin_magnitude_mixture,
            use_chi_eff_mixture=use_chi_eff_mixture,
            use_skew_normal_chi_eff=use_skew_normal_chi_eff,
            use_truncated_normal_chi_p=use_truncated_normal_chi_p,
            use_tilt=use_tilt,
            use_eccentricity_mixture=use_eccentricity_mixture,
            use_redshift=use_redshift,
        )

        Monk.__init__(
            self,
            NBrokenPowerlawMGaussian,
            data_loader,
            prior_filename,
            poisson_mean_filename,
            sampler_settings_filename,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            analysis_name="othree_n_pls_m_gs",
            n_samples=n_samples,
            n_mom_samples=n_mom_samples,
            max_iter_mean=max_iter_mean,
            max_iter_cov=max_iter_cov,
        )


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = monk_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    NBrokenPowerlawMGaussianMonk.init_rng_seed(seed=args.seed)

    NBrokenPowerlawMGaussianMonk(
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
        data_loader=data_loader,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_config,
        n_samples=args.n_samples,
        n_mom_samples=args.n_mom_samples,
        max_iter_mean=args.max_iter_mean,
        max_iter_cov=args.max_iter_cov,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
