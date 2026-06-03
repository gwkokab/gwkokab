# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable

from numpyro.distributions.distribution import enable_validation

from gwkokab.analysis.core.discrete_base import discrete_arg_parser, DiscreteBase
from gwkokab.analysis.core.flowMC_base import flowMC_arg_parser, FlowMCBase
from gwkokab.analysis.core.inference_io import (
    DiscretePELoader as DataLoader,
    SamplerConfig,
)
from gwkokab.analysis.core.numpyro_base import NumpyroBase
from gwkokab.analysis.n_pls_m_gs.common import (
    model_arg_parser,
    NPowerlawMGaussianCore,
    where_fns_list,
)
from gwkokab.analysis.utils.logger import log_info
from gwkokab.inference.factory import get_likelihood_fn
from gwkokab.models import NPowerlawMGaussian


class NPowerlawMGaussianDiscreteAnalysis(NPowerlawMGaussianCore, DiscreteBase):
    def __init__(
        self,
        N_pl: int,
        N_g: int,
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
        use_powerlaw_redshift: bool,
        use_madau_dickinson_redshift: bool,
        use_cos_iota: bool,
        use_phi_12: bool,
        use_polarization_angle: bool,
        use_right_ascension: bool,
        use_sin_declination: bool,
        use_detection_time: bool,
        use_phi_1: bool,
        use_phi_2: bool,
        use_phi_orb: bool,
        use_mean_anomaly: bool,
        likelihood_fn: Callable[..., Callable],
        data_loader: DataLoader,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_cfg,
        variance_cut_threshold: float | None,
        n_buckets: int,
        threshold: float,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        NPowerlawMGaussianCore.__init__(
            self,
            N_pl=N_pl,
            N_g=N_g,
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
            use_powerlaw_redshift=use_powerlaw_redshift,
            use_madau_dickinson_redshift=use_madau_dickinson_redshift,
            use_cos_iota=use_cos_iota,
            use_phi_12=use_phi_12,
            use_polarization_angle=use_polarization_angle,
            use_right_ascension=use_right_ascension,
            use_sin_declination=use_sin_declination,
            use_detection_time=use_detection_time,
            use_phi_1=use_phi_1,
            use_phi_2=use_phi_2,
            use_phi_orb=use_phi_orb,
            use_mean_anomaly=use_mean_anomaly,
        )

        DiscreteBase.__init__(
            self,
            likelihood_fn=likelihood_fn,
            model=NPowerlawMGaussian,
            data_loader=data_loader,
            prior_filename=prior_filename,
            poisson_mean_filename=poisson_mean_filename,
            sampler_cfg=sampler_cfg,
            variance_cut_threshold=variance_cut_threshold,
            analysis_name="n_pls_m_gs",
            n_buckets=n_buckets,
            threshold=threshold,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            where_fns=where_fns_list(use_beta_spin_magnitude=use_beta_spin_magnitude),
        )


class NPowerlawMGaussianFDiscreteAnalysis(
    NPowerlawMGaussianDiscreteAnalysis, FlowMCBase
):
    pass


class NPowerlawMGaussianNDiscreteAnalysis(
    NPowerlawMGaussianDiscreteAnalysis, NumpyroBase
):
    pass


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = discrete_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    sampler_cfg = SamplerConfig.from_json(args.sampler_cfg)
    data_loader = DataLoader.from_json(args.data_loader_cfg)

    likelihood_fn = get_likelihood_fn(
        sampler_name=sampler_cfg.sampler_name,
        analysis_type="discrete",
    )

    AnalysisClass = (
        NPowerlawMGaussianFDiscreteAnalysis
        if sampler_cfg.sampler_name == "flowMC"
        else NPowerlawMGaussianNDiscreteAnalysis
    )

    AnalysisClass.init_rng_seed(seed=args.seed)

    AnalysisClass(
        N_pl=args.n_pl,
        N_g=args.n_g,
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
        use_powerlaw_redshift=args.add_powerlaw_redshift,
        use_madau_dickinson_redshift=args.add_madau_dickinson_redshift,
        use_cos_iota=args.add_cos_iota,
        use_phi_12=args.add_phi_12,
        use_polarization_angle=args.add_polarization_angle,
        use_right_ascension=args.add_right_ascension,
        use_sin_declination=args.add_sin_declination,
        use_detection_time=args.add_detection_time,
        use_phi_1=args.add_phi_1,
        use_phi_2=args.add_phi_2,
        use_phi_orb=args.add_phi_orb,
        use_mean_anomaly=args.add_mean_anomaly,
        likelihood_fn=likelihood_fn,
        data_loader=data_loader,
        prior_filename=args.prior_cfg,
        poisson_mean_filename=args.pmean_cfg,
        sampler_cfg=sampler_cfg,
        variance_cut_threshold=args.variance_cut_threshold,
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
