# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Callable, Dict, List, Optional

from jaxtyping import Array, ArrayLike
from numpyro.distributions.distribution import Distribution, enable_validation

from gwkanal.core.flowMC_based import flowMC_arg_parser, FlowMCBased
from gwkanal.core.inference_io import DiscretePELoader as DataLoader
from gwkanal.core.numpyro_based import numpyro_arg_parser, NumpyroBased
from gwkanal.core.sage import Sage, sage_arg_parser
from gwkanal.o4_n_bpls_m_gs.common import (
    model_arg_parser,
    NBrokenPowerlawMGaussianCore,
    where_fns_list,
)
from gwkanal.utils.logger import log_info
from gwkokab.inference import flowMC_poisson_likelihood, numpyro_poisson_likelihood
from gwkokab.models import NBrokenPowerlawMGaussian
from gwkokab.models.utils import JointDistribution, ScaledMixture


class NBrokenPowerlawMGaussianSage(NBrokenPowerlawMGaussianCore, Sage):
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
                Callable[..., Distribution],
                JointDistribution,
                Dict[str, Distribution],
                Dict[str, int],
                ArrayLike,
                Callable[[ScaledMixture], Array],
                Optional[List[Callable[..., Array]]],
                Dict[str, Array],
            ],
            Callable,
        ],
        data_loader: DataLoader,
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

        Sage.__init__(
            self,
            likelihood_fn=likelihood_fn,
            model=NBrokenPowerlawMGaussian,
            data_loader=data_loader,
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


class NBrokenPowerlawMGaussianFSage(NBrokenPowerlawMGaussianSage, FlowMCBased):
    pass


class NBrokenPowerlawMGaussianNSage(NBrokenPowerlawMGaussianSage, NumpyroBased):
    pass


def f_main() -> None:
    enable_validation()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = sage_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    log_info(start=True)

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    NBrokenPowerlawMGaussianFSage.init_rng_seed(seed=args.seed)

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
        data_loader=data_loader,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_cfg,
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

    data_loader = DataLoader.from_json(args.data_loader_cfg)

    NBrokenPowerlawMGaussianNSage.init_rng_seed(seed=args.seed)

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
        data_loader=data_loader,
        prior_filename=args.prior_json,
        poisson_mean_filename=args.pmean_cfg,
        sampler_settings_filename=args.sampler_config,
        variance_cut_threshold=args.variance_cut_threshold,
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
