# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``analytical_gwalk_subpopulation`` console script: sub-population inference over
per-event Gaussian summaries.

One cell of the analysis matrix. The family half is :class:`~gwkokab.analysis.subpopulation.common.SubPopulationModelCore`, which names the
parameters and hyper-parameters; the data-representation half is :class:`~gwkokab.analysis.core.analytical_gwalk_base.AnalyticalGWalkBase`, which
supplies ``read_data`` and ``run``. The sampler half is chosen at *runtime* by
``sampler_cfg.json``, which is why there are two trivial subclasses -- one mixing in
flowMC, one NumPyro -- and :func:`main` picks between them after reading that file.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable

from numpyro.distributions.distribution import enable_validation

from gwkokab.analysis.core.analytical_gwalk_base import (
    analytical_gwalk_arg_parser,
    AnalyticalGWalkBase,
)
from gwkokab.analysis.core.flowMC_base import FlowMCBase
from gwkokab.analysis.core.inference_io import (
    AnalyticalGWalkPELoader as DataLoader,
    SamplerConfig,
)
from gwkokab.analysis.core.numpyro_base import NumpyroBase
from gwkokab.analysis.subpopulation.common import (
    model_arg_parser,
    SubPopulationModelCore,
)
from gwkokab.analysis.utils.logger import log_info
from gwkokab.inference.factory import get_likelihood_fn
from gwkokab.models import SubPopulationModel


class SubPopulationModelAnalyticalGWalkAnalysis(
    SubPopulationModelCore, AnalyticalGWalkBase
):
    """Sub-population analysis over per-event Gaussian summaries.

    Combines :class:`~gwkokab.analysis.subpopulation.common.SubPopulationModelCore` with :class:`~gwkokab.analysis.core.analytical_gwalk_base.AnalyticalGWalkBase`. Sampler-agnostic: mix in
    :class:`~gwkokab.analysis.core.flowMC_base.FlowMCBase` or
    :class:`~gwkokab.analysis.core.numpyro_base.NumpyroBase` to get a runnable
    analysis.
    """

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
        sampler_cfg,
        variance_cut_threshold: float | None,
        n_samples: int,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        """Configure both halves of the analysis.

        The model and component arguments are forwarded to :class:`~gwkokab.analysis.subpopulation.common.SubPopulationModelCore`, and the data,
        prior, selection-function and sampler arguments to :class:`~gwkokab.analysis.core.analytical_gwalk_base.AnalyticalGWalkBase`. See those two
        constructors for the full parameter list.
        """
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

        AnalyticalGWalkBase.__init__(
            self,
            likelihood_fn,
            SubPopulationModel,
            data_loader,
            prior_filename,
            poisson_mean_filename,
            sampler_cfg,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            analysis_name="subpopulation",
            n_samples=n_samples,
            variance_cut_threshold=variance_cut_threshold,
        )


class SubPopulationModelFAnalyticalGWalkAnalysis(
    SubPopulationModelAnalyticalGWalkAnalysis, FlowMCBase
):
    """Sub-population analysis over per-event Gaussian summaries, sampled with flowMC.

    Selected by :func:`main` when ``sampler_cfg.json`` names ``flowMC``.
    """

    pass


class SubPopulationModelNAnalyticalGWalkAnalysis(
    SubPopulationModelAnalyticalGWalkAnalysis, NumpyroBase
):
    """Sub-population analysis over per-event Gaussian summaries, sampled with NumPyro.

    Selected by :func:`main` when ``sampler_cfg.json`` names ``numpyro``.
    """

    pass


def main() -> None:
    """Console script entry point for ``analytical_gwalk_subpopulation``.

    Parses the command line, reads the sampler and data loader configurations, picks the
    likelihood and the analysis class matching the configured sampler, seeds the PRNG,
    and runs the analysis.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = analytical_gwalk_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    sampler_cfg = SamplerConfig.read_from_json(args.sampler_cfg)
    data_loader = DataLoader.read_from_json(args.data_loader_cfg)

    likelihood_fn = get_likelihood_fn(
        sampler_name=sampler_cfg.sampler_name,
        analysis_type="analytical_gwalk",
    )

    AnalysisClass = (
        SubPopulationModelFAnalyticalGWalkAnalysis
        if sampler_cfg.sampler_name == "flowMC"
        else SubPopulationModelNAnalyticalGWalkAnalysis
    )

    AnalysisClass.init_rng_seed(seed=args.seed)

    AnalysisClass(
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
        likelihood_fn=likelihood_fn,
        data_loader=data_loader,
        prior_filename=args.prior_cfg,
        poisson_mean_filename=args.pmean_cfg,
        sampler_cfg=sampler_cfg,
        variance_cut_threshold=args.variance_cut_threshold,
        n_samples=args.n_samples,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
