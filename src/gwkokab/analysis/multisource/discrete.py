# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``discrete_multisource`` console script: multi-source inference over per-event
posterior samples.

One cell of the analysis matrix. The family half is :class:`~gwkokab.analysis.multisource.common.MultiSourceModelCore`, which names the
parameters and hyper-parameters; the data-representation half is :class:`~gwkokab.analysis.core.discrete_base.DiscreteBase`, which
supplies ``read_data`` and ``run``. The sampler half is chosen at *runtime* by
``sampler_cfg.json``, which is why there are two trivial subclasses -- one mixing in
flowMC, one NumPyro -- and :func:`main` picks between them after reading that file.
"""

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
from gwkokab.analysis.multisource.common import (
    model_arg_parser,
    MultiSourceModelCore,
    where_fns_list,
)
from gwkokab.analysis.utils.logger import log_info
from gwkokab.inference.factory import get_likelihood_fn
from gwkokab.models import MultiSourceModel


class MultiSourceModelDiscreteAnalysis(MultiSourceModelCore, DiscreteBase):
    """Multi-source analysis over per-event posterior samples.

    Combines :class:`~gwkokab.analysis.multisource.common.MultiSourceModelCore` with :class:`~gwkokab.analysis.core.discrete_base.DiscreteBase`. Sampler-agnostic: mix in
    :class:`~gwkokab.analysis.core.flowMC_base.FlowMCBase` or
    :class:`~gwkokab.analysis.core.numpyro_base.NumpyroBase` to get a runnable
    analysis.
    """

    def __init__(
        self,
        N_spl: int,
        N_bpl: int,
        N_gpl: int,
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
        n_buckets: int,
        threshold: float,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
    ) -> None:
        """Configure both halves of the analysis.

        The model and component arguments are forwarded to :class:`~gwkokab.analysis.multisource.common.MultiSourceModelCore`, and the data,
        prior, selection-function and sampler arguments to :class:`~gwkokab.analysis.core.discrete_base.DiscreteBase`. See those two
        constructors for the full parameter list.
        """
        MultiSourceModelCore.__init__(
            self,
            N_spl=N_spl,
            N_bpl=N_bpl,
            N_gpl=N_gpl,
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
            use_eccentricity_powerlaw=use_eccentricity_powerlaw,
            use_mean_anomaly=use_mean_anomaly,
            use_powerlaw_redshift=use_powerlaw_redshift,
            use_madau_dickinson_redshift=use_madau_dickinson_redshift,
        )

        DiscreteBase.__init__(
            self,
            likelihood_fn=likelihood_fn,
            model=MultiSourceModel,
            data_loader=data_loader,
            prior_filename=prior_filename,
            poisson_mean_filename=poisson_mean_filename,
            sampler_cfg=sampler_cfg,
            variance_cut_threshold=variance_cut_threshold,
            analysis_name="multisource",
            n_buckets=n_buckets,
            threshold=threshold,
            debug_nans=debug_nans,
            profile_memory=profile_memory,
            check_leaks=check_leaks,
            where_fns=where_fns_list(use_beta_spin_magnitude=use_beta_spin_magnitude),
        )


class MultiSourceModelFDiscreteAnalysis(MultiSourceModelDiscreteAnalysis, FlowMCBase):
    """Multi-source analysis over per-event posterior samples, sampled with flowMC.

    Selected by :func:`main` when ``sampler_cfg.json`` names ``flowMC``.
    """

    pass


class MultiSourceModelNDiscreteAnalysis(MultiSourceModelDiscreteAnalysis, NumpyroBase):
    """Multi-source analysis over per-event posterior samples, sampled with NumPyro.

    Selected by :func:`main` when ``sampler_cfg.json`` names ``numpyro``.
    """

    pass


def main() -> None:
    """Console script entry point for ``discrete_multisource``.

    Parses the command line, reads the sampler and data loader configurations, picks the
    likelihood and the analysis class matching the configured sampler, seeds the PRNG,
    and runs the analysis.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = model_arg_parser(parser)
    parser = discrete_arg_parser(parser)
    parser = flowMC_arg_parser(parser)

    args = parser.parse_args()

    enable_validation()

    log_info(start=True)

    sampler_cfg = SamplerConfig.read_from_json(args.sampler_cfg)
    data_loader = DataLoader.read_from_json(args.data_loader_cfg)

    likelihood_fn = get_likelihood_fn(
        sampler_name=sampler_cfg.sampler_name,
        analysis_type="discrete",
    )

    AnalysisClass = (
        MultiSourceModelFDiscreteAnalysis
        if sampler_cfg.sampler_name == "flowMC"
        else MultiSourceModelNDiscreteAnalysis
    )

    AnalysisClass.init_rng_seed(seed=args.seed)

    AnalysisClass(
        N_spl=args.n_spl,
        N_bpl=args.n_bpl,
        N_gpl=args.n_gpl,
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
        n_buckets=args.n_buckets,
        threshold=args.threshold,
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
    ).run()
