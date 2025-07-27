# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from typing import List, Tuple

import numpy as np
from jax import numpy as jnp, random as jrd
from jaxtyping import Array
from loguru import logger

import gwkokab
from gwkokab.inference import Bake, poisson_likelihood
from gwkokab.models import NSmoothedPowerlawMSmoothedGaussian
from gwkokab.models.utils import (
    create_smoothed_gaussians_raw,
    create_smoothed_powerlaws_raw,
    create_truncated_normal_distributions,
)
from gwkokab.parameters import (
    COS_TILT_1,
    COS_TILT_2,
    ECCENTRICITY,
    MASS_RATIO,
    PRIMARY_MASS_SOURCE,
    PRIMARY_SPIN_MAGNITUDE,
    REDSHIFT,
    SECONDARY_MASS_SOURCE,
    SECONDARY_SPIN_MAGNITUDE,
)
from gwkokab.poisson_mean import PoissonMean
from gwkokab.utils.tools import error_if
from kokab.utils import poisson_mean_parser, sage_parser
from kokab.utils.common import (
    expand_arguments,
    flowMC_default_parameters,
    get_posterior_data,
    get_processed_priors,
    LOG_REF_PRIOR_NAME,
    read_json,
    vt_json_read_and_process,
)
from kokab.utils.flowMC_helper import flowMChandler


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = sage_parser.get_parser(parser)

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
        help="Include redshift parameters in the model.",
    )
    model_group.add_argument(
        "--add-truncated-normal-eccentricity",
        action="store_true",
        help="Include truncated normal eccentricity in the model.",
    )
    model_group.add_argument(
        "--raw",
        action="store_true",
        help="The raw parameters for this model are primary mass and mass ratio. To"
        "align with the rest of the codebase, we transform primary mass and mass ratio"
        "to primary and secondary mass. This flag will use the raw parameters i.e."
        "primary mass and mass ratio.",
    )

    return parser


def main() -> None:
    r"""Main function of the script."""
    logger.warning(
        "If you have made any changes to any parameters, please make sure"
        " that the changes are reflected in scripts that generate plots.",
    )

    parser = make_parser()
    args = parser.parse_args()

    SEED = args.seed
    KEY = jrd.PRNGKey(SEED)
    KEY1, KEY2, KEY3, KEY4 = jrd.split(KEY, 4)
    POSTERIOR_REGEX = args.posterior_regex
    POSTERIOR_COLUMNS: list[str] = args.posterior_columns

    has_log_ref_prior = LOG_REF_PRIOR_NAME in POSTERIOR_COLUMNS
    if has_log_ref_prior:
        POSTERIOR_COLUMNS.remove(LOG_REF_PRIOR_NAME)

    N_pl = args.n_pl
    N_g = args.n_g

    has_spin = args.add_beta_spin or args.add_truncated_normal_spin
    has_tilt = args.add_tilt
    has_eccentricity = args.add_truncated_normal_eccentricity
    has_redshift = args.add_redshift

    prior_dict = read_json(args.prior_json)

    all_params: List[Tuple[str, int]] = [
        ("alpha_pl", N_pl),
        ("beta_pl", N_pl),
        ("mmax_pl", N_pl),
        ("mmin_pl", N_pl),
        ("delta_pl", N_pl),
        ("log_rate", N_pl + N_g),
        ("lamb_scale_g", N_g),
        ("lamb_scale_pl", N_pl),
        ("loc_g", N_g),
        ("scale_g", N_g),
        ("beta_g", N_g),
        ("mmin_g", N_g),
        ("mmax_g", N_g),
        ("delta_g", N_g),
    ]

    where_fns = []

    parameters = [PRIMARY_MASS_SOURCE]

    if args.raw:
        gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_powerlaw_distributions = create_smoothed_powerlaws_raw
        gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_gaussian_distributions = create_smoothed_gaussians_raw
        parameters.append(MASS_RATIO)
    else:
        parameters.append(SECONDARY_MASS_SOURCE)

    if has_spin:
        parameters.extend([PRIMARY_SPIN_MAGNITUDE, SECONDARY_SPIN_MAGNITUDE])
        if args.add_truncated_normal_spin:
            gwkokab.models.nsmoothedpowerlawmsmoothedgaussian._model.build_spin_distributions = create_truncated_normal_distributions
            all_params.extend(
                [
                    ("chi1_high_g", N_g),
                    ("chi1_high_pl", N_pl),
                    ("chi1_loc_g", N_g),
                    ("chi1_loc_pl", N_pl),
                    ("chi1_low_g", N_g),
                    ("chi1_low_pl", N_pl),
                    ("chi1_scale_g", N_g),
                    ("chi1_scale_pl", N_pl),
                    ("chi2_high_g", N_g),
                    ("chi2_high_pl", N_pl),
                    ("chi2_loc_g", N_g),
                    ("chi2_loc_pl", N_pl),
                    ("chi2_low_g", N_g),
                    ("chi2_low_pl", N_pl),
                    ("chi2_scale_g", N_g),
                    ("chi2_scale_pl", N_pl),
                ]
            )
        if args.add_beta_spin:
            all_params.extend(
                [
                    ("chi1_mean_g", N_g),
                    ("chi1_mean_pl", N_pl),
                    ("chi1_variance_g", N_g),
                    ("chi1_variance_pl", N_pl),
                    ("chi2_mean_g", N_g),
                    ("chi2_mean_pl", N_pl),
                    ("chi2_variance_g", N_g),
                    ("chi2_variance_pl", N_pl),
                ]
            )

            def mean_variance_check(**kwargs) -> Array:
                if N_pl > 0:
                    means_pl = jnp.stack(
                        [
                            kwargs[f"chi{i}_mean_pl_{j}"]
                            for j in range(N_pl)
                            for i in (1, 2)
                        ]
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
                        [
                            kwargs[f"chi{i}_mean_g_{j}"]
                            for j in range(N_g)
                            for i in (1, 2)
                        ]
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

                valid_var = variances <= means * (1 - means)
                valid_var = jnp.logical_and(valid_var, variances > 0.0)
                return jnp.all(valid_var)

            where_fns.append(mean_variance_check)

    if has_tilt:
        parameters.extend([COS_TILT_1, COS_TILT_2])
        all_params.extend(
            [
                ("cos_tilt_zeta_g", N_g),
                ("cos_tilt_zeta_pl", N_pl),
                ("cos_tilt1_scale_g", N_g),
                ("cos_tilt1_scale_pl", N_pl),
                ("cos_tilt2_scale_g", N_g),
                ("cos_tilt2_scale_pl", N_pl),
            ]
        )

        def tilt_scale_should_be_positive(**kwargs) -> Array:
            if N_pl > 0:
                scale_pl = jnp.stack(
                    [
                        kwargs[f"cos_tilt{i}_scale_pl_{j}"]
                        for j in range(N_pl)
                        for i in (1, 2)
                    ]
                )
            if N_g > 0:
                scale_g = jnp.stack(
                    [
                        kwargs[f"cos_tilt{i}_scale_g_{j}"]
                        for j in range(N_g)
                        for i in (1, 2)
                    ]
                )

            if N_pl > 0 and N_g > 0:
                scales = jnp.concatenate([scale_pl, scale_g])
            elif N_pl > 0:
                scales = scale_pl
            else:
                scales = scale_g

            return jnp.all(scales > 0.0)

        where_fns.append(tilt_scale_should_be_positive)

    if has_eccentricity:
        parameters.append(ECCENTRICITY)
        all_params.extend(
            [
                ("ecc_high_g", N_g),
                ("ecc_high_pl", N_pl),
                ("ecc_loc_g", N_g),
                ("ecc_loc_pl", N_pl),
                ("ecc_low_g", N_g),
                ("ecc_low_pl", N_pl),
                ("ecc_scale_g", N_g),
                ("ecc_scale_pl", N_pl),
            ]
        )
    if has_redshift:
        parameters.append(REDSHIFT)
        all_params.extend(
            [
                ("redshift_kappa_g", N_g),
                ("redshift_kappa_pl", N_pl),
                ("redshift_z_max_g", N_g),
                ("redshift_z_max_pl", N_pl),
            ]
        )

    error_if(
        set(POSTERIOR_COLUMNS) != set(map(lambda p: p.name, parameters)),
        msg="The parameters in the posterior data do not match the parameters in the model.",
    )

    extended_params = []
    for params in all_params:
        extended_params.extend(expand_arguments(*params))

    model_prior_param = get_processed_priors(extended_params, prior_dict)

    model = Bake(NSmoothedPowerlawMSmoothedGaussian)(
        N_pl=N_pl,
        N_g=N_g,
        use_spin=has_spin,
        use_tilt=has_tilt,
        use_eccentricity=has_eccentricity,
        use_redshift=has_redshift,
        **model_prior_param,
    )

    parameters_name = [param.name for param in parameters]
    nvt = vt_json_read_and_process(parameters_name, args.vt_json)

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(nvt, key=KEY4, **pmean_kwargs)  # type: ignore[arg-type]

    if has_log_ref_prior:
        POSTERIOR_COLUMNS.append(LOG_REF_PRIOR_NAME)

    data = get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS)
    if has_log_ref_prior:
        log_ref_priors = [d[..., -1] for d in data]
        data = [d[..., :-1] for d in data]
    else:
        logger.info("Reading reference priors")
        log_ref_priors = [np.zeros(d.shape[:-1]) for d in data]

    variables_index, priors, poisson_likelihood_fn = poisson_likelihood(
        parameters_name=parameters_name,
        dist_builder=model,
        data=data,
        log_ref_priors=log_ref_priors,
        ERate_fn=erate_estimator.__call__,
        n_buckets=args.n_buckets,
        threshold=args.threshold,
    )

    constants = model.constants

    constants["N_pl"] = N_pl
    constants["N_g"] = N_g
    constants["use_spin"] = int(has_spin)
    constants["use_tilt"] = int(has_tilt)
    constants["use_eccentricity"] = int(has_eccentricity)
    constants["use_redshift"] = int(has_redshift)

    with open("constants.json", "w") as f:
        json.dump(constants, f)

    with open("nf_samples_mapping.json", "w") as f:
        json.dump(variables_index, f)

    FLOWMC_HANDLER_KWARGS = read_json(args.flowMC_json)

    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["rng_key"] = KEY1
    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["key"] = KEY2

    N_CHAINS = FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_chains"]
    initial_position = priors.sample(KEY3, (N_CHAINS,))

    FLOWMC_HANDLER_KWARGS["nf_model_kwargs"]["n_features"] = initial_position.shape[1]
    FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["n_dim"] = initial_position.shape[1]

    FLOWMC_HANDLER_KWARGS["data_dump_kwargs"]["labels"] = list(model.variables.keys())

    FLOWMC_HANDLER_KWARGS = flowMC_default_parameters(**FLOWMC_HANDLER_KWARGS)

    if args.adam_optimizer:
        from flowMC.strategy.optimization import optimization_Adam

        adam_kwargs = read_json(args.adam_json)
        Adam_opt = optimization_Adam(**adam_kwargs)

        FLOWMC_HANDLER_KWARGS["sampler_kwargs"]["strategies"] = [Adam_opt, "default"]

    handler = flowMChandler(
        logpdf=poisson_likelihood_fn,
        initial_position=initial_position,
        **FLOWMC_HANDLER_KWARGS,
    )

    handler.run(
        debug_nans=args.debug_nans,
        profile_memory=args.profile_memory,
        check_leaks=args.check_leaks,
        file_prefix="n_spls_m_sgs",
    )
