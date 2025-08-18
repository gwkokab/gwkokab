# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
from typing import Tuple

import arviz as az
import jax
import numpy as np
from jax import random as jrd
from jaxtyping import Array
from loguru import logger
from numpyro.infer import MCMC, NUTS

from gwkokab.inference import Bake, numpyro_poisson_likelihood
from gwkokab.inference.jenks import pad_and_stack
from gwkokab.parameters import ECCENTRICITY, PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE
from gwkokab.poisson_mean import PoissonMean
from gwkokab.utils.tools import error_if
from kokab.ecc_matters.common import EccentricityMattersModel
from kokab.utils import nsage_parser, poisson_mean_parser
from kokab.utils.common import (
    get_posterior_data,
    get_processed_priors,
    LOG_REF_PRIOR_NAME,
    read_json,
    vt_json_read_and_process,
    write_json,
)


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = nsage_parser.get_parser(parser)

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
    KEY1, KEY2 = jrd.split(KEY)
    POSTERIOR_REGEX = args.posterior_regex
    POSTERIOR_COLUMNS: list[str] = args.posterior_columns

    has_log_ref_prior = LOG_REF_PRIOR_NAME in POSTERIOR_COLUMNS
    if has_log_ref_prior:
        log_ref_prior_idx = POSTERIOR_COLUMNS.index(LOG_REF_PRIOR_NAME)
        POSTERIOR_COLUMNS.remove(LOG_REF_PRIOR_NAME)

    prior_dict = read_json(args.prior_json)

    model_prior_param = get_processed_priors(
        ["log_rate", "alpha_m", "mmin", "mmax", "loc", "scale", "low", "high"],
        prior_dict,
    )

    model = Bake(EccentricityMattersModel)(**model_prior_param)

    parameters = [PRIMARY_MASS_SOURCE, SECONDARY_MASS_SOURCE, ECCENTRICITY]
    error_if(
        set(POSTERIOR_COLUMNS) != set(map(lambda p: p.name, parameters)),
        msg="The parameters in the posterior data do not match the parameters in the model.",
    )

    parameters_name = [param.name for param in parameters]
    nvt = vt_json_read_and_process(parameters_name, args.vt_json)

    pmean_kwargs = poisson_mean_parser.poisson_mean_parser(args.pmean_json)
    erate_estimator = PoissonMean(nvt, key=KEY1, **pmean_kwargs)  # type: ignore[arg-type]

    if has_log_ref_prior:
        POSTERIOR_COLUMNS.append(LOG_REF_PRIOR_NAME)

    data = get_posterior_data(glob(POSTERIOR_REGEX), POSTERIOR_COLUMNS)
    if has_log_ref_prior:
        log_ref_priors = [d[..., log_ref_prior_idx] for d in data]
        data = [np.delete(d, log_ref_prior_idx, axis=-1) for d in data]
    else:
        log_ref_priors = [np.zeros(d.shape[:-1]) for d in data]

    _data_group, _log_ref_priors_group, _masks_group = pad_and_stack(
        data, log_ref_priors, n_buckets=args.n_buckets, threshold=args.threshold
    )

    _data_group = tuple(_data_group)
    _log_ref_priors_group = tuple(_log_ref_priors_group)
    _masks_group = tuple(_masks_group)

    data_group: Tuple[Array] = jax.block_until_ready(
        jax.device_put(_data_group, may_alias=True)
    )
    log_ref_priors_group: Tuple[Array] = jax.block_until_ready(
        jax.device_put(_log_ref_priors_group, may_alias=True)
    )
    masks_group: Tuple[Array] = jax.block_until_ready(
        jax.device_put(_masks_group, may_alias=True)
    )

    logger.debug(
        "data_group.shape: {shape}",
        shape=", ".join([str(d.shape) for d in data_group]),
    )
    logger.debug(
        "log_ref_priors_group.shape: {shape}",
        shape=", ".join([str(d.shape) for d in log_ref_priors_group]),
    )
    logger.debug(
        "masks_group.shape: {shape}",
        shape=", ".join([str(d.shape) for d in masks_group]),
    )

    n_events = len(data)
    sum_log_size = sum([np.log(d.shape[0]) for d in data])
    log_constants = -sum_log_size  # -Σ log(M_i)
    log_constants += n_events * np.log(erate_estimator.time_scale)

    likelihood_fn, variables_index = numpyro_poisson_likelihood(
        dist_builder=model,
        log_constants=log_constants,
        ERate_obj=erate_estimator,
    )

    constants = model.constants

    with open("constants.json", "w") as f:
        json.dump(constants, f)

    with open("nf_samples_mapping.json", "w") as f:
        json.dump(variables_index, f)

    kernel = NUTS(likelihood_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    mcmc.run(
        KEY2,
        data_group=data_group,
        log_ref_priors_group=log_ref_priors_group,
        masks_group=masks_group,
    )

    posterior_samples = mcmc.get_samples(group_by_chain=True)
    write_json("posterior_samples.json", posterior_samples)

    mcmc_data = az.from_numpyro(mcmc)

    summary = az.summary(mcmc_data)
    summary.to_csv("posterior_summary.csv")

    fig = az.plot_trace(mcmc_data, compact=True, figsize=(15, 25))
    fig.savefig("posterior_trace.png")
