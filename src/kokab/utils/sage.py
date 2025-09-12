# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import jax
import numpy as np
from jaxtyping import Array, ArrayLike
from loguru import logger
from numpyro._typing import DistributionLike
from numpyro.distributions import Distribution

from gwkokab.models.utils import JointDistribution
from gwkokab.poisson_mean import PoissonMean
from gwkokab.utils.tools import warn_if
from kokab.utils.common import get_posterior_data
from kokab.utils.literals import LOG_REF_PRIOR_NAME
from kokab.utils.poisson_mean_parser import read_pmean

from .guru import Guru
from .jenks import pad_and_stack


class Sage(Guru):
    def __init__(
        self,
        likelihood_fn: Callable[
            [
                Callable[..., DistributionLike],
                JointDistribution,
                Dict[str, DistributionLike],
                Dict[str, int],
                ArrayLike,
                PoissonMean,
                Optional[List[Callable[..., Array]]],
                Dict[str, Array],
            ],
            Callable,
        ],
        model: Union[Distribution, Callable[..., Distribution]],
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
        analysis_name: str = "",
    ) -> None:
        """
        Parameters
        ----------
        model : Union[Distribution, Callable[..., Distribution]]
            model to be used in the Sage class. It can be a Distribution or a callable
            that returns a Distribution.
        where_fns : Optional[List[Callable[..., Array]]]
            List of functions to apply to the data before passing it to the model.
        posterior_regex : str
            Regular expression to match posterior files.
        posterior_columns : List[str]
            List of columns to extract from the posterior files.
        seed : int
            Seed for the random number generator.
        prior_filename : str
            Path to the JSON file containing the prior distributions.
        selection_fn_filename : str
            Path to the JSON file containing the selection function.
        poisson_mean_filename : str
            Path to the JSON file containing the Poisson mean configuration.
        sampler_settings_filename : str
            Path to the JSON file containing the sampler settings.
        n_buckets : int
            Number of buckets to use for padding and stacking the data.
        threshold : float
            Threshold for padding and stacking the data.
        debug_nans : bool, optional
            If True, checks for NaNs in each computation. See details in the
            [documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug_nans.html#jax.debug_nans),
            by default False
        profile_memory : bool, optional
            If True, enables memory profiling, by default False
        check_leaks : bool, optional
            If True, checks for JAX Tracer leaks. See details in the
            [documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.checking_leaks.html#jax.checking_leaks),
            by default False
        analysis_name : str, optional
            Name of the analysis, by default ""
        """
        assert all(letter.isalpha() or letter == "_" for letter in analysis_name), (
            "Analysis name must contain only letters and underscores."
        )
        self.likelihood_fn = likelihood_fn
        self.n_buckets = n_buckets
        self.posterior_columns = posterior_columns
        self.posterior_regex = posterior_regex
        self.threshold = threshold
        self.where_fns = where_fns
        self.set_rng_key(seed=seed)

        super().__init__(
            analysis_name=analysis_name,
            check_leaks=check_leaks,
            debug_nans=debug_nans,
            model=model,
            poisson_mean_filename=poisson_mean_filename,
            prior_filename=prior_filename,
            profile_memory=profile_memory,
            sampler_settings_filename=sampler_settings_filename,
            selection_fn_filename=selection_fn_filename,
        )

    def read_data(
        self,
    ) -> Tuple[int, float, Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...]]:
        data = get_posterior_data(glob(self.posterior_regex), self.posterior_columns)
        if LOG_REF_PRIOR_NAME in self.posterior_columns:
            idx = self.posterior_columns.index(LOG_REF_PRIOR_NAME)
            self.posterior_columns.pop(idx)
            warn_if(
                True,
                msg="Please ensure that reference prior is in the last column of the posteriors.",
            )
            log_ref_priors = [d[..., idx] for d in data]
            data = [np.delete(d, idx, axis=-1) for d in data]
        else:
            log_ref_priors = [np.zeros(d.shape[:-1]) for d in data]
        assert len(data) == len(log_ref_priors), (
            "Data and log reference priors must have the same length"
        )
        _data_group, _log_ref_priors_group, _masks_group = pad_and_stack(
            data, log_ref_priors, n_buckets=self.n_buckets, threshold=self.threshold
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

        return n_events, log_constants, data_group, log_ref_priors_group, masks_group

    def run(self) -> None:
        constants, dist_fn, priors, variables, variables_index = self.bake_model()

        n_events, log_constants, data_group, log_ref_priors_group, masks_group = (
            self.read_data()
        )

        ERate_obj = read_pmean(
            self.rng_key,
            self.parameters,
            self.poisson_mean_filename,
            self.selection_fn_filename,
        )

        log_constants += n_events * np.log(ERate_obj.time_scale)  # type: ignore

        logpdf = self.likelihood_fn(
            dist_fn=dist_fn,
            priors=priors,
            variables=variables,
            variables_index=variables_index,
            log_constants=log_constants,
            ERate_obj=ERate_obj,
            where_fns=self.where_fns,
            constants=constants,  # type: ignore
        )

        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "data_group": data_group,
                "log_ref_priors_group": log_ref_priors_group,
                "masks_group": masks_group,
            },
            labels=sorted(variables.keys()),
        )


def sage_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Sage script.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """

    sage_group = parser.add_argument_group("Sage Options")
    sage_group.add_argument(
        "--posterior-regex",
        help="Regex for the posterior samples.",
        type=str,
        required=True,
    )
    sage_group.add_argument(
        "--posterior-columns",
        help="Columns of the posterior samples.",
        nargs="+",
        type=str,
        required=True,
    )

    optm_group = parser.add_argument_group("Optimization Options")
    optm_group.add_argument(
        "--n-buckets",
        help="Number of buckets for the data arrays to be split into. "
        "This is useful for large datasets to avoid memory issues. "
        "See https://github.com/gwkokab/gwkokab/issues/568 for more details.",
        type=int,
        default=None,
    )
    optm_group.add_argument(
        "--threshold",
        help="Threshold to determine best number of buckets, if the number of buckets "
        "is not specified. It should be between 0 and 100.",
        type=float,
        default=3.0,
    )

    return parser
