# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from glob import glob
from typing import Dict, List, Tuple, Union

import jax
import numpy as np
from jaxtyping import Array
from loguru import logger
from numpyro.distributions import Distribution
from numpyro.infer import MCMC, NUTS

from gwkokab.inference import Bake, numpyro_poisson_likelihood
from gwkokab.inference.jenks import pad_and_stack
from gwkokab.utils.tools import warn_if
from kokab.utils.common import (
    get_posterior_data,
    get_processed_priors,
    LOG_REF_PRIOR_NAME,
    read_json,
    save_inference_data,
    write_json,
)
from kokab.utils.poisson_mean_parser import read_pmean

from .guru import Guru


class NSage(Guru):
    def __init__(
        self,
        model: Union[Distribution, Callable[..., Distribution]],
        posterior_regex: str,
        posterior_columns: List[str],
        seed: int,
        prior_filename: str,
        selection_fn_filename: str,
        poisson_mean_filename: str,
        sampler_settings_filename: str,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
        analysis_name: str = "",
    ) -> None:
        """

        Parameters
        ----------
        model : Union[Distribution, Callable[..., Distribution]]
            model to be used in the NSage class. It can be a Distribution or a callable
            that returns a Distribution.
        posterior_regex : str
            path to the HDF5 file containing the data.
        posterior_columns : List[str]
            list of columns to extract from the posterior samples.
        seed : int
            seed for the random number generator.
        prior_filename : str
            path to the JSON file containing the prior distributions.
        selection_fn_filename : str
            path to the JSON file containing the selection function.
        poisson_mean_filename : str
            path to the JSON file containing the Poisson mean configuration.
        sampler_settings_filename : str
            path to the JSON file containing the sampler settings.
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
            "Analysis name must be alphabetic characters only."
        )
        self.model = model
        self.analysis_name = "nsage_" + (analysis_name or model.__name__)
        self.posterior_regex = posterior_regex
        self.prior_filename = prior_filename
        self.posterior_columns = posterior_columns
        self.selection_fn_filename = selection_fn_filename
        self.poisson_mean_filename = poisson_mean_filename
        self.sampler_settings_filename = sampler_settings_filename
        self.debug_nans = debug_nans
        self.profile_memory = profile_memory
        self.check_leaks = check_leaks
        self.set_rng_key(seed=seed)

    @property
    def baked_model(self) -> Bake:
        """Returns a Bake object for the model.

        Returns
        -------
        Bake
            A Bake object that wraps the model with the constants and prior parameters.
        """
        prior_dict = read_json(self.prior_filename)
        model_prior_param = get_processed_priors(self.model_parameters, prior_dict)
        return Bake(self.model)(**self.constants, **model_prior_param)

    @property
    def parameters(self) -> List[str]:
        """Returns the parameters (intrinsic + extrinsic).

        Returns
        -------
        List[str]
            List of parameters.

        Raises
        ------
        NotImplementedError
            If the NSage class is used directly, this method raises a NotImplementedError.
            It is expected that subclasses of NSage will implement this method.
        """
        msg = (
            "The NSage class should not be used directly. Please use a subclass that "
            "implements the parameters property."
        )
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def model_parameters(self) -> List[str]:
        """Returns the model parameters.

        Returns
        -------
        List[str]
            List of model parameters.

        Raises
        ------
        NotImplementedError
            If the Monk class is used directly, this method raises a NotImplementedError.
            It is expected that subclasses of Monk will implement this method.
        """
        msg = (
            "The Monk class should not be used directly. Please use a subclass that "
            "implements the model_parameters property."
        )
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def constants(self) -> Dict[str, Union[int, float, bool]]:
        """Returns the constants used in the model.

        Returns
        -------
        Dict[str, Union[int, float, bool]]
            A dictionary containing the constants used in the model.
        """
        return {}

    def read_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        return data, log_ref_priors

    def run(self, n_buckets: int, threshold: float) -> None:
        """Runs the NSage analysis."""
        logger.debug("Baking the model")
        constants, variables, duplicates, dist_fn = self.baked_model.get_dist()  # type: ignore
        variables_index: dict[str, int] = {
            key: i for i, key in enumerate(sorted(variables.keys()))
        }
        for key, value in duplicates.items():
            variables_index[key] = variables_index[value]

        # TODO(Qazalbash): refactor logic for grouping variables and logging them into a
        # function and use it for both Sage and Monk.
        group_variables: dict[int, list[str]] = {}
        for key, value in variables_index.items():  # type: ignore
            group_variables[value] = group_variables.get(value, []) + [key]  # type: ignore

        logger.debug(
            "Number of recovering variables: {num_vars}", num_vars=len(group_variables)
        )

        for key, value in constants.items():  # type: ignore
            logger.debug(
                "Constant variable: {name} = {variable}", name=key, variable=value
            )

        for value in group_variables.values():  # type: ignore
            logger.debug("Recovering variable: {variable}", variable=", ".join(value))

        write_json("constants.json", constants)
        write_json("nf_samples_mapping.json", variables_index)

        erate_estimator = read_pmean(
            self.rng_key,
            self.parameters,
            self.poisson_mean_filename,
            self.selection_fn_filename,
        )

        data, log_ref_priors = self.read_data()

        _data_group, _log_ref_priors_group, _masks_group = pad_and_stack(
            data, log_ref_priors, n_buckets=n_buckets, threshold=threshold
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

        likelihood_fn = numpyro_poisson_likelihood(
            dist_fn=dist_fn,
            variables=variables,
            variables_index=variables_index,
            log_constants=log_constants,
            ERate_obj=erate_estimator,
        )

        sampler_config = read_json(self.sampler_settings_filename)

        def f() -> MCMC:
            kernel = NUTS(likelihood_fn, **sampler_config["kernel"])
            mcmc = MCMC(kernel, **sampler_config["mcmc"])
            mcmc.run(
                self.rng_key,
                data_group=data_group,
                log_ref_priors_group=log_ref_priors_group,
                masks_group=masks_group,
            )
            return mcmc

        if self.debug_nans:
            with jax.debug_nans(True):
                mcmc = f()
        elif self.profile_memory:
            mcmc = f()

            import datetime

            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{self.analysis_name}_memory_{time}.prof"
            jax.profiler.save_device_memory_profile(filename)
        elif self.check_leaks:
            with jax.checking_leaks():
                mcmc = f()
        else:
            mcmc = f()

        save_inference_data(mcmc)


def get_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the NSage
    script.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """

    nsage_group = parser.add_argument_group("Sage Options")

    nsage_group.add_argument(
        "--posterior-regex",
        help="Regex for the posterior samples.",
        type=str,
        required=True,
    )
    nsage_group.add_argument(
        "--posterior-columns",
        help="Columns of the posterior samples.",
        nargs="+",
        type=str,
        required=True,
    )
    nsage_group.add_argument(
        "--seed",
        help="Seed for the random number generator.",
        default=37,
        type=int,
    )

    vt_group = parser.add_argument_group("VT Options")

    vt_group.add_argument(
        "--vt-json",
        help="Path to the JSON file containing the VT options.",
        type=str,
        default="vt.json",
    )

    pmean_group = parser.add_argument_group("Poisson Mean Options")

    pmean_group.add_argument(
        "--pmean-json",
        help="Path to the JSON file containing the Poisson mean options.",
        type=str,
        default="pmean.json",
    )

    sampler_group = parser.add_argument_group("Numpyro Options")
    sampler_group.add_argument(
        "--sampler-config",
        help="Path to the JSON file containing the sampler configuration.",
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

    prior_group = parser.add_argument_group("Prior Options")
    prior_group.add_argument(
        "--prior-json",
        type=str,
        help="Path to a JSON file containing the prior distributions.",
        default="prior.json",
    )

    debug_group = parser.add_argument_group("Debug Options")
    debug_group.add_argument(
        "--debug-nans",
        help="Checks for NaNs in each computation. See details in the documentation: "
        "https://jax.readthedocs.io/en/latest/_autosummary/jax.debug_nans.html#jax.debug_nans.",
        action="store_true",
    )
    debug_group.add_argument(
        "--profile-memory",
        help="Enable memory profiling.",
        action="store_true",
    )
    debug_group.add_argument(
        "--check-leaks",
        help="Check for JAX Tracer leaks. See details in the documentation: "
        "https://jax.readthedocs.io/en/latest/_autosummary/jax.checking_leaks.html#jax.checking_leaks.",
        action="store_true",
    )

    return parser
