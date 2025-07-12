# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentParser
from collections.abc import Callable
from typing import Dict, List, Tuple, Union

import h5py
from jaxtyping import Array
from loguru import logger
from numpyro.distributions import Distribution
from numpyro.distributions.distribution import enable_validation

from gwkokab.inference import analytical_likelihood, Bake
from gwkokab.models.utils import JointDistribution
from gwkokab.utils.tools import error_if
from kokab.utils.common import (
    flowMC_default_parameters,
    get_processed_priors,
    read_json,
    write_json,
)
from kokab.utils.flowMC_helper import flowMChandler
from kokab.utils.poisson_mean_parser import read_pmean

from .guru import Guru


def _read_mean_covariances(filename: str) -> Tuple[List[Array], List[Array]]:
    """Reads means and covariances from a file.

    Args:
        filename (str): The path to the file containing means and covariances.

    Returns:
        Tuple[List[Array], List[Array]]: A tuple containing two lists:
            - list_of_means: List of mean arrays.
            - list_of_covariances: List of covariance arrays.
    """
    assert filename.endswith(".hdf5") or filename.endswith(".h5"), (
        "The filename must end with '.hdf5' or '.h5'."
    )

    logger.info("Reading means and covariances from {filename}", filename=filename)

    list_of_means = []
    list_of_covariances = []

    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if not key.startswith("event_"):
                continue

            group = f[key]
            error_if(
                "mean" not in group,
                msg=f"Key 'mean' not found in group {key} of file {filename}.",
            )
            error_if(
                "cov" not in group,
                msg=f"Key 'cov' not found in group {key} of file {filename}.",
            )

            mean = group["mean"][:]
            cov = group["cov"][:]

            n_dim = mean.shape[0]
            error_if(
                cov.shape != (n_dim, n_dim),
                msg=f"Covariance shape {cov.shape} does not match mean shape "
                f"{mean.shape} in group {key} of file {filename}.",
            )

            list_of_means.append(mean)
            list_of_covariances.append(cov)

    return list_of_means, list_of_covariances


class Monk(Guru):
    def __init__(
        self,
        model: Union[Distribution, Callable[..., Distribution]],
        data_filename: str,
        seed: int,
        prior_filename: str,
        selection_fn_filename: str,
        poisson_mean_filename: str,
        flowMC_settings_filename: str,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
        analysis_name: str = "",
    ) -> None:
        """

        Parameters
        ----------
        model : Union[Distribution, Callable[..., Distribution]]
            model to be used in the Monk class. It can be a Distribution or a callable
            that returns a Distribution.
        data_filename : str
            path to the HDF5 file containing the data.
        seed : int
            seed for the random number generator.
        prior_filename : str
            path to the JSON file containing the prior distributions.
        selection_fn_filename : str
            path to the JSON file containing the selection function.
        poisson_mean_filename : str
            path to the JSON file containing the Poisson mean configuration.
        flowMC_settings_filename : str
            path to the JSON file containing the flowMC settings.
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
        self.analysis_name = analysis_name or model.__name__
        self.data_filename = data_filename
        self.prior_filename = prior_filename
        self.selection_fn_filename = selection_fn_filename
        self.poisson_mean_filename = poisson_mean_filename
        self.flowMC_settings_filename = flowMC_settings_filename
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
            If the Monk class is used directly, this method raises a NotImplementedError.
            It is expected that subclasses of Monk will implement this method.
        """
        msg = (
            "The Monk class should not be used directly. Please use a subclass that "
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

    def run(self) -> None:
        """Runs the Monk analysis."""
        parameters = self.parameters
        if "redshift" in parameters:
            redshift_index = parameters.index("redshift")
        else:
            redshift_index = None

        logger.debug("Baking the model")
        print(self.baked_model.get_dummy())
        constants, variables, duplicates, dist_fn = self.baked_model.get_dist()  # type: ignore
        variables_index: dict[str, int] = {
            key: i for i, key in enumerate(variables.keys())
        }
        for key, value in duplicates.items():
            variables_index[key] = variables_index[value]

        priors = JointDistribution(*variables.values(), validate_args=True)

        write_json("constants.json", constants)
        write_json("nf_samples_mapping.json.json", variables_index)

        flowmc_handler_kwargs = read_json(self.flowMC_settings_filename)

        flowmc_handler_kwargs["sampler_kwargs"]["rng_key"] = self.rng_key
        flowmc_handler_kwargs["nf_model_kwargs"]["key"] = self.rng_key

        n_chains = flowmc_handler_kwargs["sampler_kwargs"]["n_chains"]
        initial_position = priors.sample(self.rng_key, (n_chains,))

        flowmc_handler_kwargs["nf_model_kwargs"]["n_features"] = initial_position.shape[
            1
        ]
        flowmc_handler_kwargs["sampler_kwargs"]["n_dim"] = initial_position.shape[1]

        flowmc_handler_kwargs["data_dump_kwargs"]["labels"] = list(variables.keys())

        flowmc_handler_kwargs = flowMC_default_parameters(**flowmc_handler_kwargs)

        list_of_means, list_of_covariances = _read_mean_covariances(self.data_filename)

        ERate_fn = read_pmean(
            self.rng_key,
            self.parameters,
            self.poisson_mean_filename,
            self.selection_fn_filename,
        )

        logpdf = analytical_likelihood(
            dist_fn,
            priors,
            variables_index,
            ERate_fn,
            redshift_index,
            list_of_means,
            list_of_covariances,
            self.rng_key,
            N_samples=10_000,
        )

        handler = flowMChandler(
            logpdf=logpdf,
            initial_position=initial_position,
            **flowmc_handler_kwargs,
        )

        handler.run(
            debug_nans=self.debug_nans,
            profile_memory=self.profile_memory,
            check_leaks=self.check_leaks,
            file_prefix=self.analysis_name,
        )


def get_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the Monk script.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """

    # Global enable validation for all distributions
    enable_validation()

    monk_group = parser.add_argument_group("Monk Options")

    monk_group.add_argument(
        "--data-filename",
        help="Path to the HDF5 file containing the data.",
        type=str,
        required=True,
    )
    monk_group.add_argument(
        "--seed",
        help="Seed for the random number generator.",
        default=37,
        type=int,
    )

    pmean_group = parser.add_argument_group("Poisson Mean Options")
    pmean_group.add_argument(
        "--vt-json",
        help="Path to the JSON file containing the VT options.",
        type=str,
        default="vt.json",
    )
    pmean_group.add_argument(
        "--pmean-json",
        help="Path to the JSON file containing the Poisson mean options.",
        type=str,
        default="pmean.json",
    )

    flowMC_group = parser.add_argument_group("flowMC Options")
    flowMC_group.add_argument(
        "--flowMC-json",
        help="Path to a JSON file containing the flowMC options. It should contains"
        "keys: local_sampler_kwargs, nf_model_kwargs, sampler_kwargs, data_dump_kwargs,"
        " and their respective values.",
        default="flowMC.json",
        type=str,
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
