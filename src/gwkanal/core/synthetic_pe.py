# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import Optional, TypeAlias

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from numpyro.distributions.distribution import enable_validation

from gwkanal.core.utils import from_structured, PRNGKeyMixin, to_structured
from gwkanal.utils.common import read_json
from gwkanal.utils.logger import log_info
from gwkanal.utils.regex import match_all
from gwkokab.parameters import default_relation_mesh, Parameters
from gwkokab.utils.tools import warn_if


ErrorFunctionRegistryType: TypeAlias = dict[
    str | Parameters | tuple[str, ...] | tuple[Parameters, ...],
    tuple[tuple[str, ...], Callable[..., np.ndarray]],
]
"""Type alias for the error function registry mapping."""


class FakeDiscretePEBase(PRNGKeyMixin):
    waveform_name = "GWKokabSyntheticDiscretePE"
    root_dir = Path("data")

    def __init__(
        self,
        filename: str,
        error_params_filename: str,
        size: int,
        derive_parameters: bool = False,
    ) -> None:
        self.filename = filename
        self.error_params_filename = error_params_filename
        self.size = size
        self.derive_parameters = derive_parameters

    @property
    def error_function_registry(self) -> ErrorFunctionRegistryType:
        r"""Returns a dictionary mapping parameters to their corresponding error
        functions.

        The keys of the dictionary can be either strings representing parameter names,
        tuples of strings representing multiple parameter names, or instances of the
        Parameters enum (P) or tuples of such instances. The values are callable functions
        that define how to apply errors to the corresponding parameters.

        Returns
        -------
        ErrorFunctionRegistryType
            A dictionary mapping parameters to their error functions.
        """
        msg = "Subclasses must implement error_function_registry."
        logger.error(msg)
        raise NotImplementedError(msg)

    def _load_injection_data(self) -> tuple[list[str], dict[str, np.ndarray]]:
        """Extracts parameters and injection data from the source HDF5."""
        with h5py.File(self.filename, "r") as f:
            parameters = f.attrs["parameters"].astype(np.str_).tolist()
            injection_data = {p: f["injection_data"][p] for p in parameters}
        return parameters, injection_data

    def _apply_error_model(
        self,
        parameters: list[str],
        injection_values: dict[str, np.ndarray],
        error_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Applies the registry functions to generate estimates."""
        estimate = {}
        for key, (err_keys, func) in self.error_function_registry.items():
            # Support both single strings and tuples of parameters
            keys = (key,) if isinstance(key, (str, Parameters)) else key

            if any(k not in parameters for k in keys):
                continue

            err_vals = match_all(err_keys, error_params)  # type: ignore[arg-type]

            val = func(**injection_values, **err_vals)

            if isinstance(key, tuple):
                for idx, k in enumerate(key):
                    estimate[k] = val[:, idx]
            else:
                estimate[key] = val
        return estimate

    def _save_event(
        self,
        index: int,
        params: list[str],
        posterior: dict[str, np.ndarray],
        injection: dict[str, np.ndarray],
        width: int,
    ):
        """Handles the HDF5 boilerplate for a single event."""
        tag = str(index).zfill(width)
        event_path = self.root_dir / f"event_{tag}.hdf5"

        # Stack and clean NaNs
        data_stack = np.stack([posterior[p] for p in params], axis=-1)
        data_stack = data_stack[~np.any(np.isnan(data_stack), axis=-1)]

        posterior_samples = to_structured(data_stack, params)
        injection_data = to_structured(
            np.array([[injection[p] for p in params]]), params
        )

        with h5py.File(event_path, "w") as ef:
            ef.attrs["parameters"] = np.array(params, dtype="S")
            group = ef.create_group(self.waveform_name)
            group.create_dataset("approximant", data=self.waveform_name)
            group.create_dataset("injection_data", data=injection_data)
            group.create_dataset("posterior_samples", data=posterior_samples)

    def generate_parameter_estimates(self):
        """Generates parameter estimates based on the defined error functions.

        This method should be implemented in subclasses to generate parameter estimates
        using the error functions defined in the `error_function_registry` property.
        """
        parameters, injection_data = self._load_injection_data()
        n_injections = len(next(iter(injection_data.values())))

        error_params = read_json(self.error_params_filename)

        self.root_dir.mkdir(parents=True, exist_ok=True)
        width = len(str(n_injections - 1))

        if self.derive_parameters:  # Load once
            mesh = default_relation_mesh()

        for i in range(n_injections):
            inj_vals = {p: injection_data[p][i] for p in parameters}
            post_est = self._apply_error_model(parameters, inj_vals, error_params)

            active_params = parameters
            if self.derive_parameters:
                post_est = mesh.resolve(initial_state=post_est)
                inj_vals = mesh.resolve(initial_state=inj_vals)
                active_params = sorted(post_est.keys())

            self._save_event(i, active_params, post_est, inj_vals, width)


def fake_discrete_pe_parser() -> ArgumentParser:
    """Create the command line argument parser.

    This function creates the command line argument parser and returns it.

    Returns
    -------
    ArgumentParser
        the command line argument parser
    """
    # Global enable validation for all distributions
    enable_validation()

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Generate fake discrete parameter estimation samples.",
        epilog="This tool generates fake parameter estimation samples based on "
        "injection data and a specified error model.",
    )
    parser.add_argument(
        "--filename",
        help="Path to the output HDF5 file to store the generated injections.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--error-params",
        help="Path to the JSON file containing the error model parameters",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--size",
        help="Number of posterior samples to generate per event.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--derive-parameters",
        help="Compute the derivable parameters.",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility.",
        default=37,
        type=int,
    )

    return parser


class SyntheticAnalyticalPE(PRNGKeyMixin):
    waveform_name = "GWKokabSyntheticAnalyticalPE"

    def __init__(
        self,
        filename: str,
        discrete_waveform: str,
        coords: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.filename = filename
        self.discrete_waveform = discrete_waveform
        self.coords = coords

    def generate_parameter_estimates(self):
        with h5py.File(self.filename, "r") as ef:
            group = ef[self.discrete_waveform]
            posterior_samples = group["posterior_samples"][:]

        posterior_samples, parameters = from_structured(posterior_samples)

        df = pd.DataFrame(posterior_samples, columns=parameters)
        if self.coords is not None:
            df = df[list(self.coords)]

        cov = df.cov().to_numpy()
        mean = df.mean().to_numpy()
        cor = df.corr().to_numpy()
        std = np.sqrt(np.diag(cov))
        limits = np.array((df.min().to_numpy(), df.max().to_numpy())).T

        with h5py.File(self.filename, "a") as ef:
            if self.waveform_name in ef:
                warn_if(
                    True,
                    msg=f"Group '{self.waveform_name}' already exists in {self.filename}. "
                    "Overwriting existing data.",
                )
                del ef[self.waveform_name]

            group = ef.create_group(self.waveform_name)
            group.create_dataset("approximant", data=self.waveform_name)
            group.attrs["discrete_waveform"] = self.discrete_waveform
            group.attrs["coords"] = self.coords
            group.create_dataset("mu", data=mean)
            group.create_dataset("std", data=std)
            group.create_dataset("cov", data=cov)
            group.create_dataset("cor", data=cor)
            group.create_dataset("limits", data=limits)


def synthetic_analytical_pe_main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Generate synthetic analytical parameter estimation samples.",
        epilog="This tool generates synthetic parameter estimation samples based on "
        "injection data and a specified error model.",
    )

    parser.add_argument(
        "filename",
        help="Path to the output HDF5 file to store the generated injections.",
        type=str,
    )
    parser.add_argument(
        "--discrete-waveform",
        help="Name of the discrete waveform used to generate the analytical PE.",
        type=str,
        default=FakeDiscretePEBase.waveform_name,
    )
    parser.add_argument(
        "--coords",
        help="Comma-separated list of coords to include in the output. "
        "If not provided, all coords will be included.",
        type=lambda s: tuple(s.split(",")),
        default=None,
    )
    parser.add_argument(
        "--seed", help="Random seed for reproducibility.", default=37, type=int
    )

    args = parser.parse_args()

    log_info(start=True)

    SyntheticAnalyticalPE.set_rng_key(seed=args.seed)

    pe = SyntheticAnalyticalPE(
        filename=args.filename,
        discrete_waveform=args.discrete_waveform,
        coords=args.coords,
    )

    pe.generate_parameter_estimates()
