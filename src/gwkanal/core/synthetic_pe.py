# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import pprint
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable
from pathlib import Path
from typing import Optional, TypeAlias

import h5py
import numpy as np
import tqdm
from jax import random as jrd
from loguru import logger
from numpyro.distributions.distribution import enable_validation

from gwkanal.core.utils import from_structured, PRNGKeyMixin, to_structured
from gwkanal.utils.common import read_json
from gwkanal.utils.logger import log_info
from gwkanal.utils.regex import match_all
from gwkokab.errors import banana_error, mock_spin_error, truncated_normal_error
from gwkokab.parameters import default_relation_mesh, Parameters, Parameters as P
from gwkokab.utils.exceptions import LoggedUserWarning, LoggedValueError


ErrorFunctionRegistryType: TypeAlias = dict[
    str | Parameters | tuple[str, ...] | tuple[Parameters, ...],
    tuple[tuple[str, ...], Callable[..., np.ndarray]],
]
"""Type alias for the error function registry mapping."""


class SyntheticDiscretePE(PRNGKeyMixin):
    waveform_name = "GWKokabSyntheticDiscretePE"
    root_dir = Path("data")

    def __init__(
        self,
        filename: str,
        error_params_filename: str,
        size: int,
        derive_parameters: bool = False,
        coords: Optional[list[str]] = None,
        is_delta_error: bool = False,
    ) -> None:
        self.filename = filename
        self.error_params_filename = error_params_filename
        self.size = size
        self.derive_parameters = derive_parameters
        self.coords = coords
        self.is_delta_error = is_delta_error

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

        def banana_error_fn(scale_Mc, scale_eta, estimates, rho, **kwargs):
            Mc = kwargs[P.CHIRP_MASS]
            eta = kwargs[P.SYMMETRIC_MASS_RATIO]
            return banana_error(
                Mc,
                eta,
                self.size,
                self.rng_key,
                scale_Mc=scale_Mc,
                scale_eta=scale_eta,
                estimates=estimates,
                rho=rho,
            )

        def generic_truncated_normal_error_fn(
            parameter: P,
            *,
            low: Optional[float] = None,
            high: Optional[float] = None,
        ) -> tuple[tuple[str, ...], Callable]:
            def error_fn(
                *, estimates, rho, default_low=low, default_high=high, **kwargs
            ):
                x = kwargs[parameter]
                scale = kwargs[parameter + "_scale"]
                low = kwargs.get(parameter + "_low", default_low)
                high = kwargs.get(parameter + "_high", default_high)

                return truncated_normal_error(
                    x=x,
                    size=self.size,
                    key=self.rng_key,
                    scale=scale,
                    low=low,
                    high=high,
                    estimates=estimates,
                    rho=rho,
                )

            error_parameters: tuple[str, ...] = (
                parameter + "_scale",
                parameter + "_low",
                parameter + "_high",
            )

            return error_parameters, error_fn

        def mock_spin_error_fn(scale_chi_eff, estimates, rho, **kwargs):
            chi_eff = kwargs[P.EFFECTIVE_SPIN]
            eta = kwargs[P.SYMMETRIC_MASS_RATIO]
            return mock_spin_error(
                chi_eff,
                eta,
                self.size,
                self.rng_key,
                scale_chi_eff=scale_chi_eff,
                estimates=estimates,
                rho=rho,
            )

        registry: ErrorFunctionRegistryType = {
            (P.CHIRP_MASS, P.SYMMETRIC_MASS_RATIO): (
                ("scale_Mc", "scale_eta"),
                banana_error_fn,
            ),
            P.EFFECTIVE_SPIN: (("scale_chi_eff",), mock_spin_error_fn),
        }

        for param, default_low, default_high in [
            (P.PRIMARY_SPIN_MAGNITUDE, 0.0, 1.0),
            (P.SECONDARY_SPIN_MAGNITUDE, 0.0, 1.0),
            (P.PRIMARY_SPIN_X, -1.0, 1.0),
            (P.SECONDARY_SPIN_X, -1.0, 1.0),
            (P.PRIMARY_SPIN_Y, -1.0, 1.0),
            (P.SECONDARY_SPIN_Y, -1.0, 1.0),
            (P.PRIMARY_SPIN_Z, -1.0, 1.0),
            (P.SECONDARY_SPIN_Z, -1.0, 1.0),
            (P.PRECESSING_SPIN, None, None),
            (P.COS_TILT_1, -1.0, 1.0),
            (P.COS_TILT_2, -1.0, 1.0),
            (P.ECCENTRICITY, 0.0, 1.0),
            (P.REDSHIFT, 1e-3, None),
            (P.SIN_DECLINATION, -1.0, 1.0),
            (P.COS_IOTA, -1.0, 1.0),
            (P.PHI_1, 0.0, 2 * np.pi),
            (P.PHI_2, 0.0, 2 * np.pi),
            (P.PHI_ORB, 0.0, 2 * np.pi),
            (P.MEAN_ANOMALY, 0.0, 2 * np.pi),
        ]:
            registry[param] = generic_truncated_normal_error_fn(
                param, low=default_low, high=default_high
            )

        return registry

    def _load_events(self) -> tuple[list[str], dict[str, np.ndarray]]:
        """Extracts parameters and injection data from the source HDF5."""
        with h5py.File(self.filename, "r") as f:
            parameters = f.attrs["parameters"].astype(np.str_).tolist()
            events = {p: f["events"][p] for p in parameters}
        return parameters, events

    def _parse_delta_thresholds(
        self,
        coords: list[str],
        error_params: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Parses delta error thresholds from the error parameters.

        This method checks for both coordinate-specific thresholds (e.g., 'mass_delta_threshold')
        and a default threshold ('delta_threshold'). If a coordinate-specific threshold is not provided,
        it falls back to the default threshold. If neither is provided, it raises an error.

        Parameters
        ----------
        coords : list[str]
            List of coordinates for which to parse delta thresholds.
        error_params : dict[str, np.ndarray]
            Dictionary containing error parameters, which may include delta thresholds.

        Returns
        -------
        dict[str, float]
            A dictionary mapping each coordinate to its corresponding delta threshold.
        """
        logger.info(
            "Parsing delta error thresholds for coordinates: {coords}", coords=coords
        )
        default_err_key = "delta_threshold"
        default_delta_threshold: Optional[float] = match_all(
            (default_err_key,), error_params
        ).pop(default_err_key, None)  # type: ignore[assignment]
        if default_delta_threshold is None:
            warnings.warn(
                f"Default threshold for delta error is not provided under key '{default_err_key}'. "
                "Each coordinate must have its own specific threshold defined as '<coord>_delta_threshold'.",
                LoggedUserWarning,
            )

        coords_err_vals = match_all(
            [coord + "_delta_threshold" for coord in coords], error_params
        )  # type: ignore[arg-type]
        delta_thresholds: dict[str, float] = {
            k.removesuffix("_delta_threshold"): v
            if v is not None
            else default_delta_threshold
            for k, v in coords_err_vals.items()
        }  # type: ignore[assignment]

        for coord in coords:
            specific_err_key = coord + "_delta_threshold"
            delta_threshold = delta_thresholds[coord]
            if delta_threshold is None:
                raise LoggedValueError(
                    f"Threshold for delta error for coordinate '{coord}' is not provided. "
                    f"Either provide '{default_err_key}' or '{specific_err_key}'."
                )
        logger.success(
            "Successfully parsed delta error thresholds:\n{thresholds}",
            thresholds=pprint.pformat(delta_thresholds),
        )
        return delta_thresholds

    def _apply_delta_error(
        self,
        coords: list[str],
        injection_values: dict[str, np.ndarray],
        delta_thresholds: dict[str, float],
    ) -> dict[str, np.ndarray]:
        estimate = {}
        for coord in coords:
            delta_threshold = delta_thresholds[coord]
            injection_val = injection_values[coord]

            val = injection_val + jrd.uniform(
                self.rng_key,
                (self.size,),
                minval=-delta_threshold,
                maxval=delta_threshold,
            )

            estimate[coord] = val
        return estimate

    def _apply_error_model(
        self,
        coords: list[str],
        injection_values: dict[str, np.ndarray],
        error_params: dict[str, np.ndarray],
        rho: float,
    ) -> dict[str, np.ndarray]:
        """Applies the registry functions to generate estimates."""
        estimates = {}
        for key, (err_keys, func) in self.error_function_registry.items():
            # Support both single strings and tuples of parameters
            keys = (key,) if isinstance(key, (str, Parameters)) else key

            if any(k not in coords for k in keys):
                continue

            err_vals = match_all(err_keys, error_params)  # type: ignore[arg-type]

            val = func(estimates=estimates, rho=rho, **injection_values, **err_vals)

            if isinstance(key, tuple):
                for idx, k in enumerate(key):
                    estimates[k] = val[:, idx]
            else:
                estimates[key] = val
        return estimates

    def _save_event(
        self,
        index: int,
        params: list[str],
        posterior: dict[str, np.ndarray],
        injection: dict[str, np.ndarray],
        rho: float,
        width: int,
    ):
        """Handles the HDF5 boilerplate for a single event."""
        tag = str(index).zfill(width)
        event_path = self.root_dir / f"event_{tag}.hdf5"

        # Stack and clean NaNs
        data_stack = np.stack([posterior[p] for p in params], axis=-1)
        data_stack = data_stack[~np.any(np.isnan(data_stack), axis=-1)]
        if data_stack.shape[0] == 0:
            warnings.warn(
                f"All posterior samples for event {index} contain NaNs. "
                "No data will be saved for this event.",
                LoggedUserWarning,
            )
            return

        posterior_samples = to_structured(data_stack, params)
        event = to_structured(np.array([[injection[p] for p in params]]), params)

        compression_args = {"compression": "gzip", "compression_opts": 9}
        with h5py.File(event_path, "w") as ef:
            ef.attrs["parameters"] = np.array(params, dtype="S")
            ef.attrs["rho"] = rho
            group = ef.create_group(self.waveform_name)
            group.create_dataset("approximant", data=self.waveform_name)
            group.create_dataset("injection_data", data=event, **compression_args)
            group.create_dataset(
                "posterior_samples", data=posterior_samples, **compression_args
            )

    def generate_parameter_estimates(self):
        """Generates parameter estimates based on the defined error functions.

        This method should be implemented in subclasses to generate parameter estimates
        using the error functions defined in the `error_function_registry` property.
        """
        available_coords, events = self._load_events()
        coords = available_coords
        if self.coords is not None:
            coords = [c for c in self.coords if c in available_coords]
            if not coords:
                raise LoggedValueError(
                    f"None of the specified coordinates were found in the events. "
                    f"Available coordinates: {available_coords}.",
                )
            if len(coords) < len(self.coords):
                missing = set(self.coords) - set(coords)
                warnings.warn(
                    f"The following specified coordinates were not found in the events and will be ignored: {missing}. "
                    f"Available coordinates: {available_coords}.",
                    LoggedUserWarning,
                )
        n_injections = len(next(iter(events.values())))

        error_params = read_json(self.error_params_filename)

        self.root_dir.mkdir(parents=True, exist_ok=True)
        width = len(str(n_injections - 1))

        if self.derive_parameters:  # Load once
            mesh = default_relation_mesh()

        rhos = [np.nan for _ in range(n_injections)]

        description = "Generating parameter estimates "
        if self.is_delta_error:
            description += "(Delta Error)"
            logger.info(
                "Delta error does not depend on rho, so all injections will have NaN rho values."
            )
        else:
            description += "(Error Model)"
            rhos = 9.0 * np.power(
                np.asarray(
                    jrd.uniform(
                        key=self.rng_key,
                        shape=(n_injections,),
                        minval=np.finfo(np.result_type(float)).tiny,
                    )
                ),
                -1.0 / 3.0,
            )
            rhos = list(map(float, rhos))

        logger.info("rho values for injections: {rhos}", rhos=pprint.pformat(rhos))
        logger.info(
            "Starting parameter estimation generation for {n} injections using {method}.",
            n=n_injections,
            method="delta error" if self.is_delta_error else "error model",
        )

        delta_thresholds = None
        for i in tqdm.tqdm(range(n_injections), desc=description):
            inj_vals = {p: events[p][i] for p in coords}

            if self.is_delta_error:
                if delta_thresholds is None:
                    delta_thresholds = self._parse_delta_thresholds(
                        coords, error_params
                    )
                post_est = self._apply_delta_error(coords, inj_vals, delta_thresholds)
            else:
                post_est = self._apply_error_model(
                    coords, inj_vals, error_params, rhos[i]
                )

            active_params = coords
            if self.derive_parameters:
                post_est = mesh.resolve(initial_state=post_est)
                inj_vals = mesh.resolve(initial_state=inj_vals)
                active_params = sorted(post_est.keys())

            self._save_event(i, active_params, post_est, inj_vals, rhos[i], width)
        logger.success(
            "Successfully generated parameter estimates for {n} injections.",
            n=n_injections,
        )


def synthetic_discrete_pe_main():
    """Command-line interface for generating synthetic discrete parameter estimation
    samples.
    """
    # Global enable validation for all distributions
    enable_validation()

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Generate synthetic discrete parameter estimation samples.",
        epilog="This tool generates synthetic parameter estimation samples based on "
        "injection data and a specified error model.",
    )
    parser.add_argument(
        "--filename",
        help="Path to the output HDF5 file to store the generated injections.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--coords",
        help="Comma-separated list of coordinates to add error into. If not provided, errors will be applied to all coordinates defined in the error function registry.",
        type=lambda s: [c.strip() for c in s.split(",")],
        default=None,
    )
    parser.add_argument(
        "--error-params",
        help="Path to the JSON file containing the error model parameters",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--delta-error",
        help="Apply delta error.",
        action="store_true",
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

    args = parser.parse_args()

    log_info(start=True)

    SyntheticDiscretePE.init_rng_seed(seed=args.seed)

    generator = SyntheticDiscretePE(
        filename=args.filename,
        error_params_filename=args.error_params,
        size=args.size,
        derive_parameters=args.derive_parameters,
        coords=args.coords,
        is_delta_error=args.delta_error,
    )

    generator.generate_parameter_estimates()


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

        filtered_posterior_samples = posterior_samples
        if self.coords is not None:
            idxs = []
            for p in self.coords:
                try:
                    idxs.append(parameters.index(p))
                except ValueError:
                    raise LoggedValueError(
                        f"Parameter '{p}' not found in available parameters: {parameters}.",
                    )
            if not idxs:
                raise LoggedValueError(
                    f"No matching parameters found for coords: {self.coords}. "
                    f"Available parameters: {parameters}.",
                )
            filtered_posterior_samples = posterior_samples[:, idxs]

        cov = np.cov(filtered_posterior_samples, rowvar=False)
        mean = np.mean(filtered_posterior_samples, axis=0)
        cor = np.corrcoef(filtered_posterior_samples, rowvar=False)
        std = np.sqrt(np.diag(cov))
        limits = np.array(
            (
                np.min(filtered_posterior_samples, axis=0),
                np.max(filtered_posterior_samples, axis=0),
            )
        ).T

        compression_args = {"compression": "gzip", "compression_opts": 9}
        with h5py.File(self.filename, "a") as ef:
            if self.waveform_name in ef:
                warnings.warn(
                    f"Group '{self.waveform_name}' already exists in {self.filename}. "
                    "Overwriting existing data.",
                    LoggedUserWarning,
                )
                del ef[self.waveform_name]

            group = ef.create_group(self.waveform_name)
            group.create_dataset("approximant", data=self.waveform_name)
            group.attrs["discrete_waveform"] = self.discrete_waveform
            group.attrs["coords"] = self.coords
            group.create_dataset("mu", data=mean, **compression_args)
            group.create_dataset("std", data=std, **compression_args)
            group.create_dataset("cov", data=cov, **compression_args)
            group.create_dataset("cor", data=cor, **compression_args)
            group.create_dataset("limits", data=limits, **compression_args)


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
        default=SyntheticDiscretePE.waveform_name,
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

    SyntheticAnalyticalPE.init_rng_seed(seed=args.seed)

    pe = SyntheticAnalyticalPE(
        filename=args.filename,
        discrete_waveform=args.discrete_waveform,
        coords=args.coords,
    )

    pe.generate_parameter_estimates()
