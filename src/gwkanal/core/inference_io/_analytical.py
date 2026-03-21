# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import glob
import warnings
from pathlib import Path
from types import ModuleType
from typing import Callable, NamedTuple, Optional

import h5py
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from gwkanal.utils.common import read_json
from gwkokab.utils.exceptions import (
    LoggedFileNotFoundError,
    LoggedImportError,
    LoggedKeyError,
    LoggedUserWarning,
    LoggedValueError,
)


def _load_module(path: Optional[str]) -> Optional[ModuleType]:
    if path is None:
        return None

    import importlib.util

    spec = importlib.util.spec_from_file_location("custom_module", path)
    if spec is None or spec.loader is None:
        raise LoggedImportError(f"Could not load spec for module at {path}")

    custom_module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(custom_module)  # type: ignore
    return custom_module


def _extract_function(
    module: Optional[ModuleType], fn_name: str, default_fn: Callable
) -> Callable:
    if module is None:
        warnings.warn(
            "No module provided. Using identity transform.",
            LoggedUserWarning,
        )
        return default_fn

    if not hasattr(module, fn_name):
        warnings.warn(
            f"The custom module must have a '{fn_name}' function. Using identity transform.",
            LoggedUserWarning,
        )
        return default_fn

    fn: Callable = getattr(module, fn_name)
    return fn


class AnalyticalPEFileData(NamedTuple):
    coords: list[str]
    cov: np.ndarray
    limits: np.ndarray
    mu: np.ndarray
    scale: np.ndarray


class AnalyticalPELoader(BaseModel):
    """Loader for Analytical PE (Parameter Estimation) samples from files matching a
    regex.

    This class handles the ingestion of gravitational-wave posterior samples, manages
    parameter aliasing, performs subsampling, and calculates log-prior weights for
    population inference.
    """

    event_paths: tuple[Path, ...]
    """Tuple of absolute paths to the files containing PE samples."""

    parameter_aliases: dict[str, str] = Field(default_factory=dict)
    """Mapping of internal parameter names to the column names used in the CSV files."""

    default_waveform: str = Field("GWKokabSyntheticAnalyticalPE")
    """Default waveform name to use when loading samples."""

    alternate_waveforms: dict[str, str] = Field(default_factory=dict)
    """Mapping of filenames to alternate waveform names, if needed."""

    analytical_to_model_coord_fn: Callable = Field(lambda x: x)
    """A function that transforms coordinates from the analytical PE format to the
    model's expected format.

    This allows for flexibility in handling different coordinate systems or
    parameterizations used in the PE samples.
    """

    log_abs_det_jacobian_analytical_to_model_coord_fn: Callable = Field(
        lambda x, y: 0.0
    )
    """A function that computes the log absolute determinant of the Jacobian of the
    transformation from analytical PE coordinates to model coordinates.
    """

    @classmethod
    def from_json(cls, config_path: str) -> "AnalyticalPELoader":
        """Initializes the loader from a JSON configuration file.

        Parameters
        ----------
        config_path : str
            Path to the JSON file containing loader settings.

        Returns
        -------
        AnalyticalPELoader
            An instance of AnalyticalPELoader.

        Raises
        ------
        KeyError
            If the 'regex' field is missing in the configuration.
        FileNotFoundError
            If no files match the provided regex pattern.
        """
        raw_data = read_json(config_path)
        if "regex" not in raw_data:
            raise LoggedKeyError("Config error: 'regex' field is required.")

        regex = raw_data.pop("regex")
        filenames = tuple(map(Path, sorted(glob.glob(regex))))

        n_files = len(filenames)
        if n_files == 0:
            raise LoggedFileNotFoundError(
                f"No files matched the regex pattern: {regex}"
            )

        logger.info(f"Initialized loader with {n_files} files found via: {regex}")

        transform_module_path = raw_data.pop("transform_module_path", None)
        transform_module = _load_module(transform_module_path)
        transform = _extract_function(
            transform_module, "analytical_to_model_coord_fn", lambda x: x
        )
        log_abs_det_jacobian_transform = _extract_function(
            transform_module,
            "log_abs_det_jacobian_analytical_to_model_coord_fn",
            lambda x, y: 0.0,
        )

        return cls(
            **raw_data,
            event_paths=filenames,
            analytical_to_model_coord_fn=transform,
            log_abs_det_jacobian_analytical_to_model_coord_fn=log_abs_det_jacobian_transform,
        )

    @classmethod
    def load_file(
        cls, filename: Path | str, waveform_name: str
    ) -> AnalyticalPEFileData:
        """Loads a single PE sample file into a DataFrame.

        Parameters
        ----------
        filename : Path | str
            Path to the sample file.
        waveform_name : str
            Name of the waveform model used.

        Returns
        -------
        AnalyticalPEFileData
            NamedTuple containing the samples and metadata from the file.
        """
        logger.info(f"Loading file '{filename}' with waveform '{waveform_name}'.")
        with h5py.File(filename, "r") as f:
            if waveform_name not in f:
                raise LoggedKeyError(
                    f"Waveform '{waveform_name}' not found in file '{filename}'. "
                    "Available waveforms: " + ", ".join(f.keys())
                )
            group = f[waveform_name]
            cov = group["cov"][()]
            mu = group["mu"][()]
            limits = group["limits"][()]
            coords = group.attrs["coords"].tolist()
            try:
                scale = group["scale"][()]
            except KeyError:
                warnings.warn(
                    f"'scale' dataset not found in '{filename}'. Defaulting to ones.",
                    LoggedUserWarning,
                )
                scale = np.ones_like(mu)

        return AnalyticalPEFileData(
            coords=coords,
            cov=cov,
            limits=limits,
            mu=mu,
            scale=scale,
        )

    def load(
        self, parameters: tuple[str, ...], seed: int = 37
    ) -> dict[str, list[np.ndarray]]:
        """Loads analytical PE data from disk.

        This method reads the mean, covariance, and limits for each event specified
        in `self.event_paths`, validates that the necessary parameters are present,
        and returns them as stacked numpy arrays.

        Parameters
        ----------
        parameters : tuple[str, ...]
            The list of parameters to extract from each file.
        seed : int, optional
            Random seed used for deterministic subsampling, by default 37

        Returns
        -------
        dict[str, list[np.ndarray]]
            A dictionary containing lists of arrays of mean, covariance, and limits for each event.
        """
        logger.info(f"Starting load of {len(self.event_paths)} events.")
        logger.info(f"Target physical parameters: {parameters}")

        posterior_columns = [self.parameter_aliases.get(p, p) for p in parameters]

        data_list: list[AnalyticalPEFileData] = []

        for event_path in self.event_paths:
            event_name = event_path.stem

            waveform_name = self.alternate_waveforms.get(
                event_name, self.default_waveform
            )
            df = self.load_file(event_path, waveform_name=waveform_name)

            if df.coords is None:
                raise LoggedValueError(
                    f"File '{event_path}' is missing 'coords' attribute. Cannot proceed."
                )
            if np.any(df.scale <= 0):
                raise LoggedValueError(
                    f"File '{event_path}' contains non-positive scale values, which are invalid."
                )

            self._validate_columns(df.coords, event_path, posterior_columns)

            data_list.append(df)

        logger.success(f"Finished loading {len(data_list)} events.")

        mean_stack = [data.mu.flatten() for data in data_list]
        cov_stack = [data.cov for data in data_list]
        lower_bound = [data.limits[..., 0].flatten() for data in data_list]
        upper_bound = [data.limits[..., 1].flatten() for data in data_list]
        scale_stack = [data.scale.flatten() for data in data_list]

        return {
            "cov": cov_stack,
            "lower_bound": lower_bound,
            "mean": mean_stack,
            "scale": scale_stack,
            "upper_bound": upper_bound,
        }

    def _validate_columns(
        self, coords: list[str], event: Path | str, columns: list[str]
    ):
        """Ensures all requested or required columns exist in the DataFrame."""
        missing = set(columns) - set(coords)
        if missing != set():
            warnings.warn(
                f"File '{event}' is missing required columns: {missing}. "
                "Use transform to map existing columns to the required ones or check the file format.",
                LoggedUserWarning,
            )
