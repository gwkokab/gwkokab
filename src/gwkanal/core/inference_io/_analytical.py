# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import glob
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import h5py
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from gwkanal.utils.common import read_json
from gwkokab.utils.tools import error_if, warn_if


def _extract_transform(path: Optional[str]) -> Callable:
    if path is None:
        warn_if(
            True,
            msg="No 'transform_module_path' provided. Using identity transform.",
        )
        return lambda x: x

    import importlib.util

    spec = importlib.util.spec_from_file_location("custom_module", path)
    error_if(
        spec is None or spec.loader is None,
        ImportError,
        f"Could not load spec for module at {path}",
    )

    spec = importlib.util.spec_from_file_location("custom_module", path)
    custom_module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(custom_module)  # type: ignore

    error_if(
        not hasattr(custom_module, "transform"),
        msg="The custom module must have a 'transform' function.",
    )

    transform: Callable = getattr(custom_module, "transform")
    return transform


class AnalyticalPEFileData(NamedTuple):
    coords: list[str]
    cov: np.ndarray
    limits: np.ndarray
    mu: np.ndarray


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
        error_if(
            "regex" not in raw_data,
            KeyError,
            msg="Config error: 'regex' field is required.",
        )

        regex = raw_data.pop("regex")
        filenames = tuple(map(Path, sorted(glob.glob(regex))))

        n_files = len(filenames)
        error_if(
            n_files == 0,
            FileNotFoundError,
            msg=f"No files matched the regex pattern: {regex}",
        )

        logger.info(f"Initialized loader with {n_files} files found via: {regex}")

        transform_module_path = raw_data.pop("transform_module_path")
        transform = _extract_transform(transform_module_path)

        return cls(
            **raw_data,
            event_paths=filenames,
            analytical_to_model_coord_fn=transform,
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
            error_if(
                waveform_name not in f,
                KeyError,
                f"Waveform '{waveform_name}' not found in file '{filename}'.",
            )
            group = f[waveform_name]
            cov = group["cov"][()]
            mu = group["mu"][()]
            limits = group["limits"][()]
            coords = group.attrs["coords"].tolist()

        return AnalyticalPEFileData(
            coords=coords,
            cov=cov,
            limits=limits,
            mu=mu,
        )

    def load(
        self, parameters: tuple[str, ...], seed: int = 37
    ) -> dict[str, np.ndarray]:
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
        dict[str, np.ndarray]
            A dictionary containing stacked arrays of mean, covariance, and limits for each event.
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

            self._validate_columns(df.coords, event_path, posterior_columns)

            data_list.append(df)

        logger.success(f"Finished loading {len(data_list)} events.")

        mean_stack = np.stack([data.mu for data in data_list], axis=0)
        cov_stack = np.stack([data.cov for data in data_list], axis=0)
        limits_stack = np.stack([data.limits.T for data in data_list], axis=0)

        return {"mean": mean_stack, "cov": cov_stack, "limits": limits_stack}

    def _validate_columns(
        self, coords: list[str], event: Path | str, columns: list[str]
    ):
        """Ensures all requested or required columns exist in the DataFrame."""
        missing = set(columns) - set(coords)
        warn_if(
            missing != set(),
            msg=f"File '{event}' is missing required columns: {missing}",
        )
