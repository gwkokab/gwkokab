# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import glob
import warnings
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, PositiveInt

from gwkanal.core.utils import from_structured
from gwkanal.utils.common import read_json
from gwkokab.cosmology import Cosmology, default_cosmology
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean._injection_based_helper import (
    aligned_spin_prior,
    chi_effective_prior_from_isotropic_spins,
    primary_mass_to_chirp_mass_jacobian,
    prior_chieff_chip_isotropic,
)
from gwkokab.utils.exceptions import (
    LoggedFileNotFoundError,
    LoggedKeyError,
    LoggedUserWarning,
    LoggedValueError,
)


class DiscretePELoader(BaseModel):
    """Loader for Discrete PE (Parameter Estimation) samples from files matching a
    regex.

    This class handles the ingestion of gravitational-wave posterior samples, manages
    parameter aliasing, performs subsampling, and calculates log-prior weights for
    population inference.
    """

    filenames: tuple[Path, ...]
    """Tuple of absolute paths to the sample files."""

    parameter_aliases: dict[str, str] = Field(default_factory=dict)
    """Mapping of internal parameter names to the column names used in the CSV files."""

    max_samples: Optional[PositiveInt] = Field(None)
    """If set, limits the number of samples loaded per event to this value."""

    default_datasets: tuple[str, ...] = Field(
        ("/GWKokabSyntheticDiscretePE/posterior_samples",)
    )
    """Default dataset names to look for in HDF5 files, in order of preference."""

    alternate_datasets: dict[str, str] = Field(default_factory=dict)
    """Mapping of filenames to an alternate dataset name, overriding the default
    dataset(s).
    """

    mass_prior: Literal[
        None,
        "flat-detector-components",
        "flat-detector-chirp-mass-ratio",
        "flat-source-components",
    ] = Field(None)
    """The mass prior assumed during the original PE run to be removed/reweighted."""

    spin_prior: Literal[None, "component"] = Field(None)
    """The spin prior assumed during the original PE run."""

    distance_prior: Literal[None, "comoving", "euclidean"] = Field(None)
    """The distance prior assumed; used to calculate volume-sensitive weights."""

    @classmethod
    def from_json(cls, config_path: str) -> "DiscretePELoader":
        """Initializes the loader from a JSON configuration file.

        Parameters
        ----------
        config_path : str
            Path to the JSON file containing loader settings.

        Returns
        -------
        DiscreteParameterEstimationLoader
            An instance of DiscreteParameterEstimationLoader.

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

        default_datasets = raw_data.pop(
            "default_datasets", ["/GWKokabSyntheticDiscretePE/posterior_samples"]
        )
        for i in range(len(default_datasets)):
            dataset = default_datasets[i]
            if not dataset.startswith("/"):
                warnings.warn(
                    f"Dataset '{dataset}' does not start with '/'. Prepending '/' to ensure valid HDF5 path.",
                    LoggedUserWarning,
                )
                default_datasets[i] = "/" + dataset
        if isinstance(default_datasets, list):
            default_datasets = tuple(default_datasets)

        alternate_datasets = raw_data.pop("alternate_datasets", {})
        for event, dataset in alternate_datasets.items():
            if not dataset.startswith("/"):
                warnings.warn(
                    f"Dataset '{dataset}' for event '{event}' does not start with '/'. Prepending '/' to ensure valid HDF5 path.",
                    LoggedUserWarning,
                )
                alternate_datasets[event] = "/" + dataset

        logger.info(f"Initialized loader with {n_files} files found via: {regex}")
        return cls(
            **raw_data,
            default_datasets=default_datasets,
            filenames=filenames,
            alternate_datasets=alternate_datasets,
        )

    @classmethod
    def load_file(
        cls, filename: Path | str, datasets: str | tuple[str, ...]
    ) -> pd.DataFrame:
        """Loads a single PE sample file into a DataFrame.

        Parameters
        ----------
        filename : Path | str
            Path to the sample file.
        datasets : str | tuple[str, ...]
            Name or tuple of names of the dataset(s) to load from the HDF5 file, in order of preference.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the samples from the file.
        """
        if isinstance(datasets, str):
            datasets = (datasets,)

        with h5py.File(filename, "r") as f:
            for dataset in datasets:
                if dataset in f:
                    data_structured = f[dataset][()]
                    data_array, columns = from_structured(data_structured)
                    df = pd.DataFrame(data=data_array, columns=columns)

                    logger.info(f"Loading file '{filename}' with dataset '{dataset}'.")

                    return df

            raise LoggedKeyError(
                f"None of the specified datasets {datasets} found in file '{filename}'."
                f" Available datasets: {list(f.keys())}"
            )

    def load(
        self, parameters: tuple[str, ...], seed: int = 37
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Loads samples from disk and computes the corresponding log-prior weights.

        It is inspired by :func:`~gwpopulation_pipe.data_collection.evaluate_prior`.

        Parameters
        ----------
        parameters : tuple[str, ...]
            The list of parameters to extract from each file.
        seed : int, optional
            Random seed used for deterministic subsampling, by default 37

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            A tuple containing:
                - A list of arrays (one per event) containing the requested parameters.
                - A list of arrays (one per event) containing the log-prior weights.
        """
        logger.info(f"Starting load of {len(self.filenames)} events.")
        logger.info(f"Target physical parameters: {parameters}")

        aliases = {
            p: self.parameter_aliases.get(p, p)
            for p in [
                P.CHIRP_MASS,
                P.CHIRP_MASS_DETECTOR,
                P.CHI_1,
                P.CHI_2,
                P.EFFECTIVE_SPIN,
                P.MASS_RATIO,
                P.PRECESSING_SPIN,
                P.PRIMARY_MASS_DETECTED,
                P.PRIMARY_MASS_SOURCE,
                P.PRIMARY_SPIN_MAGNITUDE,
                P.REDSHIFT,
                P.SECONDARY_MASS_DETECTED,
                P.SECONDARY_MASS_SOURCE,
                P.SECONDARY_SPIN_MAGNITUDE,
            ]
        }

        posterior_columns = [self.parameter_aliases.get(p, p) for p in parameters]
        cosmo = default_cosmology()
        data_list, log_prior_list = [], []

        for i, event_path in enumerate(self.filenames):
            event_name = event_path.stem

            datasets = self.alternate_datasets.get(event_name, self.default_datasets)
            df = self.load_file(event_path, datasets=datasets)

            self._validate_columns(df, event_path, posterior_columns)

            df = self._subsample(df, event_path, seed + i)

            log_prior = np.zeros(len(df))
            should_log = i == 0  # Log logic only once to avoid spam

            # Perform prior reweighting
            log_prior += self._calculate_distance_prior(
                df, parameters, cosmo, aliases, should_log
            )
            log_prior += self._calculate_mass_prior(df, parameters, aliases, should_log)
            log_prior += self._calculate_spin_prior(df, parameters, aliases, should_log)

            data_list.append(df[posterior_columns].to_numpy())
            log_prior_list.append(log_prior)

        logger.success(f"Finished loading {len(data_list)} events.")
        return data_list, log_prior_list

    def _subsample(
        self, df: pd.DataFrame, event: Path | str, seed: int
    ) -> pd.DataFrame:
        """Helper to downsample a DataFrame if it exceeds max_samples."""
        if self.max_samples is None:
            return df

        n_total = len(df)
        if self.max_samples >= n_total:
            warnings.warn(
                f"Subsampling skipped: {event} has {n_total} samples (requested {self.max_samples}).",
                LoggedUserWarning,
            )
            return df

        return df.sample(n=self.max_samples, random_state=seed)

    def _validate_columns(
        self, df: pd.DataFrame, event: Path | str, columns: list[str]
    ):
        """Ensures all requested or required columns exist in the DataFrame."""
        missing = set(columns) - set(df.columns)
        if missing != set():
            raise LoggedValueError(
                f"File '{event}' is missing required columns: {missing}"
            )

    def _get_q(self, df: pd.DataFrame, aliases: dict) -> np.ndarray:
        """Calculates mass ratio q = m2/m1, handling various alias possibilities."""
        if aliases[P.MASS_RATIO] in df.columns:
            return df[aliases[P.MASS_RATIO]].to_numpy()
        if aliases[P.PRIMARY_MASS_SOURCE] in df.columns:
            return (
                df[aliases[P.SECONDARY_MASS_SOURCE]].to_numpy()
                / df[aliases[P.PRIMARY_MASS_SOURCE]].to_numpy()
            )
        return (
            df[aliases[P.SECONDARY_MASS_DETECTED]].to_numpy()
            / df[aliases[P.PRIMARY_MASS_DETECTED]].to_numpy()
        )

    def _calculate_distance_prior(
        self,
        df: pd.DataFrame,
        parameters: tuple[str, ...],
        cosmo: Cosmology,
        aliases: dict,
        log: bool,
    ) -> np.ndarray:
        """Calculates the log-weight for the distance/redshift prior."""
        if self.distance_prior is None:
            return 0.0

        if P.REDSHIFT not in parameters:
            raise LoggedValueError("Distance prior requires Redshift.")

        z = df[aliases[P.REDSHIFT]].to_numpy()
        if self.distance_prior == "comoving":
            if log:
                logger.info("Using Comoving Distance prior.")
            return cosmo.logdVcdz(z) + np.log(4 * np.pi) - np.log1p(z)

        if self.distance_prior == "euclidean":
            if log:
                logger.info("Using Euclidean Distance prior.")
            dl = cosmo.z_to_DL(z)
            return 2.0 * np.log(dl) + np.log(cosmo.dDLdz(z))

        return 0.0

    def _calculate_mass_prior(
        self, df: pd.DataFrame, parameters: tuple[str, ...], aliases: dict, log: bool
    ) -> np.ndarray:
        """Calculates the log-weight for mass-related priors and Jacobians."""
        if self.mass_prior is None:
            return 0.0

        lp = np.zeros(len(df))
        q = self._get_q(df, aliases)
        if P.REDSHIFT not in parameters:
            raise LoggedValueError("Mass prior reweighting requires Redshift.")

        z = df[aliases[P.REDSHIFT]].to_numpy()

        if log:
            logger.info(
                "Applying mass prior reweighting: {prior}", prior=self.mass_prior
            )

        if P.PRIMARY_MASS_SOURCE in parameters:
            m1_src = df[aliases[P.PRIMARY_MASS_SOURCE]].to_numpy()
            if self.mass_prior == "flat-detector-components":
                lp += 2.0 * np.log1p(z)
            elif self.mass_prior == "flat-detector-chirp-mass-ratio":
                lp -= (
                    np.log(m1_src)
                    - np.log1p(z)
                    + np.log(primary_mass_to_chirp_mass_jacobian(q))
                )
            if P.MASS_RATIO in parameters:
                lp += np.log(m1_src)

        elif P.PRIMARY_MASS_DETECTED in parameters:
            m1_det = df[aliases[P.PRIMARY_MASS_DETECTED]].to_numpy()
            if self.mass_prior == "flat-detector-chirp-mass-ratio":
                lp -= np.log(m1_det) + np.log(primary_mass_to_chirp_mass_jacobian(q))
            if P.MASS_RATIO in parameters:
                lp += np.log(m1_det)

        if any(k in parameters for k in [P.CHIRP_MASS, P.CHIRP_MASS_DETECTOR]):
            lp += np.log(primary_mass_to_chirp_mass_jacobian(q))

        return lp

    def _calculate_spin_prior(
        self, df: pd.DataFrame, parameters: tuple[str, ...], aliases: dict, log: bool
    ) -> np.ndarray:
        """Calculates log-weights for spin priors (effective, precessing, and
        magnitude).
        """
        lp = 0.0
        if self.spin_prior == "component":
            if log:
                logger.info("Reweighting with uniform component spin prior.")
            lp -= np.log(4.0)

        if P.EFFECTIVE_SPIN in parameters:
            chi_eff = df[aliases[P.EFFECTIVE_SPIN]].to_numpy()
            q = self._get_q(df, aliases)

            if P.PRECESSING_SPIN in parameters:
                if log:
                    logger.info(
                        "Applying Effective Spin and Precessing Spin isotropic prior."
                    )
                chi_p = df[aliases[P.PRECESSING_SPIN]].to_numpy()
                lp += np.log(prior_chieff_chip_isotropic(chi_eff, chi_p, q))
            else:
                if log:
                    logger.info("Applying Effective Spin isotropic prior.")
                lp += np.log(chi_effective_prior_from_isotropic_spins(chi_eff, q))

        for key in [P.CHI_1, P.CHI_2]:
            if key in parameters:
                if log:
                    logger.info(f"Applying aligned spin prior for {key}")
                lp += np.log(aligned_spin_prior(df[aliases[key]].to_numpy()))

        return lp
