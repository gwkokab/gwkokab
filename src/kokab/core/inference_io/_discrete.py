# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import glob
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, PositiveInt

from gwkokab.cosmology import PLANCK_2015_Cosmology
from gwkokab.parameters import Parameters as P
from gwkokab.poisson_mean._injection_based_helper import (
    aligned_spin_prior,
    chi_effective_prior_from_isotropic_spins,
    primary_mass_to_chirp_mass_jacobian,
    prior_chieff_chip_isotropic,
)
from gwkokab.utils.tools import error_if, warn_if
from kokab.utils.common import read_json


class DiscreteParameterEstimationLoader(BaseModel):
    """Loader for discrete PE samples from files matching a regex.

    It is inspired by
    `gwpopulation_pipe.data_collection.evaluate_prior <https://docs.ligo.org/RatesAndPopulations/gwpopulation_pipe/api/gwpopulation_pipe.data_collection.evaluate_prior.html#gwpopulation_pipe.data_collection.evaluate_prior>`_.
    """

    filenames: Tuple[str, ...]
    """List of filenames to load samples from."""

    parameter_aliases: Dict[str, str] = Field(default_factory=dict)
    """Alternate names for parameters in the files.

    For example, GWKokab uses 'chirp_mass_detector' but the files may use 'mc_det'. This
    dictionary maps GWKokab parameter names to file column names.
    """

    max_samples: Optional[PositiveInt] = Field(None)
    """Maximum number of samples to load from each file.

    If None, all samples are used.
    """

    mass_prior: Literal[
        None,
        "flat-detector-components",
        "flat-detector-chirp-mass-ratio",
        "flat-source-components",
    ] = Field(None)
    """Mass prior to apply when calculating log prior weights."""

    spin_prior: Literal[None, "component"] = Field(None)
    """Spin prior to apply when calculating log prior weights."""

    distance_prior: Literal[None, "comoving", "euclidean"] = Field(None)
    """Distance prior to apply when calculating log prior weights.

    It assumes cosmo samples.
    """

    @classmethod
    def from_json(cls, config_path: str) -> "DiscreteParameterEstimationLoader":
        """Create a loader from a JSON configuration file.

        Parameters
        ----------
        config_path : str
            Path to JSON configuration file.

        Returns
        -------
        DiscreteParameterEstimationLoader
            Populated loader instance.
        """
        raw_data = read_json(config_path)
        error_if("regex" not in raw_data, msg="The 'regex' field is required.")

        filenames = tuple(sorted(glob.glob(raw_data.pop("regex"))))
        error_if(not filenames, msg="No files matched the regex pattern.")

        return cls(**raw_data, filenames=filenames)

    def load(
        self, parameters: Tuple[str, ...], seed: int = 37
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load samples and calculate log prior weights for each event.

        Parameters
        ----------
        parameters : Tuple[str, ...]
            Parameters to load from the files.
        seed : int, optional
            Seed for random subsampling, by default 37

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing:
            - A list of NumPy arrays with the loaded samples for each event.
            - A list of NumPy arrays with the log prior weights for each event.
        """
        # Pre-resolve aliases to avoid dict lookups in the loop
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
        cosmo = PLANCK_2015_Cosmology()

        data_list, log_prior_list = [], []

        for i, event in enumerate(self.filenames):
            df = pd.read_csv(event, delimiter=" ")
            self._validate_columns(df, event, posterior_columns)

            if self.max_samples:
                df = self._subsample(df, event, seed + i)

            # Initialize log_prior array
            log_prior = np.zeros(len(df))

            # Priors (Log only on first iteration)
            should_log = i == 0

            log_prior += self._calculate_distance_prior(
                df, parameters, cosmo, aliases, should_log
            )
            log_prior += self._calculate_mass_prior(df, parameters, aliases, should_log)
            log_prior += self._calculate_spin_prior(df, parameters, aliases, should_log)

            data_list.append(df[posterior_columns].to_numpy())
            log_prior_list.append(log_prior)

        return data_list, log_prior_list

    def _subsample(self, df: pd.DataFrame, event: str, seed: int) -> pd.DataFrame:
        if self.max_samples is None:  # for type checker
            return df

        n_total = len(df)
        if self.max_samples > n_total:
            warn_if(
                True,
                msg=f"Requested {self.max_samples}, using all {n_total} in {event}.",
            )
            return df
        return df.sample(n=self.max_samples, random_state=seed)

    def _validate_columns(self, df: pd.DataFrame, event: str, columns: List[str]):
        missing = set(columns) - set(df.columns)
        error_if(bool(missing), KeyError, f"File '{event}' missing: {missing}")

    def _get_q(self, df: pd.DataFrame, aliases: Dict) -> np.ndarray:
        """Helper to extract or calculate mass ratio q."""
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
        parameters: Tuple[str, ...],
        cosmo: Any,
        aliases: Dict,
        log: bool,
    ) -> np.ndarray:
        if self.distance_prior is None:
            return 0.0

        z = df[aliases[P.REDSHIFT]].to_numpy()
        if self.distance_prior == "comoving":
            if log:
                logger.info("Using uniform comoving source frame distance prior.")
            return cosmo.logdVcdz(z) + np.log(4 * np.pi) - np.log1p(z)

        if self.distance_prior == "euclidean":
            if log:
                logger.info("Using Euclidean distance prior.")
            dl = cosmo.z_to_DL(z)
            return 2.0 * np.log(dl) + np.log(cosmo.dDLdz(z))

        return 0.0

    def _calculate_mass_prior(
        self, df: pd.DataFrame, parameters: Tuple[str, ...], aliases: Dict, log: bool
    ) -> np.ndarray:
        if self.mass_prior is None:
            return 0.0

        lp = np.zeros(len(df))
        q = self._get_q(df, aliases)
        error_if(
            P.REDSHIFT not in parameters,
            ValueError,
            "Redshift must be included in parameters when using a mass prior.",
        )

        z = df[aliases[P.REDSHIFT]].to_numpy()

        if log and aliases[P.MASS_RATIO] in parameters:
            logger.info(
                "Model is defined in terms of mass ratio, adjusting prior accordingly."
            )

        # Logic for primary mass source/detected
        if aliases[P.PRIMARY_MASS_SOURCE] in df.columns:
            m1_src = df[aliases[P.PRIMARY_MASS_SOURCE]].to_numpy()
            if self.mass_prior == "flat-detector-components":
                lp += 2.0 * np.log1p(z)
            elif self.mass_prior == "flat-detector-chirp-mass-ratio":
                lp -= (
                    np.log(m1_src)
                    - np.log1p(z)
                    + np.log(primary_mass_to_chirp_mass_jacobian(q))
                )
            if aliases[P.MASS_RATIO] in parameters:
                lp += np.log(m1_src)

        elif aliases[P.PRIMARY_MASS_DETECTED] in df.columns:
            m1_det = df[aliases[P.PRIMARY_MASS_DETECTED]].to_numpy()
            if self.mass_prior == "flat-detector-chirp-mass-ratio":
                lp -= np.log(m1_det) + np.log(primary_mass_to_chirp_mass_jacobian(q))
            if aliases[P.MASS_RATIO] in parameters:
                lp += np.log(m1_det)

        if any(
            k in df.columns
            for k in [aliases[P.CHIRP_MASS], aliases[P.CHIRP_MASS_DETECTOR]]
        ):
            if log:
                logger.info(
                    "Model is defined in terms of chirp mass, adjusting prior accordingly."
                )
            lp += np.log(primary_mass_to_chirp_mass_jacobian(q))

        return lp

    def _calculate_spin_prior(
        self, df: pd.DataFrame, parameters: Tuple[str, ...], aliases: Dict, log: bool
    ) -> np.ndarray:
        lp = 0.0
        if self.spin_prior == "component":
            if log:
                logger.info("Assuming uniform in component spin prior for all events.")
            lp -= np.log(4.0)

        # Effective and Precessing Spin
        if aliases[P.EFFECTIVE_SPIN] in df.columns:
            chi_eff = df[aliases[P.EFFECTIVE_SPIN]].to_numpy()
            q = self._get_q(df, aliases)

            if aliases[P.PRECESSING_SPIN] in df.columns:
                chi_p = df[aliases[P.PRECESSING_SPIN]].to_numpy()
                lp += np.log(prior_chieff_chip_isotropic(chi_eff, chi_p, q))
            else:
                lp += np.log(chi_effective_prior_from_isotropic_spins(chi_eff, q))

        # Magnitude Priors
        for key in [P.CHI_1, P.CHI_2]:
            if aliases[key] in df.columns:
                lp += np.log(aligned_spin_prior(df[aliases[key]].to_numpy()))

        return lp
