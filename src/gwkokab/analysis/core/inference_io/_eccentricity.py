# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit native eccentricity conventions and source-reference transport."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gwkokab.utils.eccentricity import evolve_eccentricity
from gwkokab.utils.exceptions import LoggedValueError


ECCENTRICITY_LOG_PRIOR_COLUMN = "_gwkokab_log_eccentricity_prior"


class EccentricityEvolutionConfig(BaseModel):
    """Opt-in quadrupole transport with a native uniform eccentricity prior.

    Frequencies must be audited from the waveform interface/run configuration.
    No starting-frequency, reference-frequency or prior-bound guessing is done.
    The full Peters inverse is a global ansatz and is the default backstop.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    native_frequency_hz: float = Field(gt=0, allow_inf_nan=False)
    """Audited frequency defining the native eccentricity (not likelihood cutoff)."""
    native_frequency_frame: Literal["detector", "source"]
    """Frame of the audited native frequency; must be explicitly supplied."""
    reference_frequency_source_hz: float = Field(
        default=10.0, gt=0, allow_inf_nan=False
    )
    method: Literal["peters", "small-e"] = "peters"
    native_prior: Literal["uniform"] = "uniform"
    native_prior_min: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    native_prior_max: float = Field(gt=0, lt=1, allow_inf_nan=False)
    condition_native_max: float | None = Field(
        default=None, gt=0, lt=1, allow_inf_nan=False
    )
    """Optional native upper cut; conditions samples AND normalizes their prior."""

    @model_validator(mode="after")
    def _validate_support(self):
        if self.native_prior_min >= self.native_prior_max:
            raise ValueError("Native prior bounds must be strictly ordered.")
        if self.condition_native_max is not None and not (
            self.native_prior_min < self.condition_native_max <= self.native_prior_max
        ):
            raise ValueError("Native conditioning bound must lie inside prior support.")
        return self

    def apply(
        self, df: pd.DataFrame, eccentricity_column: str, redshift_column: str
    ) -> pd.DataFrame:
        """Condition native rows, then transport eccentricity and its PE prior.

        Return a copy with the eccentricity column replaced, all other columns
        and row alignment intact, and a reserved log reference-prior column.
        Subsample the returned whole rows, never the eccentricity alone.
        """
        if (
            df.attrs.get("eccentricity_evolution") is not None
            or "eccentricity_source_ref" in df.columns
            or ECCENTRICITY_LOG_PRIOR_COLUMN in df.columns
        ):
            raise LoggedValueError("Refusing to evolve already converted eccentricity.")
        for column in (eccentricity_column, redshift_column):
            if column not in df:
                raise LoggedValueError(
                    "Eccentricity evolution requires column {}.", column
                )
        e = df[eccentricity_column].to_numpy(dtype=float)
        z = df[redshift_column].to_numpy(dtype=float)
        if np.any(~np.isfinite(z)) or np.any(z < 0):
            raise LoggedValueError(
                "Eccentricity evolution requires finite redshift >= 0."
            )
        if np.any(~np.isfinite(e)) or np.any(
            (e < self.native_prior_min) | (e > self.native_prior_max)
        ):
            raise LoggedValueError("Samples are outside the declared native e prior.")
        upper = (
            self.native_prior_max
            if self.condition_native_max is None
            else self.condition_native_max
        )
        result = df.loc[e <= upper].copy()
        if len(result) == 0:
            raise LoggedValueError(
                "Native eccentricity conditioning removed every row."
            )
        e = result[eccentricity_column].to_numpy(dtype=float)
        z = result[redshift_column].to_numpy(dtype=float)
        native_source = self.native_frequency_hz * (
            1.0 + z if self.native_frequency_frame == "detector" else np.ones_like(z)
        )
        ratio = self.reference_frequency_source_hz / native_source
        evolved, log_jacobian = evolve_eccentricity(e, ratio, self.method)
        result[eccentricity_column] = evolved
        result[ECCENTRICITY_LOG_PRIOR_COLUMN] = log_jacobian - np.log(
            upper - self.native_prior_min
        )
        result.attrs["eccentricity_evolution"] = self.model_dump(mode="json")
        return result
