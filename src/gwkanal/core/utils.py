# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Optional

import numpy as np
from jax import random as jrd
from jaxtyping import PRNGKeyArray
from loguru import logger
from numpyro.util import is_prng_key

from gwkokab.utils.tools import error_if


class PRNGKeyMixin:
    _rng_key: PRNGKeyArray

    @property
    def rng_key(self) -> PRNGKeyArray:
        self._rng_key, subkey = jrd.split(self._rng_key)
        return subkey

    @classmethod
    def set_rng_key(
        cls, *, key: Optional[PRNGKeyArray] = None, seed: Optional[int] = None
    ) -> None:
        error_if(
            key is None and seed is None,
            msg="Either 'key' or 'seed' must be provided to set the random number generator key.",
        )
        if key is not None:
            error_if(
                not is_prng_key(key),
                msg=f"Expected a PRNGKeyArray, got {type(key)}.",
            )
            logger.info(f"Setting the random number generator key to {key}.")
            cls._rng_key = key

        if seed is not None:
            error_if(
                not isinstance(seed, int),
                msg=f"Expected an integer seed, got {type(seed)}.",
            )
            error_if(
                seed < 0,
                msg=f"Seed must be a non-negative integer, got {seed}.",
            )
            logger.info(f"Setting the random number generator key with seed {seed}.")
            key = jrd.key(seed)
            cls._rng_key = key


def to_structured(data: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Converts a 2D array to a structured array with given field names.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_features).
    names : Sequence[str]
        List of field names for the structured array.

    Returns
    -------
    np.ndarray
        Structured array with fields named according to `names`.
    """
    dtype = [(n, "<f8") for n in names]
    return np.core.records.fromarrays(data.T, dtype=dtype)


def from_structured(data: np.ndarray) -> tuple[np.ndarray, Sequence[str]]:
    """Converts a structured array to a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Structured array.

    Returns
    -------
    tuple[np.ndarray, Sequence[str]]
        A tuple containing:
        - 2D array of shape (n_samples, n_features).
        - List of field names from the structured array.
    """
    return np.vstack([data[name] for name in data.dtype.names]).T, data.dtype.names
