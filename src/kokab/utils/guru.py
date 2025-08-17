# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from jax import random as jrd
from jaxtyping import PRNGKeyArray
from loguru import logger
from numpyro.util import is_prng_key

from gwkokab.utils.tools import error_if


class Guru:
    """Guru is a class which contains all the common functionality among Genie, Sage and
    Monk classes.
    """

    _rng_key: PRNGKeyArray

    @property
    def rng_key(self) -> PRNGKeyArray:
        self._rng_key, subkey = jrd.split(self._rng_key)
        return subkey

    def set_rng_key(
        self, *, key: Optional[PRNGKeyArray] = None, seed: Optional[int] = None
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
            self._rng_key = key
        elif seed is not None:
            error_if(
                not isinstance(seed, int),
                msg=f"Expected an integer seed, got {type(seed)}.",
            )
            error_if(
                seed < 0,
                msg=f"Seed must be a non-negative integer, got {seed}.",
            )
            logger.info(f"Setting the random number generator key with seed {seed}.")
            key = jrd.PRNGKey(seed)
            self._rng_key = key
