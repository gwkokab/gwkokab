# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
import warnings
from collections.abc import Callable
from typing import List, Optional, Tuple

import numpy as np
import tqdm
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger
from numpyro.util import is_prng_key

from ..models.utils import ScaledMixture
from ..models.wrappers import ModelRegistry
from ._utils import ensure_dat_extension


PROGRESS_BAR_TEXT_WITDH = 25

error_magazine = ModelRegistry()


class PopulationFactory:
    error_dir: str = "posteriors"
    event_filename: str = "event_{}"
    injection_filename: str = "injections"
    realizations_dir: str = "realization_{}"
    root_dir: str = "data"
    verbose: bool = True

    def __init__(
        self,
        model: ScaledMixture,
        parameters: List[str],
        logVT_fn: Optional[Callable[[Array], Array]],
        ERate_fn: Callable[[ScaledMixture], Array],
        num_realizations: int = 5,
        error_size: int = 2_000,
    ) -> None:
        """Class with methods equipped to generate population for each realizations and
        adding errors in it.

        Parameters
        ----------
        model : ScaledMixture
            Model for the population.
        parameters : List[str]
            Parameters for the model in order.
        logVT_fn : Callable[[Array], Array]
            logarithm of volume time sensitivity function.
        ERate_fn : Callable[[ScaledMixture], Array]
            Expected rate function.
        num_realizations : int, optional
            Number of realizations to generate, by default 5
        error_size : int, optional
            Size of the error to add in the population, by default 2_000

        Raises
        ------
        ValueError
            If model is not provided.
        ValueError
            If provided model is not a `ScaledMixture` model.
        ValueError
            If parameters are not provided.
        """
        self.model = model
        self.parameters = parameters
        self.logVT_fn = logVT_fn
        self.ERate_fn = ERate_fn
        self.num_realizations = num_realizations
        self.error_size = error_size

        self.event_filename = ensure_dat_extension(self.event_filename)
        self.injection_filename = ensure_dat_extension(self.injection_filename)

        if self.model is None:
            raise ValueError("Model is not provided.")
        if not isinstance(self.model, ScaledMixture):
            raise ValueError(
                "The model must be a `ScaledMixture` model for multi-rate model."
                "See `gwkokab.model.utils.ScaledMixture` for more details."
            )

        if self.parameters == []:
            raise ValueError("Parameters are not provided.")

    def _generate_population(
        self, size: int, *, key: PRNGKeyArray
    ) -> Tuple[Array, Array, Array, Array]:
        r"""Generate population for a realization."""

        old_size = size
        if self.logVT_fn is not None:
            size += int(1e4)

        population, [indices] = self.model.sample_with_intermediates(key, (size,))

        raw_population = population
        raw_indices = indices

        if self.logVT_fn is not None:
            _, key = jrd.split(key)

            vt = jnn.softmax(self.logVT_fn(population))
            _, key = jrd.split(key)
            index = jrd.choice(
                key, jnp.arange(population.shape[0]), p=vt, shape=(old_size,)
            )

            population = population[index]
            indices = indices[index]

        return raw_population, raw_indices, population, indices

    def _generate_realizations(self, key: PRNGKeyArray) -> None:
        r"""Generate realizations for the population."""
        poisson_key, rate_key = jrd.split(key)
        exp_rate = self.ERate_fn(self.model)
        logger.debug(f"Expected rate for the population is {exp_rate}")
        size = int(jrd.poisson(poisson_key, exp_rate))
        logger.debug(f"Population size is {size}")
        key = rate_key
        if size <= 0:
            raise ValueError(
                "Population size is zero. This can be a result of following:\n"
                "\t1. The rate is zero.\n"
                "\t2. The volume is zero.\n"
                "\t3. The models are not selected for rate calculation.\n"
                "\t4. VT file is not provided or is not valid.\n"
                "\t5. Or some other reason."
            )
        pop_keys = jrd.split(key, self.num_realizations)
        os.makedirs(self.root_dir, exist_ok=True)
        with tqdm.tqdm(
            range(self.num_realizations),
            unit="realization",
            total=self.num_realizations,
        ) as pbar:
            pbar.set_description(
                "Generating population".ljust(PROGRESS_BAR_TEXT_WITDH, " ")
            )
            for i in pbar:
                realizations_path = os.path.join(
                    self.root_dir, self.realizations_dir.format(i)
                )
                os.makedirs(realizations_path, exist_ok=True)

                raw_population, raw_indices, population, indices = (
                    self._generate_population(size, key=pop_keys[i])
                )

                if population.shape == ():
                    continue

                injections_file_path = os.path.join(
                    realizations_path, self.injection_filename
                )
                color_indices_file_path = os.path.join(realizations_path, "color.dat")
                np.savetxt(
                    injections_file_path,
                    population,
                    header=" ".join(self.parameters),
                    comments="",  # To remove the default comment character '#'
                )
                np.savetxt(
                    color_indices_file_path,
                    indices,
                    comments="",  # To remove the default comment character '#'
                    fmt="%d",
                )
                raw_injections_file_path = os.path.join(
                    realizations_path, "raw_" + self.injection_filename
                )
                raw_color_indices_file_path = os.path.join(
                    realizations_path, "raw_color.dat"
                )
                np.savetxt(
                    raw_injections_file_path,
                    raw_population,
                    header=" ".join(self.parameters),
                    comments="",  # To remove the default comment character '#'
                )
                np.savetxt(
                    raw_color_indices_file_path,
                    raw_indices,
                    comments="",  # To remove the default comment character '#'
                    fmt="%d",
                )

    def _add_error(self, realization_number, *, key: PRNGKeyArray) -> None:
        r"""Adds error to the realizations' population."""
        realizations_path = os.path.join(
            self.root_dir, self.realizations_dir.format(realization_number)
        )

        heads: List[List[int]] = []
        error_fns: List[Callable[[Array, int, PRNGKeyArray], Array]] = []

        for head, err_fn in error_magazine.registry.items():
            _head = []
            for h in head:
                i = self.parameters.index(h)
                _head.append(i)
            heads.append(_head)
            error_fns.append(err_fn)

        output_dir = os.path.join(
            realizations_path, self.error_dir, self.event_filename
        )

        os.makedirs(os.path.join(realizations_path, self.error_dir), exist_ok=True)

        injections_file_path = os.path.join(realizations_path, self.injection_filename)
        data_inj = np.loadtxt(injections_file_path, skiprows=1)
        keys = jrd.split(key, data_inj.shape[0] * (len(heads) + 1))

        for index in range(data_inj.shape[0]):
            noisy_data = np.empty((self.error_size, len(self.parameters)))
            data = data_inj[index]
            i = 0
            for head, err_fn in zip(heads, error_fns):
                key_idx = index * len(heads) + i
                noisy_data_i: Array = err_fn(data[head], self.error_size, keys[key_idx])
                if noisy_data_i.ndim == 1:
                    noisy_data_i = noisy_data_i.reshape(self.error_size, -1)
                noisy_data[:, head] = noisy_data_i
                i += 1
            nan_mask = np.isnan(noisy_data).any(axis=1)
            noisy_data = noisy_data[~nan_mask]  # type: ignore
            count = np.count_nonzero(noisy_data)
            if count < 1:
                warnings.warn(
                    f"Skipping file {index} due to all NaN values or insufficient data.",
                    category=UserWarning,
                )
                continue
            np.savetxt(
                output_dir.format(index),
                noisy_data,
                header=" ".join(self.parameters),
                comments="",  # To remove the default comment character '#'
            )

        if self.logVT_fn is None:
            raw_output_dir = os.path.join(
                realizations_path, self.error_dir, self.event_filename
            )

            os.makedirs(
                os.path.join(realizations_path, "raw_" + self.error_dir), exist_ok=True
            )

            injections_file_path = os.path.join(
                realizations_path, "raw_" + self.injection_filename
            )
            raw_data_inj = np.loadtxt(injections_file_path, skiprows=1)
            keys = jrd.split(key, raw_data_inj.shape[0] * len(heads))

            for index in range(raw_data_inj.shape[0]):
                noisy_data = np.empty((self.error_size, len(self.parameters)))
                data = raw_data_inj[index]
                i = 0
                for head, err_fn in zip(heads, error_fns):
                    key_idx = index * len(heads) + i
                    noisy_data_i: Array = err_fn(
                        data[head], self.error_size, keys[key_idx]
                    )
                    if noisy_data_i.ndim == 1:
                        noisy_data_i = noisy_data_i.reshape(self.error_size, -1)
                    noisy_data[:, head] = noisy_data_i
                    i += 1
                nan_mask = np.isnan(noisy_data).any(axis=1)
                masked_noisey_data = noisy_data[~nan_mask]
                count = np.count_nonzero(masked_noisey_data)
                if count < 2:
                    warnings.warn(
                        f"Skipping file {index} due to all NaN values or insufficient data.",
                        category=UserWarning,
                    )
                    continue
                np.savetxt(
                    raw_output_dir.format(index),
                    masked_noisey_data,
                    header=" ".join(self.parameters),
                    comments="",  # To remove the default comment character '#'
                )

    def produce(self, key: Optional[PRNGKeyArray] = None) -> None:
        """Generate realizations and add errors to the populations.

        Parameters
        ----------
        key : Optional[PRNGKeyArray], optional
            Pseudo-random number generator key for reproducibility, by default None
        """
        if key is None:
            key = jrd.PRNGKey(np.random.randint(0, 2**32 - 1))
        else:
            assert is_prng_key(key)
        self._generate_realizations(key)
        keys = jrd.split(key, self.num_realizations)
        with tqdm.tqdm(
            range(self.num_realizations),
            unit="realization",
            total=self.num_realizations,
        ) as pbar:
            pbar.set_description("Adding errors".ljust(PROGRESS_BAR_TEXT_WITDH, " "))
            for i in pbar:
                self._add_error(i, key=keys[i])
