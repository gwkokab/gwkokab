# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
import warnings
from collections.abc import Callable
from typing import List, Optional

import numpy as np
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from numpyro.util import is_prng_key

from ..models.utils import ScaledMixture
from ..models.wrappers import ModelRegistry
from ._utils import ensure_dat_extension, get_progress_bar


PROGRESS_BAR_TEXT_WITDH = 25

error_magazine = ModelRegistry()


class PopulationFactory:
    error_dir: str = "{}_posteriors"
    event_filename: str = "event_{}"
    injection_filename: str = "{}_injections"
    realizations_dir: str = "realization_{}"
    root_dir: str = "data"
    verbose: bool = True

    heads: List[List[int]] = []
    error_fns: List[Callable[[Array, int, PRNGKeyArray], Array]] = []

    save_raw_injections: bool = False
    save_raw_PEs: bool = False
    save_weighted_injections: bool = False
    save_weighted_PEs: bool = False
    save_ref_probs: bool = False

    def __init__(
        self,
        model: ScaledMixture,
        parameters: List[str],
        logVT_fn: Optional[Callable[[Array], Array]],
        ERate_fn: Callable[[ScaledMixture], Array],
        num_realizations: int = 5,
        error_size: int = 2_000,
        save_raw_injections: bool = False,
        save_raw_PEs: bool = False,
        save_weighted_injections: bool = False,
        save_weighted_PEs: bool = False,
        save_ref_probs: bool = False,
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

        self.save_raw_injections = save_raw_injections
        self.save_raw_PEs = save_raw_PEs
        self.save_ref_probs = save_ref_probs
        self.save_weighted_injections = save_weighted_injections
        self.save_weighted_PEs = save_weighted_PEs

        if not any(
            [
                save_raw_injections,
                save_raw_PEs,
                save_weighted_injections,
                save_weighted_PEs,
            ]
        ):
            raise ValueError(
                "At least one of the following should be true.\n"
                " - save_raw_injections\n"
                " - save_raw_PEs\n"
                " - save_weighted_injections\n"
                " - save_weighted_PEs"
            )

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

    def _generate_realizations(self, key: PRNGKeyArray) -> None:
        r"""Generate realizations for the population."""
        poisson_key, rate_key = jrd.split(key)
        exp_rate = self.ERate_fn(self.model)
        size = int(jrd.poisson(poisson_key, exp_rate))
        key = rate_key
        if size == 0:
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
        scale = int(self.save_raw_injections) + int(self.save_weighted_injections)

        with get_progress_bar("Injections", self.verbose) as progress:
            realization_task = progress.add_task(
                "Generating realizations", total=self.num_realizations * scale
            )
            for i in range(self.num_realizations):
                realizations_path = os.path.join(
                    self.root_dir, self.realizations_dir.format(i)
                )
                os.makedirs(realizations_path, exist_ok=True)

                if self.save_raw_injections:
                    raw_population, [raw_indices] = (
                        self.model.sample_with_intermediates(pop_keys[i], (size,))
                    )
                    raw_injections_file_path = os.path.join(
                        realizations_path, self.injection_filename.format("raw")
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
                    progress.advance(realization_task, 1)

                if self.save_weighted_injections:
                    old_size = size
                    if self.logVT_fn is not None:
                        size += int(1e4)

                    population, [indices] = self.model.sample_with_intermediates(
                        pop_keys[i], (size,)
                    )

                    raw_population = population
                    raw_indices = indices

                    if self.logVT_fn is not None:
                        _, key = jrd.split(key)

                        vt = jnn.softmax(self.logVT_fn(population))
                        _, key = jrd.split(key)
                        index = jrd.choice(
                            key,
                            jnp.arange(population.shape[0]),
                            p=vt,
                            shape=(old_size,),
                        )

                        population = population[index]
                        indices = indices[index]

                    if population.shape == ():
                        warnings.warn(
                            f"Population size is zero for realization {i}. Skipping this realization.",
                            category=UserWarning,
                        )
                    else:
                        injections_file_path = os.path.join(
                            realizations_path,
                            self.injection_filename.format("weighted"),
                        )
                        color_indices_file_path = os.path.join(
                            realizations_path, "color.dat"
                        )
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
                    progress.advance(realization_task, 1)

    def _add_error_to_an_injection(
        self, *, key: PRNGKeyArray, data: Array, size: int
    ) -> Array:
        """Add error to an injection.

        Parameters
        ----------
        key : PRNGKeyArray
            Pseudo-random number generator key for reproducibility.
        data : Array
            Data (injection) to which error will be added.
        size : int
            Size of the error to add.

        Returns
        -------
        Array
            Noisy data with added error.
        """
        keys = jrd.split(key, len(self.heads))
        noisy_data_collection = []

        for i, (head, err_fn) in enumerate(zip(self.heads, self.error_fns)):
            noisy_data_i: Array = err_fn(data[head], size, keys[i])
            if noisy_data_i.ndim == 1:
                noisy_data_i = noisy_data_i.reshape(size, -1)
            noisy_data_collection.append(noisy_data_i)

        noisy_data = jnp.column_stack(noisy_data_collection)
        nan_mask = np.isnan(noisy_data).any(axis=1)
        noisy_data = noisy_data[~nan_mask]
        return noisy_data

    def _add_error(
        self,
        *,
        n_realizations: int,
        extra_size: int,
        key: PRNGKeyArray,
    ) -> None:
        r"""Adds error to the realizations' population."""

        N_heads = len(self.heads)
        assert N_heads > 0

        realizations_path = os.path.join(
            self.root_dir, self.realizations_dir.format(n_realizations)
        )

        if self.save_raw_PEs:
            error_dir = self.error_dir.format("raw")
            posterior_path = os.path.join(
                realizations_path,
                error_dir,
                self.event_filename,
            )

            os.makedirs(os.path.join(realizations_path, error_dir), exist_ok=True)

            injection_path = os.path.join(
                realizations_path, self.injection_filename.format("raw")
            )
            data_inj = np.loadtxt(injection_path, skiprows=1)

            keys = jrd.split(key, data_inj.shape[0] * 2)

            for index in range(data_inj.shape[0]):
                data = data_inj[index]
                noisy_data = self._add_error_to_an_injection(
                    key=keys[index], data=data, size=self.error_size
                )
                count = np.count_nonzero(noisy_data)
                if count < 1:
                    warnings.warn(
                        f"Skipping file {index} due to all NaN values or insufficient data.",
                        category=UserWarning,
                    )
                    continue
                np.savetxt(
                    posterior_path.format(index),
                    noisy_data,
                    header=" ".join(self.parameters),
                    comments="",  # To remove the default comment character '#'
                )

        if self.save_weighted_PEs:
            error_dir = self.error_dir.format("weighted")
            posterior_path = os.path.join(
                realizations_path,
                error_dir,
                self.event_filename,
            )

            os.makedirs(os.path.join(realizations_path, error_dir), exist_ok=True)

            injection_path = os.path.join(
                realizations_path, self.injection_filename.format("weighted")
            )
            data_inj = np.loadtxt(injection_path, skiprows=1)

            error_size_before_selection = self.error_size + extra_size
            for index in range(data_inj.shape[0]):
                data = data_inj[index]
                noisy_data = self._add_error_to_an_injection(
                    key=keys[index],
                    data=data,
                    size=error_size_before_selection,
                )
                weights = np.array(jnn.softmax(self.model.log_prob(noisy_data)))
                noisy_data = jrd.choice(  # type: ignore
                    key=keys[data_inj.shape[0] + index],
                    a=noisy_data,
                    shape=(self.error_size,),
                    p=weights,
                )
                if noisy_data.shape[0] < self.error_size:
                    warnings.warn(
                        f"Insufficient data for file {index}. Expected {self.error_size}, got {noisy_data.shape[0]}.",
                        category=UserWarning,
                    )
                count = np.count_nonzero(noisy_data)
                if count < 1:
                    warnings.warn(
                        f"Skipping file {index} due to all NaN values or insufficient data.",
                        category=UserWarning,
                    )
                    continue
                np.savetxt(
                    posterior_path.format(index),
                    noisy_data,
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

        heads = []
        error_fns = []

        for head, err_fn in error_magazine.registry.items():
            _head = []
            for h in head:
                i = self.parameters.index(h)
                _head.append(i)
            heads.append(_head)
            error_fns.append(err_fn)

        self.heads = heads
        self.error_fns = error_fns

        del heads  # deleting heads, use self.heads instead
        del error_fns  # deleting error_fns, use self.error_fns instead

        keys = jrd.split(key, self.num_realizations)

        with get_progress_bar("Errors", self.verbose) as progress:
            scale = int(self.save_raw_PEs) + int(self.save_weighted_PEs)
            adding_error_task = progress.add_task(
                "Errors", total=self.num_realizations * scale
            )
            for n_realizations in range(self.num_realizations):
                self._add_error(
                    n_realizations=n_realizations,
                    extra_size=10_000,
                    key=keys[i],
                )
                progress.advance(adding_error_task, scale)
