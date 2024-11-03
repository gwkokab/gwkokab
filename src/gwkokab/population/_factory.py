# Copyright 2023 The GWKokab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import List, Optional, Tuple

import numpy as np
from jax import numpy as jnp, random as jrd
from jax.nn import softmax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpyro.util import is_prng_key

from ..models.utils import ScaledMixture
from ..models.wrappers import ModelRegistry
from ._utils import ensure_dat_extension, get_progress_bar


PROGRESS_BAR_TEXT_WITDH = 25

error_magazine = ModelRegistry()


class PopulationFactory:
    r"""Class with methods equipped to generate population for each realizations and
    adding errors in it."""

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
        analysis_time: float,
        logVT_fn: Callable[[Array], Array],
        vt_params: List[str],
        scale_factor: float = 1.0,
        num_realizations: Int[int, "..."] = 5,
        error_size: Int[int, ">0"] = 2_000,
        constraint: Callable[[Array], Bool[Array, "..."]] = lambda x: jnp.ones(
            x.shape[0], dtype=bool
        ),
    ) -> None:
        r"""Check if the parameters are provided."""
        self.model = model
        self.parameters = parameters
        self.analysis_time = analysis_time
        self.logVT_fn = logVT_fn
        self.vt_params = vt_params
        self.scale_factor = scale_factor
        self.num_realizations = num_realizations
        self.error_size = error_size
        self.constraint = constraint

        self.event_filename = ensure_dat_extension(self.event_filename)
        self.injection_filename = ensure_dat_extension(self.injection_filename)

        if self.logVT_fn is None:
            raise ValueError("`logVT_fn` is not provided.")

        if self.model is None:
            raise ValueError("Model is not provided.")
        if not isinstance(self.model, ScaledMixture):
            raise ValueError(
                "The model must be a `ScaledMixture` model for multi-rate model."
                "See `gwkokab.model.utils.ScaledMixture` for more details."
            )

        if self.parameters == []:
            raise ValueError("Parameters are not provided.")
        if self.vt_params == []:
            raise ValueError("VT Parameters are not provided.")

        self.vt_selection_mask = []
        for param in self.vt_params:
            if param in self.parameters:
                self.vt_selection_mask.append(self.parameters.index(param))
            else:
                raise ValueError(f"VT parameter '{param}' is not found in parameters.")

    def exp_rate(self, *, key: PRNGKeyArray) -> Float[Array, ""]:
        r"""Calculates the expected rate."""
        N = int(5e4)
        value = self.model.sample(key, (N,))[..., self.vt_selection_mask]
        sum_of_rates = jnp.sum(jnp.exp(self.model._log_scales))
        return (
            self.analysis_time
            * self.scale_factor
            * sum_of_rates
            * jnp.mean(jnp.exp(self.logVT_fn(value).flatten()))
        )

    def _generate_population(
        self, size: int, *, key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        r"""Generate population for a realization."""

        if self.logVT_fn is not None:
            old_size = size
            size += int(1e5)

        population, [indices] = self.model.sample_with_intermediates(key, (size,))
        constraints = self.constraint(population)
        population = population[constraints]
        indices = indices[constraints]

        _, key = jrd.split(key)

        value = population[..., self.vt_selection_mask]

        vt = softmax(self.logVT_fn(value).flatten())
        vt = jnp.nan_to_num(vt, nan=0.0)
        _, key = jrd.split(key)
        index = jrd.choice(
            key, jnp.arange(population.shape[0]), p=vt, shape=(old_size,)
        )

        population = population[index]
        indices = indices[index]

        return population, indices

    def _generate_realizations(self, key: PRNGKeyArray) -> None:
        r"""Generate realizations for the population."""
        poisson_key, rate_key = jrd.split(key)
        size = int(jrd.poisson(poisson_key, self.exp_rate(key=rate_key)))
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
        with get_progress_bar("Injections", self.verbose) as progress:
            realization_task = progress.add_task(
                "Generating realizations", total=self.num_realizations
            )
            for i in range(self.num_realizations):
                population, indices = self._generate_population(size, key=pop_keys[i])

                if population.shape == ():
                    continue

                realizations_path = os.path.join(
                    self.root_dir, self.realizations_dir.format(i)
                )
                os.makedirs(realizations_path, exist_ok=True)
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
                progress.advance(realization_task, 1)

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
        keys = jrd.split(key, data_inj.shape[0] * len(heads))

        for index in range(data_inj.shape[0]):
            noisey_data = np.empty((self.error_size, len(self.parameters)))
            data = data_inj[index]
            i = 0
            for head, err_fn in zip(heads, error_fns):
                noisey_data_i: Array = err_fn(
                    data[head], self.error_size, keys[index + i]
                )
                if noisey_data_i.ndim == 1:
                    noisey_data_i = noisey_data_i.reshape(self.error_size, -1)
                noisey_data[:, head] = noisey_data_i
                i += 1
            nan_mask = np.isnan(noisey_data).any(axis=1)
            masked_noisey_data = noisey_data[~nan_mask]
            count = np.count_nonzero(masked_noisey_data)
            if count < 2:
                warnings.warn(
                    f"Skipping file {index} due to all NaN values or insufficient data.",
                    category=UserWarning,
                )
                continue
            np.savetxt(
                output_dir.format(index),
                masked_noisey_data,
                header=" ".join(self.parameters),
                comments="",  # To remove the default comment character '#'
            )

    def produce(self, key: Optional[PRNGKeyArray] = None) -> None:
        r"""Generate realizations and add errors to the populations."""
        if key is None:
            key = jrd.PRNGKey(np.random.randint(0, 2**32 - 1))
        else:
            assert is_prng_key(key)
        self._generate_realizations(key)
        keys = jrd.split(key, self.num_realizations)
        with get_progress_bar("Errors", self.verbose) as progress:
            adding_error_task = progress.add_task("Errors", total=self.num_realizations)
            for i in range(self.num_realizations):
                self._add_error(i, key=keys[i])
                progress.advance(adding_error_task, 1)
