#  Copyright 2023 The GWKokab Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import os
import warnings
from typing_extensions import Callable, List, Optional, Self

import numpy as np
from jax import numpy as jnp, random as jrd, tree as jtr
from jax.nn import softmax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpyro.distributions import (
    CategoricalProbs,
    Distribution,
    MixtureGeneral,
)
from numpyro.util import is_prng_key
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..models.utils import JointDistribution
from ..models.wrappers import ModelRegistry


__all__ = ["popfactory", "popmodel_magazine", "error_magazine"]

PROGRESS_BAR_TEXT_WITDH = 25

popmodel_magazine = ModelRegistry()
error_magazine = ModelRegistry()


class PopulationFactory:
    r"""Class with methods equipped to generate population for each
    realizations and adding errors in it."""

    injection_filename: str = "injections"
    realizations_dir: str = "realization_{}"
    error_dir: str = "posteriors"
    root_dir: str = "data"
    event_filename: str = "event_{}"
    num_realizations: Int = 5
    error_size: Int = 2_000
    constraint: Callable[[Array], Bool] = lambda x: jnp.ones(x.shape[0], dtype=bool)
    rate: Optional[Float | List[Float]] = None
    analysis_time: Optional[Float] = None
    log_VT_fn: Optional[Callable[[Array], Array]] = None
    VT_params: Optional[list[str]] = None
    verbose: bool = True

    def check_params(self) -> None:
        r"""Check if the parameters are provided."""
        assert self.rate is None, "RATE is not provided."
        assert self.analysis_time is None, "ANALYSIS_TIME is not provided."
        assert self.log_VT_fn is None, "LOG_VT is not provided."
        assert self.VT_params is None, "VT_PARAMS is not provided."

        if not self.event_filename.endswith(".dat"):
            self.event_filename += ".dat"
        else:
            raise ValueError("Event filename should have .dat extension.")

        if not self.injection_filename.endswith(".dat"):
            self.injection_filename += ".dat"
        else:
            raise ValueError("Injection filename should have .dat extension.")

    def __init__(self) -> None:
        self.check_params()

    def pre_process(self) -> None:
        """Pre processes the data for the generation of population."""
        models = popmodel_magazine.registry
        models_values = list(models.values())
        self.is_multi_rate_model = (
            isinstance(self.rate, list)
            and len(self.rate) > 1
            and len(models.values()) == 1
        )

        if self.is_multi_rate_model:
            assert isinstance(
                models_values[0], MixtureGeneral
            ), "Model must be a mixture model for multi-rate models."
            assert len(models_values[0].component_distributions) == len(
                self.rate
            ), "Number of components in the model must be equal to the number of rates."

        if self.is_multi_rate_model:
            self.model = models_values[0]
        else:
            self.model = JointDistribution(*models_values)

        headers: list[str] = []
        for output_var in models.keys():
            headers.extend(output_var)
        self.headers = headers

        self.vt_selection_mask = [headers.index(param) for param in self.VT_params]

    def exp_rate(
        self: Self, *, key: PRNGKeyArray, rate: Float, model: Distribution
    ) -> Float:
        r"""Calculates the expected rate."""
        N = int(1e4)
        value = model.sample(key, (N,))[..., self.vt_selection_mask]
        return (
            self.analysis_time
            * rate
            * jnp.mean(jnp.exp(self.log_VT_fn(value).flatten()))
        )

    def _generate_population(self, size: Int, *, key: PRNGKeyArray) -> Array:
        r"""Generate population for a realization."""

        if self.log_VT_fn is not None:
            old_size = size
            size += int(1e5)

        population = self.model.sample(key, (size,))
        population = population[self.constraint(population)]

        if self.log_VT_fn is None:
            return population

        _, key = jrd.split(key)

        value = population[..., self.vt_selection_mask]

        vt = softmax(self.log_VT_fn(value).flatten())
        vt = jnp.nan_to_num(vt, nan=0.0)
        _, key = jrd.split(key)
        index = jrd.choice(
            key, jnp.arange(population.shape[0]), p=vt, shape=(old_size,)
        )

        population = population[index]

        return population

    def _generate_realizations(self, key: PRNGKeyArray) -> None:
        r"""Generate realizations for the population."""
        size = self.rate
        self.pre_process()
        if self.log_VT_fn is not None:
            if self.is_multi_rate_model:
                total_rate = sum(self.rate)
                rate_list = [rate / total_rate for rate in self.rate]
                rate = jnp.array(rate_list)
                self.model._mixing_distribution = CategoricalProbs(probs=rate)
                keys = list(jrd.split(key, len(self.rate)))
                exp_rates = jtr.map(
                    lambda model, k, rate: self.exp_rate(key=k, rate=rate, model=model),
                    self.model.component_distributions,
                    keys,
                    self.rate,
                    is_leaf=lambda x: isinstance(x, Distribution),
                )
                _, key = jrd.split(keys[-1])
                size = int(jrd.poisson(key, sum(exp_rates)))
            else:
                poisson_key, rate_key = jrd.split(key)
                size: Int = int(
                    jrd.poisson(
                        poisson_key,
                        self.exp_rate(key=rate_key, rate=self.rate, model=self.model),
                    )
                )
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
        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]Injections".ljust(PROGRESS_BAR_TEXT_WITDH),
                justify="left",
            ),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.2f}%",
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            "•",
            MofNCompleteColumn(separator=" realizations out of "),
            disable=not self.verbose,
        ) as progress:
            realization_task = progress.add_task(
                "Generating realizations", total=self.num_realizations
            )
            for i in range(self.num_realizations):
                population = self._generate_population(size, key=pop_keys[i])

                if population.shape == ():
                    continue

                realizations_path = os.path.join(
                    self.root_dir, self.realizations_dir.format(i)
                )
                os.makedirs(realizations_path, exist_ok=True)
                injections_file_path = os.path.join(
                    realizations_path, self.injection_filename
                )
                np.savetxt(
                    injections_file_path,
                    population,
                    comments="#",
                    header=" ".join(self.headers),
                )
                progress.advance(realization_task, 1)

    def _add_error(self, realization_number, *, key: PRNGKeyArray) -> None:
        r"""Adds error to the realizations' population."""
        realizations_path = os.path.join(
            self.root_dir, self.realizations_dir.format(realization_number)
        )

        heads: list[list[int]] = []
        error_fns: list[Callable] = []

        for head, err_fn in error_magazine.registry.items():
            _head = []
            for h in head:
                i = self.headers.index(h)
                _head.append(i)
            heads.append(_head)
            error_fns.append(err_fn)

        output_dir = os.path.join(
            realizations_path, self.error_dir, self.event_filename
        )

        os.makedirs(os.path.join(realizations_path, self.error_dir), exist_ok=True)

        injections_file_path = os.path.join(realizations_path, self.injection_filename)
        data_inj = np.loadtxt(injections_file_path)
        keys = jrd.split(key, data_inj.shape[0] * len(heads))

        for index in range(data_inj.shape[0]):
            noisey_data = np.empty((self.error_size, len(self.headers)))
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
                header=" ".join(self.headers),
                comments="#",
            )

    def produce(self, key: Optional[PRNGKeyArray] = None) -> None:
        r"""Generate realizations and add errors to the populations."""
        if key is None:
            key = jrd.PRNGKey(np.random.randint(0, 2**32 - 1))
        else:
            assert is_prng_key(key)
        self._generate_realizations(key)
        keys = jrd.split(key, self.num_realizations)
        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]Errors".ljust(PROGRESS_BAR_TEXT_WITDH),
                justify="left",
            ),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.2f}%",
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            "•",
            MofNCompleteColumn(separator=" realizations out of "),
            disable=not self.verbose,
        ) as progress:
            adding_error_task = progress.add_task("Errors", total=self.num_realizations)
            for i in range(self.num_realizations):
                self._add_error(i, key=keys[i])
                progress.advance(adding_error_task, 1)


popfactory = PopulationFactory()
