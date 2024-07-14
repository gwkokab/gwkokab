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

import glob
import os
import warnings
from typing_extensions import Callable, Optional, Self

import numpy as np
from jax import numpy as jnp, random as jrd, tree as jtr
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpyro.distributions import *
from numpyro.distributions.constraints import *
from numpyro.util import is_prng_key

from ..models import *
from ..models.utils.constraints import *
from ..models.utils.jointdistribution import JointDistribution
from ..models.utils.wrappers import ModelRegistry


__all__ = ["popfactory", "popmodel_magazine", "error_magazine"]

popmodel_magazine = ModelRegistry()
error_magazine = ModelRegistry()


class PopulationFactory:
    r"""Class with methods equipped to generate population for each
    realizations and adding errors in it."""

    injections_dir: str = "injections"
    realizations_dir: str = "realization_{}"
    error_dir: str = "posteriors"
    root_dir: str = "data"
    event_filename: str = "event_{}"
    num_realizations: Int = 5
    error_size: Int = 2_000
    constraint: Callable[[Array], Bool] = lambda x: jnp.ones(x.shape[0], dtype=bool)
    rate: Optional[Float] = None
    analysis_time: Optional[Float] = None
    log_VT_fn: Optional[Callable[[Array], Array]] = None
    VT_params: Optional[list[str]] = None

    def check_params(self) -> None:
        r"""Check if the parameters are provided."""
        assert self.rate is None, "RATE is not provided."
        assert self.analysis_time is None, "ANALYSIS_TIME is not provided."
        assert self.log_VT_fn is None, "LOG_VT is not provided."
        assert self.VT_params is None, "VT_PARAMS is not provided."

    def __init__(self) -> None:
        self.check_params()

    def pre_process(self) -> None:
        """Pre processes the data for the generation of population."""
        models = popmodel_magazine.registry
        self.models = list(models.values())

        headers: list[str] = []
        for output_var in models.keys():
            headers.extend(output_var)
        self.headers = headers

        self.vt_param_dist = None
        self.vt_selection_mask = None

        if self.log_VT_fn is not None:
            vt_models: list[Distribution] = []
            vt_selection_mask: list[int] = []
            vt_params = list(self.VT_params)

            k = 0
            for i, output_var in enumerate(models.keys()):
                flag = False
                for out in output_var:
                    if out in vt_params:
                        index = output_var.index(out) + k
                        vt_selection_mask.append(index)
                        vt_models.append(self.models[i])
                        flag = True
                if flag:
                    k += len(output_var)

            self.vt_param_dist = JointDistribution(*vt_models)
            self.vt_selection_mask = vt_selection_mask

    def exp_rate(self: Self, *, key: PRNGKeyArray) -> Float:
        r"""Calculates the expected rate."""
        N = int(1e4)
        value = self.vt_param_dist.sample(key, (N,))[..., self.vt_selection_mask]
        return (
            self.analysis_time
            * self.rate
            * jnp.mean(jnp.exp(self.log_VT_fn(value).flatten()))
        )

    def _generate_population(self, size: Int, *, key: PRNGKeyArray) -> Array:
        r"""Generate population for a realization."""
        keys = list(jrd.split(key, len(self.models)))
        if self.log_VT_fn is not None:
            old_size = size
            size += int(1e5)
        population = jtr.map(
            lambda model, key: model.sample(key, (size,)).reshape(size, -1),
            self.models,
            keys,
            is_leaf=lambda x: isinstance(x, Distribution),
        )

        population = jtr.reduce(
            lambda x, y: jnp.concatenate((x, y), axis=-1), population
        )

        population = population[self.constraint(population)]

        if self.log_VT_fn is None:
            return population

        value = jnp.column_stack(
            [
                population[:, self.headers.index(vt_params)]
                for vt_params in self.VT_params
            ]
        )

        vt = jnp.exp(self.log_VT_fn(value).flatten())
        vt = jnp.nan_to_num(vt, nan=0.0)
        vt /= jnp.sum(vt)
        _, key = jrd.split(keys[-1])
        index = jrd.choice(
            key, jnp.arange(population.shape[0]), p=vt, shape=(old_size,)
        )

        population = population[index]

        return population

    def _generate_realizations(self, key: PRNGKeyArray) -> None:
        r"""Generate realizations for the population."""
        if key is None:
            key = jrd.PRNGKey(np.random.randint(0, 2**32 - 1))
        assert is_prng_key(key)
        size = self.rate
        self.pre_process()
        if self.log_VT_fn is not None:
            poisson_key, rate_key = jrd.split(key)
            size: Int = int(jrd.poisson(poisson_key, self.exp_rate(key=rate_key)))
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
        for i in range(self.num_realizations):
            population = self._generate_population(size, key=pop_keys[i])

            if population.shape == ():
                continue

            realizations_path = os.path.join(
                self.root_dir, self.realizations_dir.format(i)
            )
            os.makedirs(realizations_path, exist_ok=True)
            injection_filename = os.path.join(realizations_path, "injections.dat")
            np.savetxt(
                injection_filename,
                population,
                comments="#",
                header=" ".join(self.headers),
            )

            injection_path = os.path.join(realizations_path, self.injections_dir)
            os.makedirs(injection_path, exist_ok=True)
            for j in range(population.shape[0]):
                injection_filename = os.path.join(
                    injection_path,
                    self.event_filename.format(j) + ".dat",
                )
                np.savetxt(
                    injection_filename,
                    population[j].reshape(1, -1),
                    comments="#",
                    header=" ".join(self.headers),
                )

    def _add_error(self, index, *, key: PRNGKeyArray) -> None:
        r"""Adds error to the realizations' population."""
        realizations_path = os.path.join(
            self.root_dir, self.realizations_dir.format(index)
        )
        injection_path = os.path.join(realizations_path, self.injections_dir)
        filenames = glob.glob(
            os.path.join(injection_path, self.event_filename.format("*") + ".dat")
        )

        output_dir = os.path.join(
            realizations_path, self.error_dir, self.event_filename
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

        index = 0

        if key is None:
            key = jrd.PRNGKey(np.random.randint(0, 2**32 - 1))

        keys = jrd.split(key, len(filenames) * len(heads))

        for filename in filenames:
            noisey_data = np.empty((self.error_size, len(self.headers)))
            data = np.loadtxt(filename)
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
            if count == 0:
                warnings.warn(
                    f"Skipping file {index} due to all NaN values",
                    category=UserWarning,
                )
                index += 1
                continue
            if masked_noisey_data.shape[0] == 1:
                masked_noisey_data = masked_noisey_data.reshape(1, -1)
            os.makedirs(os.path.dirname(output_dir.format(index)), exist_ok=True)
            np.savetxt(
                output_dir.format(index),
                masked_noisey_data,
                header=" ".join(self.headers),
                comments="#",
            )
            index += 1

    def produce(self, key: Optional[PRNGKeyArray] = None) -> None:
        r"""Generate realizations and add errors to the populations."""
        if key is None:
            key = jrd.PRNGKey(np.random.randint(0, 2**32 - 1))
        self._generate_realizations(key)
        keys = jrd.split(key, self.num_realizations)
        for i in range(self.num_realizations):
            self._add_error(i, key=keys[i])


popfactory = PopulationFactory()
