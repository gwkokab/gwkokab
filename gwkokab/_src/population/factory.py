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
from typing_extensions import Callable, Optional, Self

import numpy as np
import numpyro
from jax import numpy as jnp, random as jrd, tree as jtr
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpyro.distributions import *
from numpyro.distributions.constraints import *
from numpyro.util import is_prng_key

from ..models import *
from ..models.utils.constraints import *
from ..models.utils.jointdistribution import JointDistribution
from .aliases import PopInfo


__all__ = ["PopulationFactory"]


class PopulationFactory:
    INJECTIONS_DIR: str = "injections"
    REALIZATIONS_DIR: str = "realization_{}"

    def __init__(
        self,
        models: dict[tuple[str, ...], numpyro.distributions.Distribution],
        popinfo: PopInfo,
        seperate_injections: Optional[bool] = None,
        constraint: Optional[Callable[[Array], Bool]] = None,
    ) -> None:
        if seperate_injections is None:
            self.seperate_injections = False
        self.seperate_injections = seperate_injections

        if constraint is None:
            constraint = lambda x: jnp.ones(x.shape[0], dtype=bool)
        self.constraint = constraint

        self.models = list(models.values())

        headers: list[str] = []
        for output_var in models.keys():
            headers.extend(output_var)
        self.headers = headers

        self.vt_param_dist = None
        self.vt_selection_mask = None

        if popinfo.LOG_VT is not None:
            vt_models: list[Distribution] = []
            vt_selection_mask: list[int] = []
            vt_params = list(popinfo.VT_PARAMS)

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

        self.popinfo = popinfo

    def exp_rate(self: Self, *, key: PRNGKeyArray) -> Float:
        N = int(1e4)
        value = self.vt_param_dist.sample(key, (N,))[..., self.vt_selection_mask]
        return (
            self.popinfo.TIME
            * self.popinfo.RATE
            * jnp.mean(jnp.exp(self.popinfo.LOG_VT(value).flatten()))
        )

    def generate_population(self, size: Int, *, key: PRNGKeyArray) -> Array:
        keys = list(jrd.split(key, len(self.models)))
        if self.popinfo.LOG_VT is not None:
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

        if self.popinfo.LOG_VT is None:
            return population

        value = jnp.column_stack(
            [
                population[:, self.headers.index(vt_params)]
                for vt_params in self.popinfo.VT_PARAMS
            ]
        )

        vt = jnp.exp(self.popinfo.LOG_VT(value).flatten())
        vt = jnp.nan_to_num(vt, nan=0.0)
        vt /= jnp.sum(vt)
        _, key = jrd.split(keys[-1])
        index = jrd.choice(
            key, jnp.arange(population.shape[0]), p=vt, shape=(old_size,)
        )

        population = population[index]

        return population

    def generate_realizations(self, key: Optional[PRNGKeyArray] = None) -> None:
        if key is None:
            key = jrd.PRNGKey(np.random.randint(0, 2**32 - 1))
        assert is_prng_key(key)
        size = self.popinfo.RATE
        if self.popinfo.LOG_VT is not None:
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
        pop_keys = jrd.split(key, self.popinfo.NUM_REALIZATIONS)
        os.makedirs(self.popinfo.ROOT_DIR, exist_ok=True)
        for i in range(self.popinfo.NUM_REALIZATIONS):
            population = self.generate_population(size, key=pop_keys[i])

            if population.shape == ():
                continue

            realizations_path = os.path.join(
                self.popinfo.ROOT_DIR, self.REALIZATIONS_DIR.format(i)
            )
            os.makedirs(realizations_path, exist_ok=True)
            injection_filename = os.path.join(realizations_path, "injections.dat")
            np.savetxt(
                injection_filename,
                population,
                comments="#",
                header=" ".join(self.headers),
            )

            if self.seperate_injections:
                injection_path = os.path.join(realizations_path, self.INJECTIONS_DIR)
                os.makedirs(injection_path, exist_ok=True)
                for j in range(population.shape[0]):
                    injection_filename = os.path.join(
                        injection_path,
                        self.popinfo.EVENT_FILENAME.format(j) + ".dat",
                    )
                    np.savetxt(
                        injection_filename,
                        population[j].reshape(1, -1),
                        comments="#",
                        header=" ".join(self.headers),
                    )
