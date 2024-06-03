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
from typing_extensions import Self

import jax
from jax import numpy as jnp, random as jrd, tree as jtr
from jaxtyping import Float, Int
from numpyro import distributions as dist
from numpyro.distributions import *
from numpyro.distributions.constraints import *

from ..models import *
from ..models.utils.constraints import *
from ..models.utils.jointdistribution import JointDistribution
from ..utils import get_key
from ..vts.neuralvt import load_model  # imported here to avoid circular import
from .aliases import ModelMeta, Parameter, PopInfo


class PopulationFactory:
    def __init__(self, models: list[dict], popinfo: PopInfo) -> None:
        for model in models:
            self._check_model(model)

        self.models: list[dist.Distribution] = [
            model[ModelMeta.NAME](**model[ModelMeta.PARAMETERS]) for model in models
        ]

        VT_selection_flag = popinfo.VT_FILE is not None

        headers: list[Parameter] = []
        if VT_selection_flag:
            no_vt_param = len(popinfo.VT_PARAMS)
            vt_models_index: list[int] = [None] * no_vt_param
            vt_params = list(popinfo.VT_PARAMS.keys())

        for i, model in enumerate(models):
            output = model[ModelMeta.OUTPUT]
            headers.extend(output)
            if VT_selection_flag:
                for out in output:
                    if out in popinfo.VT_PARAMS:
                        index = vt_params.index(out)
                        vt_models_index[index] = i
        self.vt_param_dist = JointDistribution(*list(map(lambda x: popinfo.VT_PARAMS.get(x), vt_params)))

        self.headers = [header.value for header in headers]
        self.popinfo = popinfo

    @staticmethod
    def _check_model(model) -> None:
        if ModelMeta.NAME not in model:
            raise ValueError("Model must have a name")
        if ModelMeta.OUTPUT not in model:
            raise ValueError("Model must have an output")
        if ModelMeta.PARAMETERS not in model:
            raise ValueError("Model must have parameters")
        if ModelMeta.SAVE_AS not in model:
            warnings.warn(
                message=f"{model[ModelMeta.NAME].__name__} does not have a save_as. Parameters will not be saved."
            )

    def exp_rate(self: Self) -> Float:
        N = int(1e4)
        value = self.vt_param_dist.sample(get_key(), (N,))
        # TODO: Complete the mechanism for the models to be selected for rate calculation
        model = JointDistribution(
            *(self.get_model_instance(self._models_dict[sm], update_config_vars=False) for sm in self._selection_models)
        )
        volume = jnp.prod(jnp.max(value, axis=0) - jnp.min(value, axis=0))
        _, logVT = load_model(self.popinfo.VT_FILE)
        logVT = jax.vmap(logVT)
        return self.popinfo.RATE * volume * jnp.mean(jnp.exp(model.log_prob(value) + logVT(value).flatten()))

    def generate_population(self, size: Int) -> None:
        keys = list(jrd.split(get_key(), len(self.models)))
        if self.popinfo.VT_FILE is not None:
            old_size = size
            size += int(5e4)
        population = jtr.map(
            lambda model, key: model.sample(key, (size,)).reshape(size, -1),
            self.models,
            keys,
            is_leaf=lambda x: isinstance(x, dist.Distribution),
        )
        population = jtr.reduce(lambda x, y: jnp.concatenate((x, y), axis=-1), population)
        if self.popinfo.VT_FILE is not None:
            m1 = population[:, self.headers.index(Parameter.PRIMARY_MASS.value)]
            m2 = population[:, self.headers.index(Parameter.SECONDARY_MASS.value)]
            value = jnp.column_stack((m1, m2))

            _, logVT = load_model(self.popinfo.VT_FILE)
            logVT = jax.vmap(logVT)

            vt = jnp.exp(logVT(value).flatten())
            vt = vt / jnp.sum(vt)

            index = jrd.choice(get_key(), jnp.arange(size), p=vt, shape=(old_size,))

            population = population[index]

        return population

    def generate_realizations(self) -> None:
        # size: Int = jrd.poisson(get_key(), self.exp_rate())
        size = 100
        os.makedirs(self.popinfo.ROOT_DIR, exist_ok=True)
        for i in range(self.popinfo.NUM_REALIZATIONS):
            population = self.generate_population(size)
            print(population)
