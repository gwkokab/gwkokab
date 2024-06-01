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

import warnings

from jax import random as jrd, tree as jtr
from numpyro import distributions as dist

from ..utils import get_key
from .aliases import ModelMeta, PopInfo


class PopulationFactory:
    def __init__(self, models: list[dict], popinfo: PopInfo) -> None:
        for model in models:
            self._check_model(model)

        self.models: list[dist.Distribution] = [
            model[ModelMeta.NAME](**model[ModelMeta.PARAMETERS]) for model in models
        ]
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

    def generate_population(self) -> None:
        keys = jrd.split(get_key(), len(self.models))
        population = jtr.map(
            lambda model, key: model.sample(key, self.popinfo.RATE, self.popinfo.TIME),
            self.models,
            keys,
        )

    def generate_realizations(self) -> None:
        for i in range(self.popinfo.NUM_REALIZATIONS):
            self.generate_population()
