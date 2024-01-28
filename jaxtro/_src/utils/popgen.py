#  Copyright 2023 The Jaxtro Authors
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
from typing_extensions import Any

import numpy as np
from jax import numpy as jnp
from jaxtyping import Array
from tqdm import tqdm

from ..models import *
from .misc import add_normal_error, dump_configurations


class PopulationGenerator:
    """Class to generate population and save them to disk."""

    def __init__(self, general: dict, models: list[dict]) -> None:
        """__init__ method for PopulationGenerator.

        Parameters
        ----------
        config : dict
            Configuration dictionary for PopulationGenerator.
        """
        self.check_general(general)
        for model in models:
            self.check_models(model)

        self._size: int = general["size"]
        self._error_scale: float = general["error_scale"]
        self._error_size: int = general["error_size"]
        self._root_container: str = general["root_container"]
        self._event_filename: str = general["event_filename"]
        self._config_filename: str = general["config_filename"]
        self._save_injections: bool = general["save_injections"]
        self._num_realizations: int = general["num_realizations"]
        self._models: list = models

    @staticmethod
    def check_general(general: dict) -> None:
        """Check if all the required configs are present."""
        assert general.get("size", None) is not None
        assert general.get("error_scale", None) is not None
        assert general.get("error_size", None) is not None
        assert general.get("root_container", None) is not None
        assert general.get("event_filename", None) is not None
        assert general.get("config_filename", None) is not None

    @staticmethod
    def check_models(model: dict) -> None:
        """Check if all the required configs are present."""
        assert model.get("model", None) is not None
        assert model.get("config_vars", None) is not None
        assert model.get("col_names", None) is not None
        assert model.get("params", None) is not None

    def model_sampler(self) -> tuple[list[str], list[tuple[Any, Any]], Array]:
        config_vals = []
        col_names = []
        realisations = jnp.empty((self._size, 0))
        for model in self._models:
            model_instance = eval(model["model"])(**model["params"])
            rvs = model_instance.samples(self._size).reshape((self._size, -1))
            realisations = jnp.concatenate((realisations, rvs), axis=-1)

            config_vals.extend([(x, model["params"][x]) for x in model["config_vars"]])
            col_names.extend(model["col_names"])
        return col_names, config_vals, realisations

    def add_error(self, col_names: list[str], container: str, realisations: Array, index: int) -> None:
        for event_num, realisation in enumerate(realisations):
            filename = f"{container}/{self._event_filename.format(event_num)}"

            realisation_err = add_normal_error(
                *realisation,
                scale=self._error_scale,
                size=self._error_size,
            )

            np.savetxt(
                filename,
                realisation_err,
                header="\t".join(col_names),
            )

    def generate(self):
        """Generate population and save them to disk."""
        os.makedirs(self._root_container, exist_ok=True)

        for i in tqdm(
            range(self._num_realizations),
            desc="Realizations",
            total=self._num_realizations,
            unit=" realization",
            unit_scale=True,
        ):
            container = f"{self._root_container}/realization_{i}/"
            config_filename = f"{container}/{self._config_filename}"
            injection_filename = f"{container}/injections.dat"

            os.makedirs(container, exist_ok=True)

            col_names, config_vals, realisations = self.model_sampler()

            dump_configurations(config_filename, *config_vals)

            if self._save_injections:
                np.savetxt(injection_filename, realisations, header="\t".join(col_names))

            self.add_error(col_names, container, realisations, i)
