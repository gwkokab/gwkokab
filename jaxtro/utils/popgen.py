# Copyright 2023 The Jaxtro Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import numpy as np
from jaxampler.rvs import ContinuousRV
from tqdm import tqdm

from .misc import add_normal_error, dump_configurations


class PopulationGenerator:
    """Class to generate population and save them to disk."""

    def __init__(self, config: dict) -> None:
        """__init__ method for PopulationGenerator.

        Parameters
        ----------
        config : dict
            Configuration dictionary for PopulationGenerator.
        """
        self._model: ContinuousRV = config.get("model", None)
        self._size: int = config.get("size", None)
        self._error_scale: float = config.get("error_scale", None)
        self._error_size: int = config.get("error_size", None)
        self._root_container: str = config.get("root_container", None)
        self._event_filename: str = config.get("event_filename", None)
        self._config_filename: str = config.get("config_filename", None)
        self._config_vars: list[str] = config.get("config_vars", None)
        self._col_names: list[str] = config.get("col_names", None)
        self._params: list[dict[str, Any]] = config.get("params", None)
        self.check_configs()

    def check_configs(self) -> None:
        """Check if all the required configs are present."""
        assert self._model is not None
        assert self._size is not None
        assert self._error_scale is not None
        assert self._error_size is not None
        assert self._root_container is not None
        assert self._event_filename is not None
        assert self._config_filename is not None
        assert self._config_vars is not None
        assert self._col_names is not None
        assert self._params is not None

    def generate(self):
        """Generate population and save them to disk."""
        os.makedirs(self._root_container, exist_ok=True)

        container = f"{self._root_container}"

        os.makedirs(container, exist_ok=True)

        model_instance: ContinuousRV = self._model(**self._params)
        realisations = model_instance.rvs(self._size)

        dump_configurations(
            f"{container}/{self._config_filename}",
            *list(map(lambda x: (x, self._params[x]), self._config_vars)),
        )

        for event_num, realisation in tqdm(enumerate(realisations),
                                           desc=f"Generating events",
                                           total=self._size,
                                           unit=" events",
                                           unit_scale=True):

            filename = f"{container}/{self._event_filename.format(event_num)}"

            realisation_err = add_normal_error(
                *realisation,
                scale=self._error_scale,
                size=self._error_size,
            )

            np.savetxt(
                filename,
                realisation_err,
                header="\t".join(self._col_names),
            )
