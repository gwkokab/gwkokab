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

import glob
import os
from typing_extensions import Any

import numpy as np
from jax import numpy as jnp
from jaxtyping import Array
from tqdm import tqdm

from ..models import *
from .misc import add_normal_error, dump_configurations
from .plotting import scatter2d_batch_plot, scatter2d_plot, scatter3d_plot


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

    def _add_error(self, col_names: list[str], container: str, realisations: Array) -> None:
        os.makedirs(f"{container}/posteriors", exist_ok=True)
        os.makedirs(f"{container}/plots", exist_ok=True)
        for event_num, realisation in enumerate(realisations):
            filename_event = f"{container}/posteriors/{self._event_filename.format(event_num)}"
            filename_inj = f"{container}/injections/inj_{event_num}.dat"

            realisation_err = add_normal_error(
                *realisation,
                scale=self._error_scale,
                size=self._error_size,
            )

            np.savetxt(
                filename_inj,
                realisation.reshape((1, -1)),
                header="\t".join(col_names),
            )

            np.savetxt(
                filename_event,
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
            unit="realization",
            unit_scale=True,
        ):
            container = f"{self._root_container}/realization_{i}"
            config_filename = f"{container}/{self._config_filename}"
            injection_filename = f"{container}/injections/population.dat"

            os.makedirs(container, exist_ok=True)

            col_names, config_vals, realisations = self.model_sampler()

            dump_configurations(config_filename, *config_vals)

            os.makedirs(f"{container}/injections", exist_ok=True)
            if self._save_injections:
                np.savetxt(injection_filename, realisations, header="\t".join(col_names))

            self._add_error(col_names, container, realisations)

        event_regex = f"{self._root_container}/realization_*/posteriors/{self._event_filename.format('*')}"
        realization_regex = f"{self._root_container}/realization_*"

        for realization in tqdm(
            glob.glob(realization_regex),
            desc="Generating 2d batch plots",
            total=self._num_realizations,
            unit="event",
            unit_scale=True,
        ):
            scatter2d_batch_plot(
                file_pattern=realization + f"/posteriors/{self._event_filename.format('*')}",
                output_filename=f"{realization}/plots/mass_scatter.png",
                x_index=0,
                y_index=1,
                x_label="$m_1 [M_\odot]$",
                y_label="$m_2 [M_\odot]$",
            )
            scatter2d_batch_plot(
                file_pattern=realization + f"/posteriors/{self._event_filename.format('*')}",
                output_filename=f"{realization}/plots/spin_scatter.png",
                x_index=2,
                y_index=3,
                x_label="$a_1$",
                y_label="$a_2$",
            )

        populations = glob.glob(f"{self._root_container}/realization_*/injections/population.dat")
        for filename in tqdm(
            populations,
            desc="Generating population plots",
            total=self._num_realizations,
            unit="realization",
            unit_scale=True,
        ):
            output_filename = filename.replace("injections", "plots")
            scatter2d_plot(
                input_filename=filename,
                output_filename=output_filename.replace("population.dat", "mass_scatter_2d.png"),
                x_index=0,
                y_index=1,
                x_label="$m_1$",
                y_label="$m_2$",
            )

            scatter3d_plot(
                input_filename=filename,
                output_filename=output_filename.replace("population.dat", "mass_ecc_scatter_3d.png"),
                x_index=0,
                y_index=1,
                z_index=4,
                x_label="$m_1$",
                y_label="$m_2$",
                z_label="$\epsilon$",
            )

        posteriors = glob.glob(event_regex)
        for filename in tqdm(
            posteriors,
            desc="Generating 2d individual plots",
            total=self._num_realizations * self._size,
            unit="event",
            unit_scale=True,
        ):
            output_filename = filename.replace("posteriors", "plots")
            scatter2d_plot(
                input_filename=filename,
                output_filename=output_filename.replace(".dat", "_mass_scatter.png"),
                x_index=0,
                y_index=1,
                x_label="$m_1 [M_\odot]$",
                y_label="$m_2 [M_\odot]$",
            )
            scatter2d_plot(
                input_filename=filename,
                output_filename=output_filename.replace(".dat", "_spin_scatter.png"),
                x_index=2,
                y_index=3,
                x_label="$a_1$",
                y_label="$a_2$",
            )
