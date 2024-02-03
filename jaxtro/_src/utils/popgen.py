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

import numpy as np
from jax import numpy as jnp, vmap
from tqdm import tqdm

from ..models import *
from .misc import dump_configurations
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

    def generate(self):
        """Generate population and save them to disk."""
        os.makedirs(self._root_container, exist_ok=True)

        col_names = []

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
            os.makedirs(f"{container}/injections", exist_ok=True)
            os.makedirs(f"{container}/posteriors", exist_ok=True)
            os.makedirs(f"{container}/plots", exist_ok=True)

            config_vals = []
            col_names = []
            realisations = jnp.empty((self._size, 0))
            realisations_err = jnp.empty((self._size, self._error_size, 0))
            for model in self._models:
                model_instance = eval(model["model"])(**model["params"])
                rvs = model_instance.samples(self._size).reshape((self._size, -1))
                err_rvs = vmap(
                    lambda x: model_instance.add_error(
                        x=x,
                        scale=self._error_scale,
                        size=self._error_size,
                    ),
                    in_axes=(0,),
                )(rvs)
                realisations = jnp.concatenate((realisations, rvs), axis=-1)
                realisations_err = jnp.concatenate((realisations_err, err_rvs), axis=-1)

                config_vals.extend([(x, model["params"][x]) for x in model["config_vars"]])
                col_names.extend(model["col_names"])

            dump_configurations(config_filename, *config_vals)

            np.savetxt(injection_filename, realisations, header="\t".join(col_names))

            for event_num, realisation in enumerate(realisations):
                filename_event = f"{container}/posteriors/{self._event_filename.format(event_num)}"
                np.savetxt(
                    filename_event,
                    realisations_err[event_num, :, :],
                    header="\t".join(col_names),
                )

                filename_inj = f"{container}/injections/inj_{event_num}.dat"
                np.savetxt(
                    filename_inj,
                    realisation.reshape((1, -1)),
                    header="\t".join(col_names),
                )

        realization_regex = f"{self._root_container}/realization_*"
        indexes = {var: i for i, var in enumerate(col_names)}

        for realization in tqdm(
            glob.glob(realization_regex),
            desc="Ploting 2D Posterior",
            total=self._num_realizations,
            unit="event",
            unit_scale=True,
        ):
            scatter2d_batch_plot(
                file_pattern=realization + f"/posteriors/{self._event_filename.format('*')}",
                output_filename=f"{realization}/plots/mass_posterior.png",
                x_index=indexes["m1_source"],
                y_index=indexes["m2_source"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                plt_title="Mass Posteriors",
            )
            scatter2d_batch_plot(
                file_pattern=realization + f"/posteriors/{self._event_filename.format('*')}",
                output_filename=f"{realization}/plots/spin_posterior.png",
                x_index=indexes["a1"],
                y_index=indexes["a2"],
                x_label=r"$a_1$",
                y_label=r"$a_2$",
                plt_title="Spin Posteriors",
            )

        populations = glob.glob(f"{self._root_container}/realization_*/injections/population.dat")
        for filename in tqdm(
            populations,
            desc="Ploting Injections",
            total=self._num_realizations,
            unit="realization",
            unit_scale=True,
        ):
            output_filename = filename.replace("injections", "plots")
            scatter2d_plot(
                input_filename=filename,
                output_filename=output_filename.replace("population.dat", "mass_injs.png"),
                x_index=indexes["m1_source"],
                y_index=indexes["m2_source"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                plt_title="Injections",
            )

            scatter3d_plot(
                input_filename=filename,
                output_filename=output_filename.replace("population.dat", "mass_ecc_injs.png"),
                x_index=indexes["m1_source"],
                y_index=indexes["m2_source"],
                z_index=indexes["ecc"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                z_label=r"$\epsilon$",
                plt_title="Injections",
            )
