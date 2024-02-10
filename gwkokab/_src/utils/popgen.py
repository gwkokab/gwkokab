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
import sys
from typing_extensions import Optional

import h5py
import jax
import numpy as np
from jax import numpy as jnp
from jaxampler._src.jobj import JObj
from jaxtyping import Array
from tqdm import tqdm

from ..models import *
from ..vts import interpolate_hdf5
from .misc import dump_configurations
from .plotting import scatter2d_plot, scatter3d_plot


class PopulationGenerator:
    """Class to generate population and save them to disk."""

    def __init__(self, general: dict, models: list[dict], selection_effect: Optional[dict] = None) -> None:
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
        self._extra_size = 1500
        self._extra_error_size = 10000
        self._vt_filename = selection_effect.get("vt_filename", None) if selection_effect else None
        self._vt_columns = selection_effect.get("vt_columns", None) if selection_effect else None

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

    def weight_over_m1m2(
        self,
        realizations: Array,
        raw_interpolator_filename: str,
        n_out: int,
        m1_col_index: int,
        m2_col_index: int,
        return_array: bool = False,
        return_index: bool = False,
    ) -> tuple[Array, Array] | Array:
        dat_mass = realizations[:, [m1_col_index, m2_col_index]]

        with h5py.File(raw_interpolator_filename, "r") as VTs:
            raw_interpolator = interpolate_hdf5(VTs)

        weights = raw_interpolator(dat_mass)
        weights /= jnp.sum(weights)  # normalizes

        indexes_all = np.arange(len(dat_mass))
        downselected = jax.random.choice(JObj.get_key(None), indexes_all, p=weights, shape=(n_out,))

        if return_index and not return_array:
            return downselected

        downselected_pop = realizations[downselected]
        if return_index:
            return downselected_pop, downselected
        return downselected_pop

    def generate_injections(self) -> dict[str, int]:
        os.makedirs(self._root_container, exist_ok=True)

        size = self._size + self._extra_size

        bar = tqdm(
            total=self._num_realizations * len(self._models),
            desc="Generating injections",
            unit="injections",
            unit_scale=True,
            file=sys.stdout,
            dynamic_ncols=True,
        )

        for i in range(self._num_realizations):
            container = f"{self._root_container}/realization_{i}"
            config_filename = f"{container}/{self._config_filename}"
            injection_filename = f"{container}/injections.dat"

            os.makedirs(container, exist_ok=True)
            os.makedirs(f"{container}/plots", exist_ok=True)

            config_vals = []
            col_names = []
            realisations = jnp.empty((size, 0))
            for model in self._models:
                model_instance: AbstractModel = eval(model["model"])(**model["params"])
                rvs = model_instance.samples(size).reshape((size, -1))
                realisations = jnp.concatenate((realisations, rvs), axis=-1)

                config_vals.extend([(x, model["params"][x]) for x in model["config_vars"]])
                col_names.extend(model["col_names"])

                bar.update(1)
            bar.refresh()

            dump_configurations(config_filename, *config_vals)

            indexes = {var: i for i, var in enumerate(col_names)}

            np.savetxt(injection_filename, realisations, header="\t".join(col_names))

            del realisations
        bar.colour = "green"
        bar.close()

        return indexes

    def generate_injections_plots(self, indexes, suffix: str) -> None:
        populations = glob.glob(f"{self._root_container}/realization_*/injections.dat")
        for filename in tqdm(
            populations,
            desc="Ploting Injections",
            total=self._num_realizations,
            unit="realization",
            unit_scale=True,
        ):
            output_filename = filename.replace("injections.dat", "plots")
            scatter2d_plot(
                input_filename=filename,
                output_filename=output_filename + f"/{suffix}_mass_injs.png",
                x_index=indexes["m1_source"],
                y_index=indexes["m2_source"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                plt_title="Injections",
            )

            scatter3d_plot(
                input_filename=filename,
                output_filename=output_filename + f"/{suffix}_mass_ecc_injs.png",
                x_index=indexes["m1_source"],
                y_index=indexes["m2_source"],
                z_index=indexes["ecc"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                z_label=r"$\epsilon$",
                plt_title="Injections",
            )

    def generate(self):
        """Generate population and save them to disk."""
        indexes = self.generate_injections()
        self.generate_injections_plots(indexes, "unweighted")
