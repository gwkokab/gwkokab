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
        self._extra_size = 1_500
        self._extra_error_size = 10_000
        self._vt_filename = selection_effect.get("vt_filename", None) if selection_effect else None

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
        input_filename: str,
        output_filename: str,
        n_out: int,
        m1_col_index: int,
        m2_col_index: int,
    ) -> tuple[Array, Array] | Array:
        realizations = np.loadtxt(input_filename)
        dat_mass = realizations[:, [m1_col_index, m2_col_index]]

        weights = self._raw_interpolator(dat_mass)
        weights /= jnp.sum(weights)  # normalizes

        indexes_all = np.arange(len(dat_mass))
        downselected = jax.random.choice(JObj.get_key(None), indexes_all, p=weights, shape=(n_out,))

        realizations = realizations[downselected]

        np.savetxt(output_filename, realizations, header="\t".join(self._col_names))

    def weighted_injection(self, raw_interpolator_filename: str):
        with h5py.File(raw_interpolator_filename, "r") as VTs:
            self._raw_interpolator = interpolate_hdf5(VTs)

        for i in tqdm(
            range(self._num_realizations),
            desc="Weighting injections",
            unit="realization",
            unit_scale=True,
            file=sys.stdout,
        ):
            container = f"{self._root_container}/realization_{i}"
            injection_filename = f"{container}/injections.dat"
            weighted_injection_filename = f"{container}/weighted_injections.dat"

            self.weight_over_m1m2(
                input_filename=injection_filename,
                output_filename=weighted_injection_filename,
                n_out=self._size,
                m1_col_index=self._col_names.index("m1_source"),
                m2_col_index=self._col_names.index("m2_source"),
            )

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

        col_names = None

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

        self._col_names = col_names

        return indexes

    def generate_injections_plots(
        self,
        indexes,
        filename: str,
        suffix: str,
        bar_title: str = "Ploting Injections",
    ) -> None:
        populations = glob.glob(f"{self._root_container}/realization_*/{filename}")
        for pop_filename in tqdm(
            populations,
            desc=bar_title,
            total=self._num_realizations,
            unit="realization",
            unit_scale=True,
        ):
            output_filename = pop_filename.replace(filename, "plots")
            scatter2d_plot(
                input_filename=pop_filename,
                output_filename=output_filename + f"/{suffix}_mass_injs.png",
                x_index=indexes["m1_source"],
                y_index=indexes["m2_source"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                plt_title="Injections",
            )

            scatter3d_plot(
                input_filename=pop_filename,
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
        self.generate_injections_plots(indexes, "injections.dat", "unweighted", "Ploting Unweighted Injections")
        if self._vt_filename:
            self.weighted_injection(self._vt_filename)
            self.generate_injections_plots(
                indexes, "weighted_injections.dat", "weighted", "Ploting Weighted Injections"
            )
