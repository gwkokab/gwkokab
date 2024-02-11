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
from jax import numpy as jnp, vmap
from jaxampler._src.jobj import JObj
from jaxtyping import Array
from tqdm import tqdm

from ..models import *
from ..vts import interpolate_hdf5
from .misc import dump_configurations
from .plotting import scatter2d_batch_plot, scatter2d_plot, scatter3d_plot


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
        self._extra_error_size = 10_00
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

            weighted_injections = np.loadtxt(weighted_injection_filename)
            os.makedirs(f"{container}/injections", exist_ok=True)
            for j in range(self._size):
                np.savetxt(
                    f"{container}/injections/{self._event_filename.format(j)}",
                    weighted_injections[j, :].reshape(1, -1),
                    header="\t".join(self._col_names),
                )

    def weighted_posteriors(self, raw_interpolator_filename: str):
        with h5py.File(raw_interpolator_filename, "r") as VTs:
            self._raw_interpolator = interpolate_hdf5(VTs)

        bar = tqdm(
            total=self._num_realizations * self._size,
            desc="Weighting posteriors",
            unit="events",
            unit_scale=True,
            file=sys.stdout,
        )

        for i in range(self._num_realizations):
            container = f"{self._root_container}/realization_{i}"
            for j in range(self._size):
                posterior_filename = f"{container}/posteriors/{self._event_filename.format(j)}"

                self.weight_over_m1m2(
                    input_filename=posterior_filename,
                    output_filename=posterior_filename,
                    n_out=self._size,
                    m1_col_index=self._col_names.index("m1_source"),
                    m2_col_index=self._col_names.index("m2_source"),
                )
                bar.update(1)

        bar.colour = "green"
        bar.close()

    def generate_injections(self) -> None:
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

            realisations = jnp.empty((size, 0))
            for model_instance in self._model_instances:
                rvs = model_instance.samples(size).reshape((size, -1))
                realisations = jnp.concatenate((realisations, rvs), axis=-1)

                bar.update(1)
            bar.refresh()

            dump_configurations(config_filename, *self._config_vals)

            np.savetxt(injection_filename, realisations, header="\t".join(self._col_names))

            del realisations
        bar.colour = "green"
        bar.close()

    def generate_injections_plots(
        self,
        filename: str,
        suffix: str,
        bar_title: str = "Plotting Injections",
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
                x_index=self._indexes["m1_source"],
                y_index=self._indexes["m2_source"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                plt_title="Injections",
            )

            scatter3d_plot(
                input_filename=pop_filename,
                output_filename=output_filename + f"/{suffix}_mass_ecc_injs.png",
                x_index=self._indexes["m1_source"],
                y_index=self._indexes["m2_source"],
                z_index=self._indexes["ecc"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                z_label=r"$\epsilon$",
                plt_title="Injections",
            )

    def add_error(self):
        error_size = self._error_size + self._extra_error_size
        bar = tqdm(
            total=self._num_realizations * self._size,
            desc="Adding error",
            unit="events",
            unit_scale=True,
            file=sys.stdout,
        )
        for i in range(self._num_realizations):
            container = f"{self._root_container}/realization_{i}"
            injection_filename = f"{container}/weighted_injections.dat"
            realizations = np.loadtxt(injection_filename)
            err_realizations = np.empty((self._size, error_size, 0))

            os.makedirs(f"{container}/posteriors", exist_ok=True)

            k = 0
            for c, model in zip(self._col_count, self._model_instances):
                rvs = vmap(
                    lambda x: model.add_error(
                        x=x,
                        scale=self._error_scale,
                        size=error_size,
                    )
                )(realizations[:, k : k + c])
                err_realizations = np.concatenate((err_realizations, rvs), axis=-1)
                err_realizations = jnp.nan_to_num(
                    err_realizations,
                    nan=-jnp.inf,
                    posinf=jnp.inf,
                    neginf=-jnp.inf,
                    copy=False,
                )
                k += c

            for j in range(self._size):
                np.savetxt(
                    f"{container}/posteriors/{self._event_filename.format(j)}",
                    err_realizations[j, :, :],
                    header="\t".join(self._col_names),
                )
                bar.update(1)
        bar.colour = "green"
        bar.close()

    def generate_posteriors_plots(self):
        realization_regex = f"{self._root_container}/realization_*"

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
                x_index=self._indexes["m1_source"],
                y_index=self._indexes["m2_source"],
                x_label=r"$m_1 [M_\odot]$",
                y_label=r"$m_2 [M_\odot]$",
                plt_title="Mass Posteriors",
            )
            scatter2d_batch_plot(
                file_pattern=realization + f"/posteriors/{self._event_filename.format('*')}",
                output_filename=f"{realization}/plots/spin_posterior.png",
                x_index=self._indexes["a1"],
                y_index=self._indexes["a2"],
                x_label=r"$a_1$",
                y_label=r"$a_2$",
                plt_title="Spin Posteriors",
            )

    def generate(self):
        """Generate population and save them to disk."""
        self._col_names = []
        self._col_count = []
        self._config_vals = []
        self._model_instances: list[AbstractModel] = []

        for model in self._models:
            model_instance: AbstractModel = eval(model["model"])(**model["params"])
            self._model_instances.append(model_instance)
            self._config_vals.extend([(x, model["params"][x]) for x in model["config_vars"]])
            self._col_names.extend(model["col_names"])
            self._col_count.append(len(model["col_names"]))

        self._indexes = {var: i for i, var in enumerate(self._col_names)}

        self.generate_injections()
        self.generate_injections_plots(
            "injections.dat",
            "unweighted",
            "Plotting Unweighted Injections",
        )
        if self._vt_filename:
            self.weighted_injection(self._vt_filename)
            self.generate_injections_plots(
                "weighted_injections.dat",
                "weighted",
                "Plotting Weighted Injections",
            )
        self.add_error()
        if self._vt_filename:
            self.weighted_posteriors(self._vt_filename)
        self.generate_posteriors_plots()
