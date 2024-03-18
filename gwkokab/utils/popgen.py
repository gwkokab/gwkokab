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
from typing_extensions import Optional

import jax
import numpy as np
from jax import numpy as jnp, vmap
from numpyro.distributions import *
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from ..errors import error_factory
from ..models import *
from ..utils.misc import get_key
from ..vts.utils import interpolate_hdf5
from .plotting import scatter2d_batch_plot, scatter2d_plot, scatter3d_batch_plot, scatter3d_plot


class PopulationGenerator(object):
    """This class is used to generate population and save them to disk."""

    key = None

    def __init__(
        self,
        general: dict,
        models: list[dict],
        selection_effect: Optional[dict] = None,
        plots: Optional[dict] = None,
    ) -> None:
        """Initialize the PopulationGenerator class.

        :param general: general configurations
        :param models: list of models
        :param selection_effect: selection effect configurations, defaults to `None`
        """
        self.check_general(general)
        for model in models:
            self.check_models(model)

        self._size: int = general["size"]
        self._error_size: int = general["error_size"]
        self._root_container: str = general["root_container"]
        self._event_filename: str = general["event_filename"]
        self._config_filename: str = general["config_filename"]
        self._num_realizations: int = general["num_realizations"]
        self._models: list = models
        self._extra_size = general["extra_size"]
        self._vt_filename = selection_effect.get("vt_filename", None) if selection_effect else None
        self._plots = plots
        self._verbose = general.get("verbose", True)

    @staticmethod
    def check_general(general: dict) -> None:
        """Check the general configurations.

        :param general: general configurations
        """
        assert general.get("size", None) is not None
        assert general.get("error_size", None) is not None
        assert general.get("root_container", None) is not None
        assert general.get("event_filename", None) is not None
        assert general.get("config_filename", None) is not None

    @staticmethod
    def check_models(model: dict) -> None:
        """Check the model configurations.

        :param model: model configurations
        """
        assert model.get("model", None) is not None
        assert model.get("config_vars", None) is not None
        assert model.get("col_names", None) is not None
        assert model.get("params", None) is not None
        assert model.get("error_type", None) is not None

    def weight_over_m1m2(
        self,
        input_filename: str,
        output_filename: str,
        n_out: int,
        m1_col_index: int,
        m2_col_index: int,
    ) -> None:
        """Weighting masses by VTs.

        :param input_filename: input file name
        :param output_filename: output file name
        :param n_out: number of output
        :param m1_col_index: index of m1 column
        :param m2_col_index: index of m2 column
        """
        realizations = np.loadtxt(input_filename)

        # weights = interpolate_hdf5(realizations[:, m1_col_index], realizations[:, m2_col_index], self._vt_filename)
        weights = self._raw_interpolator(realizations[:, m1_col_index], realizations[:, m2_col_index])
        weights /= np.sum(weights)  # normalizes

        indexes_all = np.arange(len(weights))
        self.key = get_key(self.key)
        downselected = jax.random.choice(self.key, indexes_all, p=weights, shape=(n_out,))

        new_realizations = realizations[downselected]

        np.savetxt(output_filename, new_realizations, header="\t".join(self._col_names))

    def weighted_injection(self, raw_interpolator_filename: str) -> None:
        """Weighting injections by VTs.

        :param raw_interpolator_filename: raw interpolator file name
        """

        self._raw_interpolator = interpolate_hdf5(raw_interpolator_filename)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Weighting injections", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            disable=not self._verbose,
        ) as progress:
            task = progress.add_task("Weighting injections", total=self._num_realizations)

            for i in range(self._num_realizations):
                container = f"{self._root_container}/realization_{i}"
                injection_filename = f"{container}/injections.dat"
                weighted_injection_filename = f"{container}/injections.dat"

                self.weight_over_m1m2(
                    input_filename=injection_filename,
                    output_filename=weighted_injection_filename,
                    n_out=self._size,
                    m1_col_index=self._col_names.index("m1_source"),
                    m2_col_index=self._col_names.index("m2_source"),
                )

                injections = np.loadtxt(weighted_injection_filename)
                os.makedirs(f"{container}/injections", exist_ok=True)
                for j in range(self._size):
                    np.savetxt(
                        f"{container}/injections/{self._event_filename.format(j)}",
                        injections[j, :].reshape(1, -1),
                        header="\t".join(self._col_names),
                    )
                progress.advance(task, 1)

    def weighted_posteriors(self, raw_interpolator_filename: str) -> None:
        """Weighting posteriors by VTs.

        :param raw_interpolator_filename: raw interpolator file name
        """

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Plotting Posterior", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            disable=not self._verbose,
        ) as progress:
            task1 = progress.add_task("Weighting posteriors", total=self._num_realizations * self._size)
            for i in range(self._num_realizations):
                container = f"{self._root_container}/realization_{i}"
                for j in range(self._size):
                    posterior_filename = f"{container}/posteriors/{self._event_filename.format(j)}"

                    self.weight_over_m1m2(
                        input_filename=posterior_filename,
                        output_filename=posterior_filename,
                        n_out=self._error_size,
                        m1_col_index=self._col_names.index("m1_source"),
                        m2_col_index=self._col_names.index("m2_source"),
                    )
                    progress.advance(task1, 1)

    def generate_injections(self) -> None:
        """Generate injections and save them to disk."""
        os.makedirs(self._root_container, exist_ok=True)

        size = self._size + self._extra_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating injections", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            disable=not self._verbose,
        ) as progress:
            task = progress.add_task("Generating injections", total=self._num_realizations * len(self._models))

            for i in range(self._num_realizations):
                container = f"{self._root_container}/realization_{i}"
                config_filename = f"{container}/{self._config_filename}"
                injection_filename = f"{container}/injections.dat"

                os.makedirs(container, exist_ok=True)

                realisations = jnp.empty((size, 0))
                for model_instance in self._model_instances:
                    self.key = get_key(self.key)
                    rvs = model_instance.sample(self.key, sample_shape=(size,)).reshape((size, -1))
                    realisations = jnp.concatenate((realisations, rvs), axis=-1)

                    progress.advance(task, 1)

                # dump_configurations(config_filename, *self._config_vals)
                np.savetxt(
                    config_filename,
                    np.array([list(zip(*self._config_vals))[1]]),
                    delimiter="\t",
                    fmt="%s",
                    header="\t".join(list(zip(*self._config_vals))[0]),
                )

                np.savetxt(injection_filename, realisations, header="\t".join(self._col_names))

                del realisations

    def generate_injections_plots(
        self,
        filename: str,
    ) -> None:
        """Generate injections plots.

        :param filename: name of the file
        :param suffix: suffix for the output file
        :param bar_title: title for the progress bar, defaults to "Plotting Injections"
        """
        populations = glob.glob(f"{self._root_container}/realization_*/{filename}")
        for realization in glob.glob(f"{self._root_container}/realization_*"):
            os.makedirs(f"{realization}/plots", exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Plotting injections", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            disable=not self._verbose,
        ) as progress:
            task = progress.add_task("Plotting injections", total=len(populations))

            for pop_filename in populations:
                output_filename = pop_filename.replace(filename, "plots")
                for ins in self._plots["injs"]:
                    if len(ins) == 2:
                        scatter2d_plot(
                            input_filename=pop_filename,
                            output_filename=output_filename + f"/{ins[0]}_{ins[1]}_injs.png",
                            x_index=self._indexes[ins[0]],
                            y_index=self._indexes[ins[1]],
                            x_label=ins[0],
                            y_label=ins[1],
                            plt_title="Injections",
                        )
                    elif len(ins) == 3:
                        scatter3d_plot(
                            input_filename=pop_filename,
                            output_filename=output_filename + f"/{ins[0]}_{ins[1]}_{ins[2]}_injs.png",
                            x_index=self._indexes[ins[0]],
                            y_index=self._indexes[ins[1]],
                            z_index=self._indexes[ins[2]],
                            x_label=ins[0],
                            y_label=ins[1],
                            z_label=ins[2],
                            plt_title="Injections",
                        )
                progress.advance(task, 1)

    def add_error(self) -> None:
        """Add error to the injections."""
        error_size = self._error_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Adding Error", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            disable=not self._verbose,
        ) as progress:
            task = progress.add_task(
                "Adding error",
                total=self._num_realizations * self._size,
            )
            for i in range(self._num_realizations):
                container = f"{self._root_container}/realization_{i}"
                injection_filename = f"{container}/injections.dat"
                realizations = np.loadtxt(injection_filename)
                err_realizations = np.empty((self._size, error_size, 0))

                os.makedirs(f"{container}/posteriors", exist_ok=True)

                k = 0
                for t, c in enumerate(self._col_count):
                    keys = jax.random.split(self.key, self._size)
                    self.key = get_key(self.key)
                    rvs = vmap(
                        lambda x, pk: error_factory(
                            error_type=self._error_type[t],
                            x=x,
                            size=error_size,
                            key=pk,
                            **self._error_params[t],
                        )
                    )(realizations[:, k : k + c], keys).reshape((self._size, error_size, -1))
                    err_realizations = np.concatenate((err_realizations, rvs), axis=-1)
                    k += c

                mask = np.isnan(err_realizations).any(axis=2)

                for j in range(self._size):
                    masked_err_realizations = err_realizations[j, ~mask[j]]

                    np.savetxt(
                        f"{container}/posteriors/{self._event_filename.format(j)}",
                        masked_err_realizations,
                        header="\t".join(self._col_names),
                    )
                    progress.advance(task, 1)

    def generate_batch_plots(self) -> None:
        realization_regex = f"{self._root_container}/realization_*"

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Plotting Posterior", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            MofNCompleteColumn(),
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
            disable=not self._verbose,
        ) as progress:
            task = progress.add_task(
                "Plotting Posterior",
                total=self._num_realizations,
            )
            for realization in glob.glob(realization_regex):
                os.makedirs(f"{realization}/plots", exist_ok=True)

                for pin in self._plots["posts"]:
                    if len(pin) == 2:
                        scatter2d_batch_plot(
                            file_pattern=realization + f"/posteriors/{self._event_filename.format('*')}",
                            output_filename=f"{realization}/plots/{pin[0]}_{pin[1]}_posts.png",
                            x_index=self._indexes[pin[0]],
                            y_index=self._indexes[pin[1]],
                            x_label=pin[0],
                            y_label=pin[1],
                            plt_title=f"Posterior {pin[0]} vs {pin[1]}",
                        )
                    elif len(pin) == 3:
                        scatter3d_batch_plot(
                            file_pattern=realization + f"/posteriors/{self._event_filename.format('*')}",
                            output_filename=f"{realization}/plots/{pin[0]}_{pin[1]}_{pin[2]}_posts.png",
                            x_index=self._indexes[pin[0]],
                            y_index=self._indexes[pin[1]],
                            z_index=self._indexes[pin[2]],
                            x_label=pin[0],
                            y_label=pin[1],
                            z_label=pin[2],
                            plt_title=f"Posterior {pin[0]} vs {pin[1]} vs {pin[2]}",
                        )
                progress.advance(task, 1)

    def generate(self) -> None:
        """Generate population and save them to disk."""
        self._col_names: list[str] = []
        self._col_count: list[int] = []
        self._config_vals: list[tuple[str, int]] = []
        self._model_instances: list[Distribution] = []
        self._error_type: list[str] = []
        self._error_params: list[dict] = []

        for model in self._models:
            model_instance: Distribution = eval(model["model"])(**model["params"])
            self._model_instances.append(model_instance)
            self._config_vals.extend([(x[1], model["params"][x[0]]) for x in model["config_vars"]])
            self._col_names.extend(model["col_names"])
            self._col_count.append(len(model["col_names"]))
            self._error_type.append(model["error_type"])
            self._error_params.append(model.get("error_params", {}))

        self._indexes = {var: i for i, var in enumerate(self._col_names)}

        self.generate_injections()
        if self._vt_filename:
            self.weighted_injection(self._vt_filename)
        self.add_error()
        if self._plots:
            if self._plots.get("posts", None):
                self.generate_batch_plots()
            if self._plots.get("injs", None):
                self.generate_injections_plots("injections.dat")
