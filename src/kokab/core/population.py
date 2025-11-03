# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import os
import warnings
from collections.abc import Callable
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import tqdm
from jax import nn as jnn, numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger
from numpyro.util import is_prng_key

from gwkokab.models.utils import ScaledMixture
from gwkokab.models.wrappers import ModelRegistry
from gwkokab.utils.tools import error_if


PROGRESS_BAR_TEXT_WITDH = 25

error_magazine = ModelRegistry()


def ensure_dat_extension(filename: str) -> str:
    """Transform a filename to end with .dat if it does not have an extension.

    Parameters
    ----------
    filename : str
        Name of the file

    Returns
    -------
    str
        Filename ending with .dat

    Raises
    ------
    ValueError
        If filename has an extension other than .dat
    """
    if filename.endswith(".dat"):
        return filename
    elif "." not in filename:
        return filename + ".dat"
    else:
        ext = filename.split(".")[-1]
        raise ValueError(
            f"Invalid filename {filename!r}: found extension '.{ext}' but must end with '.dat' or have no extension"
        )


def add_mean_and_covariance(
    noisy_data: np.ndarray,
    parameters: List[str],
    tile_covariance: Optional[List[List[str]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_dim = len(parameters)
    means = np.empty((n_dim,))
    covs = np.zeros((n_dim, n_dim))

    mean = np.mean(noisy_data, axis=0)
    error_if(np.isnan(mean).any(), msg="Mean contains NaN values.")
    means = mean

    if tile_covariance is not None:
        for tile in tile_covariance:
            indices = [parameters.index(param) for param in tile]
            noisy_data_tile = noisy_data[:, indices]
            cov_tile = np.cov(noisy_data_tile, rowvar=False)
            error_if(
                np.isnan(cov_tile).any(),
                msg=("Covariance contains NaN values for tile {}".format(tile)),
            )
            if cov_tile.shape == ():
                cov_tile = cov_tile.reshape(1, 1)
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    covs[idx_i, idx_j] = cov_tile[i, j]
    else:
        cov = np.cov(noisy_data, rowvar=False)
        error_if(np.isnan(cov).any(), msg="Covariance contains NaN values.")
        covs = cov

    return means, covs


class PopulationFactory:
    error_dir: str = "posteriors"
    event_filename: str = "event_{}"
    injection_filename: str = "injections"
    realizations_dir: str = "realization_{}"
    root_dir: str = "data"
    verbose: bool = True
    mean_covs_filename: str = "means_covs.hdf5"

    def __init__(
        self,
        model_fn: Union[ScaledMixture, Callable[..., ScaledMixture]],
        model_params: dict[str, Array],
        parameters: List[str],
        log_selection_fn: Optional[Callable[[Array], Array]],
        poisson_mean_estimator: Callable[[ScaledMixture], Array],
        num_realizations: int = 5,
        error_size: int = 2_000,
        tile_covariance: Optional[List[List[str]]] = None,
    ) -> None:
        """Class with methods equipped to generate population for each realizations and
        adding errors in it.

        Parameters
        ----------
        model_fn : Union[ScaledMixture, Callable[..., ScaledMixture]]
            Model for the population. If a callable is provided, it should return a
            `ScaledMixture` model.
        model_params : dict[str, Array]
            Parameters for the model.
        parameters : List[str]
            Parameters for the model in order.
        log_selection_fn : Callable[[Array], Array]
            logarithm of volume time sensitivity function.
        poisson_mean_estimator : Callable[[ScaledMixture], Array]
            Expected rate function.
        num_realizations : int, optional
            Number of realizations to generate, by default 5
        error_size : int, optional
            Size of the error to add in the population, by default 2_000

        Raises
        ------
        ValueError
            If model is not provided.
        ValueError
            If provided model is not a `ScaledMixture` model.
        ValueError
            If parameters are not provided.
        """
        self.model: ScaledMixture = model_fn(**model_params)
        self.parameters = parameters
        self.log_selection_fn = log_selection_fn
        self.poisson_mean_estimator = poisson_mean_estimator
        self.num_realizations = num_realizations
        self.error_size = error_size
        self.tile_covariance = tile_covariance

        self.event_filename = ensure_dat_extension(self.event_filename)
        self.injection_filename = ensure_dat_extension(self.injection_filename)

    def _generate_population(
        self, size: int, *, key: PRNGKeyArray
    ) -> Tuple[Array, Array, Array, Array]:
        r"""Generate population for a realization."""

        old_size = size
        if self.log_selection_fn is not None:
            size += int(1e4)

        population, [indices] = self.model.sample_with_intermediates(key, (size,))

        raw_population = population
        raw_indices = indices

        m1_index = self.parameters.index("mass_1_source")
        m2_index = self.parameters.index("mass_2_source")

        m1 = population[:, m1_index]
        m2 = population[:, m2_index]

        mask = np.less_equal(m2, m1)

        count = jnp.sum(mask)

        logger.debug(
            "Number of injections with m2 <= m1: {count} out of {size}",
            count=count,
            size=size,
        )

        population = population[mask]
        indices = indices[mask]

        if self.log_selection_fn is not None:
            _, key = jrd.split(key)
            log_selection = self.log_selection_fn(population)
            log_selection = jnp.nan_to_num(
                log_selection,
                nan=-jnp.inf,
                posinf=-jnp.inf,
                neginf=-jnp.inf,
            )
            vt = jnn.softmax(log_selection)
            _, key = jrd.split(key)
            index = jrd.choice(
                key, jnp.arange(population.shape[0]), p=vt, shape=(old_size,)
            )

            population = population[index]
            indices = indices[index]

        return raw_population, raw_indices, population, indices

    def _generate_realizations(self, key: PRNGKeyArray) -> None:
        r"""Generate realizations for the population."""
        poisson_key, rate_key = jrd.split(key)
        exp_rate = self.poisson_mean_estimator(self.model)
        logger.debug(f"Expected rate for the population is {exp_rate}")
        size = int(jrd.poisson(poisson_key, exp_rate))
        logger.debug(f"Population size is {size}")
        key = rate_key
        if size <= 0:
            raise ValueError(
                f"Population size is {size}. This can be a result of following:\n"
                "\t1. The rate is zero.\n"
                "\t2. The volume is zero.\n"
                "\t3. The models are not selected for rate calculation.\n"
                "\t4. VT file is not provided or is not valid.\n"
                "\t5. Or some other reason."
            )
        pop_keys = jrd.split(key, self.num_realizations)
        os.makedirs(self.root_dir, exist_ok=True)
        with tqdm.tqdm(
            range(self.num_realizations),
            unit="realization",
            total=self.num_realizations,
        ) as pbar:
            pbar.set_description(
                "Generating population".ljust(PROGRESS_BAR_TEXT_WITDH, " ")
            )
            for i in pbar:
                realizations_path = os.path.join(
                    self.root_dir, self.realizations_dir.format(i)
                )
                os.makedirs(realizations_path, exist_ok=True)

                raw_population, raw_indices, population, indices = (
                    self._generate_population(size, key=pop_keys[i])
                )

                if population.shape == ():
                    continue

                injections_file_path = os.path.join(
                    realizations_path, self.injection_filename
                )
                color_indices_file_path = os.path.join(realizations_path, "color.dat")
                np.savetxt(
                    injections_file_path,
                    population,
                    header=" ".join(self.parameters),
                    comments="",  # To remove the default comment character '#'
                )
                np.savetxt(
                    color_indices_file_path,
                    indices,
                    comments="",  # To remove the default comment character '#'
                    fmt="%d",
                )
                raw_injections_file_path = os.path.join(
                    realizations_path, "raw_" + self.injection_filename
                )
                raw_color_indices_file_path = os.path.join(
                    realizations_path, "raw_color.dat"
                )
                np.savetxt(
                    raw_injections_file_path,
                    raw_population,
                    header=" ".join(self.parameters),
                    comments="",  # To remove the default comment character '#'
                )
                np.savetxt(
                    raw_color_indices_file_path,
                    raw_indices,
                    comments="",  # To remove the default comment character '#'
                    fmt="%d",
                )

    def _add_error(self, realization_number, *, key: PRNGKeyArray) -> None:
        r"""Adds error to the realizations' population."""
        realizations_path = os.path.join(
            self.root_dir, self.realizations_dir.format(realization_number)
        )

        heads: List[List[int]] = []
        error_fns: List[Callable[[Array, int, PRNGKeyArray], Array]] = []

        for head, err_fn in error_magazine.registry.items():
            _head = []
            for h in head:
                i = self.parameters.index(h)
                _head.append(i)
            heads.append(_head)
            error_fns.append(err_fn)

        output_dir = os.path.join(
            realizations_path, self.error_dir, self.event_filename
        )

        os.makedirs(os.path.join(realizations_path, self.error_dir), exist_ok=True)

        injections_file_path = os.path.join(realizations_path, self.injection_filename)
        data_inj = np.loadtxt(injections_file_path, skiprows=1).reshape(
            -1, len(self.parameters)
        )

        n_injections = data_inj.shape[0]

        keys = jrd.split(key, n_injections * (len(heads) + 1))
        means: List[Optional[np.ndarray]] = [None for _ in range(n_injections)]
        covs: List[Optional[np.ndarray]] = [None for _ in range(n_injections)]
        for index in range(n_injections):
            noisy_data = np.empty((self.error_size, len(self.parameters)))
            data = data_inj[index]
            i = 0
            for head, err_fn in zip(heads, error_fns):
                key_idx = index * len(heads) + i
                noisy_data_i: Array = err_fn(data[head], self.error_size, keys[key_idx])
                if noisy_data_i.ndim == 1:
                    noisy_data_i = noisy_data_i.reshape(self.error_size, -1)
                noisy_data[:, head] = noisy_data_i
                i += 1
            nan_mask = np.isnan(noisy_data).any(axis=1)
            noisy_data = noisy_data[~nan_mask]  # type: ignore

            if noisy_data.shape[0] < 1:
                warnings.warn(
                    f"Skipping file {index} due to all NaN values or insufficient data.",
                    category=UserWarning,
                )
                continue

            np.savetxt(
                output_dir.format(index),
                noisy_data,
                header=" ".join(self.parameters),
                comments="",  # To remove the default comment character '#'
            )

            mean, cov = add_mean_and_covariance(
                noisy_data,
                self.parameters,
                tile_covariance=self.tile_covariance,
            )
            means[index] = mean
            covs[index] = cov
        with h5py.File(
            os.path.join(realizations_path, self.mean_covs_filename), "w"
        ) as f:
            for i, (mean, cov) in enumerate(zip(means, covs)):
                if mean is not None and cov is not None:
                    event_name = "event_{}".format(i)
                    event_group = f.create_group(event_name)
                    event_group.create_dataset("mean", data=mean)
                    event_group.create_dataset("cov", data=cov)

    def produce(self, key: PRNGKeyArray) -> None:
        """Generate realizations and add errors to the populations.

        Parameters
        ----------
        key : PRNGKeyArray
            Pseudo-random number generator key for reproducibility, by default None
        """
        assert is_prng_key(key)
        self._generate_realizations(key)
        keys = jrd.split(key, self.num_realizations)
        with tqdm.tqdm(
            range(self.num_realizations),
            unit="realization",
            total=self.num_realizations,
        ) as pbar:
            pbar.set_description("Adding errors".ljust(PROGRESS_BAR_TEXT_WITDH, " "))
            for i in pbar:
                self._add_error(i, key=keys[i])
