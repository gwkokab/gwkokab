# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable

import h5py
import numpy as np
from jax import nn as jnn, numpy as jnp, random as jrd
from loguru import logger
from numpyro.distributions.distribution import enable_validation

from gwkanal.core.inference_io import PoissonMeanEstimationLoader
from gwkanal.core.utils import PRNGKeyMixin, to_structured
from gwkanal.utils.common import read_json
from gwkanal.utils.regex import match_all
from gwkokab.models.utils import ScaledMixture
from gwkokab.parameters import default_relation_mesh, Parameters as P
from gwkokab.utils.tools import error_if


class SyntheticEventsBase(PRNGKeyMixin, ABC):
    def __init__(
        self,
        filename: str,
        model_fn: Callable[..., ScaledMixture],
        model_params_filename: str,
        poisson_mean_filename: str,
        derive_parameters: bool = False,
    ) -> None:
        self.filename = filename
        self.poisson_mean_filename = poisson_mean_filename
        self.derive_parameters = derive_parameters

        # Initialize model
        raw_params = read_json(model_params_filename)
        matched_params = match_all(self.model_parameters, raw_params)
        self.model_params = self.modify_model_params(matched_params)
        self.model_fn = model_fn(**self.model_params)

    @property
    @abstractmethod
    def parameters(self) -> tuple[str, ...]:
        """Returns the parameters (intrinsic + extrinsic).

        Returns
        -------
        tuple[str, ...]
            list of parameters.
        """
        pass

    @property
    @abstractmethod
    def model_parameters(self) -> list[str]:
        """Returns the model parameters.

        Returns
        -------
        list[str]
            list of model parameters.
        """
        pass

    def modify_model_params(self, params: dict) -> dict:
        """Hook for subclasses to modify parameters before model instantiation."""
        return params

    def _ensure_mass_ordering(self, population: jnp.ndarray) -> jnp.ndarray:
        """Enforces m1 >= m2 for mass parameters if present in the dataset."""
        mass_pairs = [
            (P.PRIMARY_MASS_SOURCE, P.SECONDARY_MASS_SOURCE),
            (P.PRIMARY_MASS_DETECTED, P.SECONDARY_MASS_DETECTED),
        ]

        for m1_key, m2_key in mass_pairs:
            if m1_key in self.parameters and m2_key in self.parameters:
                idx1, idx2 = (
                    self.parameters.index(m1_key),
                    self.parameters.index(m2_key),
                )
                m1, m2 = population[:, idx1], population[:, idx2]

                swapped_mask = m2 > m1
                if jnp.any(swapped_mask):
                    logger.debug(
                        f"Ordering masses for {jnp.sum(swapped_mask)} samples."
                    )
                    new_m1 = jnp.maximum(m1, m2)
                    new_m2 = jnp.minimum(m1, m2)
                    population = population.at[:, idx1].set(new_m1)
                    population = population.at[:, idx2].set(new_m2)

        return population

    def _generate_population(
        self, size: int, log_selection_fn: Callable
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate population for a realization via rejection/importance sampling."""

        # Oversample to account for selection effects
        buffer_size = size + 10_000
        raw_pop, [raw_indices] = self.model_fn.sample_with_intermediates(
            self.rng_key, (buffer_size,)
        )

        raw_pop = self._ensure_mass_ordering(raw_pop)

        # Compute selection weights (VT)
        log_selection = log_selection_fn(raw_pop)
        log_selection = jnp.nan_to_num(
            log_selection, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
        )
        weights = jnn.softmax(log_selection)

        # Resample based on weights
        resample_idx = jrd.choice(
            self.rng_key, jnp.arange(buffer_size), p=weights, shape=(size,)
        )

        return raw_pop, raw_indices, raw_pop[resample_idx], raw_indices[resample_idx]

    def from_inverse_transform_sampling(self) -> None:
        pmean_loader = PoissonMeanEstimationLoader.from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        log_selection_fn, pmean_estimator, _, _ = pmean_loader.get_estimators()

        exp_rate = pmean_estimator(self.model_fn)
        size = int(jrd.poisson(self.rng_key, exp_rate))

        logger.info(f"Expected rate: {exp_rate:.2f} | Realized size: {size}")

        error_if(
            size <= 0, msg=f"Population size is {size}. Check your VT or model configs."
        )

        raw_pop, raw_idx, pop, idx = self._generate_population(size, log_selection_fn)
        self.save_population(pop, idx, raw_pop, raw_idx)

    def save_population(
        self,
        population: np.ndarray,
        indices: np.ndarray,
        raw_population: np.ndarray,
        raw_indices: np.ndarray,
    ) -> None:
        current_params = self.parameters

        if self.derive_parameters:
            mesh = default_relation_mesh()
            population, current_params = mesh.resolve_from_arrays(
                population, self.parameters
            )
            raw_population, _ = mesh.resolve_from_arrays(
                raw_population, self.parameters
            )

        with h5py.File(self.filename, "w") as f:
            f.create_dataset(
                "injection_data", data=to_structured(population, current_params)
            )
            f.create_dataset("injection_indices", data=indices.astype(np.uint32))
            f.create_dataset(
                "raw_injection_data", data=to_structured(raw_population, current_params)
            )
            f.create_dataset(
                "raw_injection_indices", data=raw_indices.astype(np.uint32)
            )
            f.attrs["parameters"] = np.array(current_params, dtype="S")


def injection_generator_parser() -> ArgumentParser:
    enable_validation()
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Generate a population of CBCs",
    )
    # Grouping arguments for better --help readability
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--output-filename",
        default="synthetic_events.hdf5",
        help="Output HDF5 path",
        type=str,
    )
    io_group.add_argument(
        "--model-params", help="JSON model params", type=str, required=True
    )
    io_group.add_argument(
        "--pmean-cfg", help="Poisson mean config", type=str, default="pmean.json"
    )

    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument("--derive-parameters", action="store_true")
    proc_group.add_argument("--seed", default=37, type=int)

    return parser
