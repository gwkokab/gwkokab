# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""Drawing a synthetic population of compact binary coalescences.

The first step of the end-to-end workflow. Given a population model and its true
hyper-parameters, :class:`SyntheticEventsBase` draws the number of detections from a
Poisson distribution with the model's own expected rate, then draws that many events
from the model and thins them by the selection function -- so the saved population is
what a search would actually have recovered, not what the universe contains.

The output HDF5 keeps both: the ``events`` that survive selection and the
``buffer_events`` they were drawn from, with the indices and resampling weights linking
them, so the effect of the selection function can be inspected afterwards.

See Also
--------
gwkokab.analysis.core.synthetic_pe :
    The next step, which blurs these true events into mock PE samples.
"""

from abc import ABC, abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Callable

import h5py
import numpy as np
from jax import nn as jnn, numpy as jnp, random as jrd
from loguru import logger
from numpyro.distributions.distribution import enable_validation

from gwkokab.analysis.core.inference_io import PoissonMeanEstimationLoader
from gwkokab.analysis.core.utils import PRNGKeyMixin, to_structured
from gwkokab.analysis.utils.common import read_json
from gwkokab.analysis.utils.regex import match_all
from gwkokab.models.utils import ScaledMixture
from gwkokab.parameters import default_relation_mesh, Parameters as P
from gwkokab.utils.exceptions import LoggedUserWarning, LoggedValueError


class SyntheticEventsBase(PRNGKeyMixin, ABC):
    """Base class for the ``synthetic_events_<family>`` console scripts.

    Subclasses supply the ``parameters`` and ``model_parameters`` properties, which name
    the event coordinates and the population hyper-parameters respectively.

    See :meth:`__init__` for the constructor arguments.
    """

    def __init__(
        self,
        filename: str,
        model_fn: Callable[..., ScaledMixture],
        model_params_filename: str,
        poisson_mean_filename: str,
        n_buffer_events: int,
        derive_parameters: bool = False,
    ) -> None:
        """Read the true hyper-parameters and instantiate the population model.

        Parameters
        ----------
        filename : str
            Destination HDF5 path for the generated population.
        model_fn : Callable[..., ScaledMixture]
            Factory building the population model from its hyper-parameters.
        model_params_filename : str
            Path to a JSON file of true hyper-parameter values, keyed by regex in the same
            way as ``prior_cfg.json``.
        poisson_mean_filename : str
            Path to ``pmean_cfg.json``, which supplies the selection function.
        n_buffer_events : int
            Size of the pool drawn from the model before selection. It must exceed the
            realised number of detections, since events are thinned out of it.
        derive_parameters : bool
            Derive every reachable parameter with
            :func:`~gwkokab.parameters.default_relation_mesh` before saving. Defaults to
            :data:`False`.
        """
        self.filename = filename
        self.poisson_mean_filename = poisson_mean_filename
        self.derive_parameters = derive_parameters
        self.n_buffer_events = n_buffer_events

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate population for a realization via rejection/importance sampling."""
        # Oversample to account for selection effects
        buffer_pop, [buffer_indices] = self.model_fn.sample_with_intermediates(
            self.rng_key, (self.n_buffer_events,)
        )

        buffer_pop = self._ensure_mass_ordering(buffer_pop)

        # Compute selection weights (VT)
        log_selection = log_selection_fn(buffer_pop)
        log_selection = jnp.nan_to_num(
            log_selection, nan=-jnp.inf, posinf=-jnp.inf, neginf=-jnp.inf
        )
        weights = jnn.softmax(log_selection)

        count_nonzero = jnp.sum(weights > 0)
        if count_nonzero < size:
            logger.warning(
                f"Only {count_nonzero} samples have non-zero selection probability. Consider increasing the buffer size.",
                LoggedUserWarning,
            )

        # Resample based on weights
        resample_idx = np.asarray(
            jrd.choice(
                self.rng_key,
                jnp.arange(self.n_buffer_events),
                p=weights,
                shape=(size,),
                replace=False,
            )
        )
        if np.unique(resample_idx).size < size:
            logger.warning(
                "Resampling resulted in duplicate indices. Consider increasing the buffer size.",
                LoggedUserWarning,
            )

        return (
            np.asarray(buffer_pop),
            np.asarray(buffer_indices, np.uint32),
            np.asarray(buffer_pop[resample_idx]),
            np.asarray(buffer_indices[resample_idx]),
            np.asarray(resample_idx, np.uint32),
            np.asarray(weights),
        )

    def from_inverse_transform_sampling(self) -> None:
        """Draw a population and write it out.

        The number of detections is itself random: it is drawn from a Poisson distribution
        whose mean is the expected rate the Poisson mean estimator reports for this model, so
        the synthetic catalogue has the same statistical character as a real one.

        Raises
        ------
        LoggedValueError
            If the drawn size exceeds ``n_buffer_events``, in which case the buffer must be
            enlarged, or if the expected rate is non-positive, which points at a
            misconfigured model or selection function.
        """
        pmean_loader = PoissonMeanEstimationLoader.read_from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        log_selection_fn, poisson_mean_estimator, pmean_kwargs = (
            pmean_loader.get_estimators()
        )

        exp_rate, _ = poisson_mean_estimator(self.model_fn, **pmean_kwargs)
        size = int(jrd.poisson(self.rng_key, exp_rate))

        if size >= self.n_buffer_events:
            raise LoggedValueError(
                f"Number of events to generate are greater than number of buffer events. Number of events to generate: {size}, and number of buffer events: {self.n_buffer_events}. Consider increasing the number of buffer events."
            )

        logger.info(f"Expected rate: {exp_rate:.2f} | Realized size: {size}")

        if size <= 0:
            raise LoggedValueError(
                f"Population size is {size}. Check your VT or model configs."
            )

        buffer_pop, buffer_idx, pop, idx, resample_idx, resample_prob = (
            self._generate_population(size, log_selection_fn)
        )
        self.save_population(
            pop, idx, buffer_pop, buffer_idx, resample_idx, resample_prob
        )

    def save_population(
        self,
        population: np.ndarray,
        indices: np.ndarray,
        buffer_population: np.ndarray,
        buffer_indices: np.ndarray,
        resample_idx: np.ndarray,
        resample_prob: np.ndarray,
    ) -> None:
        """Write the population and its provenance to HDF5.

        When ``derive_parameters`` is set, every parameter reachable from those drawn is
        derived first, so downstream steps can use whichever coordinates they prefer.

        Parameters
        ----------
        population : np.ndarray
            The events that survived selection.
        indices : np.ndarray
            Index of the mixture component each surviving event came from.
        buffer_population : np.ndarray
            The full pool the events were drawn from, before selection.
        buffer_indices : np.ndarray
            Component index of each pooled event.
        resample_idx : np.ndarray
            Positions within the pool that were selected.
        resample_prob : np.ndarray
            Selection weight of each pooled event.
        """
        current_params = self.parameters

        if self.derive_parameters:
            mesh = default_relation_mesh()
            population, current_params = mesh.resolve_from_arrays(
                population, self.parameters
            )
            buffer_population, _ = mesh.resolve_from_arrays(
                buffer_population, self.parameters
            )

        compression_args = {"compression": "gzip", "compression_opts": 9}
        with h5py.File(self.filename, "w") as f:
            f.create_dataset(
                "events",
                data=to_structured(population, current_params),
                **compression_args,
            )
            f.create_dataset(
                "indices", data=indices.astype(np.uint32), **compression_args
            )
            f.create_dataset(
                "buffer_events",
                data=to_structured(buffer_population, current_params),
                **compression_args,
            )
            f.create_dataset(
                "buffer_indices",
                data=buffer_indices.astype(np.uint32),
                **compression_args,
            )
            f.create_dataset(
                "resample_indices",
                data=resample_idx.astype(np.uint32),
                **compression_args,
            )
            f.create_dataset(
                "resample_prob",
                data=resample_prob.astype(np.float32),
                **compression_args,
            )
            f.attrs["parameters"] = np.array(current_params, dtype="S")


def injection_generator_parser() -> ArgumentParser:
    """Build the command line argument parser shared by the population generators.

    Also enables NumPyro distribution argument validation, so a physically impossible
    hyper-parameter in the model parameters file is caught at construction rather than
    producing NaNs.

    Returns
    -------
    ArgumentParser
        A parser carrying the input/output, model and selection-function arguments common
        to every ``synthetic_events_<family>`` script.
    """
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
    proc_group.add_argument(
        "--n-buffer-events",
        default=10_000,
        type=int,
        help="Number of extra events from which few events will be selected based on selection effects.",
    )

    return parser
