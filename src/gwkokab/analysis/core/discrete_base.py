# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0

"""The *discrete* data representation: events as clouds of posterior samples.

:class:`DiscreteBase` supplies :meth:`~DiscreteBase.read_data` and
:meth:`~DiscreteBase.run` for analyses whose events each come with a set of parameter
estimation samples. Because events carry different numbers of samples, they are grouped
into buckets of similar size by :mod:`~gwkokab.analysis.utils.jenks` and padded into
rectangular arrays, with a mask marking the real entries -- one ``jit`` compilation per
bucket instead of one per event, without wasting memory on the largest event's shape.

See Also
--------
gwkokab.analysis.core.analytical_gwalk_base :
    The alternative representation, where each event is a Gaussian summary.
"""

import warnings
from argparse import ArgumentParser
from collections.abc import Callable
from typing import List, Optional, Tuple, Union

import jax
import numpy as np
from jaxtyping import Array
from loguru import logger
from numpyro.distributions import Distribution

from gwkokab.analysis.core.inference_io import (
    DiscretePELoader,
    PoissonMeanEstimationLoader,
)
from gwkokab.parameters import Parameters as P
from gwkokab.utils.exceptions import LoggedUserWarning, LoggedValueError

from ..utils.jenks import pad_and_stack
from .analysis_base import AnalysisBase


class DiscreteBase(AnalysisBase):
    """Data-representation base for analyses over per-event posterior samples.

    Mixed with a model family's ``Core`` class, which supplies ``parameters`` and
    ``model_parameters``, and with a sampler mixin, which supplies ``driver``.

    See :meth:`__init__` for the constructor arguments.
    """

    def __init__(
        self,
        likelihood_fn: Callable[..., Callable[..., Array]],
        model: Union[Distribution, Callable[..., Distribution]],
        where_fns: Optional[List[Callable[..., Array]]],
        data_loader: DiscretePELoader,
        prior_filename: str,
        poisson_mean_filename: str,
        sampler_cfg,
        variance_cut_threshold: float | None,
        n_buckets: Optional[int],
        threshold: float,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
        analysis_name: str = "",
    ) -> None:
        """Configure a discrete analysis.

        Parameters
        ----------
        likelihood_fn : Callable[..., Callable[..., Array]]
            Likelihood factory, as returned by
            :func:`~gwkokab.inference.factory.get_likelihood_fn`.
        model : Union[Distribution, Callable[..., Distribution]]
            The population model, or a factory that builds it from hyper-parameters.
        where_fns : Optional[List[Callable[..., Array]]]
            Extra validity predicates on the hyper-parameters, beyond the prior support.
        data_loader : DiscretePELoader
            The parsed ``data_loader_cfg.json``: which event files to read and how to undo
            the PE prior.
        prior_filename : str
            Path to ``prior_cfg.json``.
        poisson_mean_filename : str
            Path to ``pmean_cfg.json``.
        sampler_cfg : SamplerConfig
            The parsed sampler configuration.
        variance_cut_threshold : float | None
            Threshold above which a noisy Poisson mean estimate is penalised.
        n_buckets : Optional[int]
            Number of buckets to partition events into. :data:`None` lets the Jenks natural
            breaks algorithm choose.
        threshold : float
            Goodness-of-variance-fit threshold, in percent, for automatic bucket selection.
        debug_nans : bool
            Run the sampler under :func:`jax.debug_nans`. Defaults to :data:`False`.
        profile_memory : bool
            Write a memory profile of the run. Defaults to :data:`False`.
        check_leaks : bool
            Run the sampler under :func:`jax.checking_leaks`. Defaults to :data:`False`.
        analysis_name : str
            Name of the analysis. Defaults to ``""``.

        Raises
        ------
        LoggedValueError
            If ``analysis_name`` contains anything but letters and underscores.
        """
        if not (all(letter.isalpha() or letter == "_" for letter in analysis_name)):
            raise LoggedValueError(
                "Analysis name must contain only letters and underscores.",
            )

        self.likelihood_fn = likelihood_fn
        self.n_buckets = n_buckets
        self.data_loader = data_loader
        self.threshold = threshold
        self.where_fns = where_fns

        logger.info(
            f"Initializing DiscreteBase class for analysis identifier: '{analysis_name}'"
        )

        super().__init__(
            analysis_name=analysis_name,
            check_leaks=check_leaks,
            debug_nans=debug_nans,
            model=model,
            poisson_mean_filename=poisson_mean_filename,
            prior_filename=prior_filename,
            profile_memory=profile_memory,
            sampler_cfg=sampler_cfg,
            variance_cut_threshold=variance_cut_threshold,
        )

    def read_data(
        self,
    ) -> Tuple[Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...]]:
        """Load the per-event posterior samples and bucket them.

        Events are partitioned into buckets of similar sample count and padded into
        rectangular arrays. Padded entries have their log PE prior zeroed and are marked in
        the masks, so the likelihood can exclude them. The result is staged onto the JAX
        devices before returning.

        Returns
        -------
        Tuple[Tuple[Array, ...], Tuple[Array, ...], Tuple[Array, ...]]
            The bucketed samples, their log PE priors, and the masks marking real entries.

        Raises
        ------
        LoggedValueError
            If the loader returns different numbers of events and log PE priors.

        Warns
        -----
        LoggedUserWarning
            If the requested ``n_buckets`` differs from what the partitioning produced, in
            which case the latter wins.
        """
        parameters = [p.value if isinstance(p, P) else p for p in self.parameters]

        data, log_ref_priors = self.data_loader.load(parameters, self.seed)

        if len(data) != len(log_ref_priors):
            raise LoggedValueError(
                "Number of data events does not match number of log reference priors.",
            )

        logger.info("Commencing data partitioning into buckets.")
        _data_group, _log_ref_priors_group, _masks_group = pad_and_stack(
            data, log_ref_priors, n_buckets=self.n_buckets, threshold=self.threshold
        )

        if self.n_buckets is None:
            self.n_buckets = len(_data_group)
            logger.info(
                f"Automatic bucket determination completed. Optimal buckets: {self.n_buckets}"
            )
        elif self.n_buckets != len(_data_group):
            overridden_buckets = len(_data_group)
            warnings.warn(
                f"Specified n_buckets ({self.n_buckets}) differs from partitioning results. "
                f"Overriding to {overridden_buckets} buckets for computational alignment.",
                LoggedUserWarning,
            )
            self.n_buckets = overridden_buckets

        for i in range(self.n_buckets):
            _log_ref_priors_group[i] = np.where(  # type: ignore
                _masks_group[i], _log_ref_priors_group[i], 0.0
            )

        _data_group = tuple(_data_group)
        _log_ref_priors_group = tuple(_log_ref_priors_group)
        _masks_group = tuple(_masks_group)

        # Monitor device placement
        primary_devices = jax.devices()
        logger.info(f"Staging data groups to JAX devices: {primary_devices}")

        data_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_data_group)
        )
        log_ref_priors_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_log_ref_priors_group)
        )
        masks_group: Tuple[Array, ...] = jax.block_until_ready(
            jax.device_put(_masks_group)
        )

        for i, group in enumerate(data_group):
            mask_count = np.sum(_masks_group[i] == 0)
            logger.debug(
                f"Bucket {i}: Shape {group.shape} | Padding elements: {mask_count}"
            )

        return data_group, log_ref_priors_group, masks_group

    def run(self) -> None:
        """Run the analysis end to end.

        Classifies the model parameters, reads and buckets the data, builds the Poisson
        mean estimator and the log posterior, and hands them to the sampler mixin's
        ``driver``.
        """
        model_name = getattr(self.model, "__name__", str(self.model))
        logger.info(f"Starting inference pipeline for model: {model_name}")

        constants, priors, variables, variables_index = self.classify_model_parameters()
        logger.debug(
            f"Parameter classification: {len(variables)} variables, {len(constants)} constants."
        )

        data_group, log_ref_priors_group, masks_group = self.read_data()

        logger.info("Parsing Poisson mean configuration and initializing estimator.")
        pmean_loader = PoissonMeanEstimationLoader.read_from_json(
            self.poisson_mean_filename, self.rng_key, self.parameters
        )
        _, poisson_mean_estimator, pmean_kwargs = pmean_loader.get_estimators()

        logger.info(
            "Constructing likelihood function and preparing for sampler execution."
        )
        logpdf = self.likelihood_fn(
            dist_fn=self.model,
            priors=priors,
            variables=variables,
            variables_index=variables_index,
            poisson_mean_estimator=poisson_mean_estimator,
            where_fns=self.where_fns,
            constants=constants,  # type: ignore
            variance_cut_threshold=self.variance_cut_threshold,
        )
        logger.success("Likelihood function construction completed successfully.")

        N_pes = tuple([
            np.asarray(np.count_nonzero(batched_masks, axis=-1), dtype=int)
            for batched_masks in masks_group
        ])
        logger.info(f"Event counts per bucket (N_pe): {N_pes}")

        logger.info(
            "Initiating sampler execution with prepared likelihood and data groups."
        )
        self.driver(
            logpdf=logpdf,
            priors=priors,
            data={
                "data_group": data_group,
                "log_ref_priors_group": log_ref_priors_group,
                "masks_group": masks_group,
                "pmean_kwargs": pmean_kwargs,
                "N_pes": N_pes,
            },
            labels=sorted(variables.keys()),
        )


def discrete_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    """Populate the command line argument parser with the arguments for the discrete
    analyses.

    Adds ``--data-loader-cfg``, plus the ``--n-buckets`` and ``--threshold`` knobs that
    control how events are partitioned.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to add the arguments to.

    Returns
    -------
    ArgumentParser
        The same parser, with the arguments added.
    """
    discrete_group = parser.add_argument_group("DiscreteBase Configuration")
    discrete_group.add_argument(
        "--data-loader-cfg",
        type=str,
        required=True,
        help="Path to JSON configuration for the DiscreteParameterEstimationLoader.",
    )

    optm_group = parser.add_argument_group("Performance Tuning Options")
    optm_group.add_argument(
        "--n-buckets",
        help="Manually specify the number of data buckets for memory management. "
        "See https://github.com/kokabsc/gwkokab/issues/568 for more details.",
        type=int,
        default=None,
    )
    optm_group.add_argument(
        "--threshold",
        help="Threshold (0-100) for automatic bucket optimization via Jenks natural breaks.",
        type=float,
        default=3.0,
    )

    return parser
