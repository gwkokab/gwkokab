# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import gc
import os
from collections.abc import Callable
from typing import Any, Dict, Optional

import equinox as eqx
import jax
import numpy as np
from flowMC.resource.buffers import Buffer
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.local_kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.local_kernel.HMC import HMC
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.nf_model.base import NFModel
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.nf_model.realNVP import RealNVP
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.resource_strategy_bundles import ResourceStrategyBundle
from flowMC.Sampler import Sampler
from flowMC.strategy.global_tuning import LocalGlobalNFSample
from jax import nn as jnn, random as jrd
from jaxtyping import Array, Float, PRNGKeyArray


__all__ = ["run_flowMC"]


def _same_length_arrays(length: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """This function pads the arrays with None to make them the same length.

    Parameters
    ----------
    length : int
        The length of the arrays.
    arrays : np.ndarray
        The arrays to pad.

    Returns
    -------
    tuple[np.ndarray, ...]
        The padded arrays.
    """
    padded_arrays = []
    for array in arrays:
        padded_array = np.empty((length,))
        padded_array[..., : array.shape[0]] = array
        padded_array[..., array.shape[0] :] = None
        padded_arrays.append(padded_array)
    return tuple(padded_arrays)


def _save_data_from_sampler(
    sampler: Sampler,
    *,
    out_dir: str,
    labels: Optional[list[str]] = None,
    n_samples: int = 5000,
    logpdf: Optional[Callable] = None,
    batch_size: int = 1000,
) -> None:
    """This functions saves the data from a sampler to disk. The data saved includes the
    samples from the flow, the chains from the training and production phases, the log
    probabilities, the global and local acceptance rates, and the loss values.

    Parameters
    ----------
    sampler : Sampler
        The sampler object.
    out_dir : str
        The output directory.
    labels : Optional[list[str]], optional
        list of labels for the samples, by default None
    n_samples : int, optional
        number of samples to draw from the flow, by default 5000
    logpdf : Callable, optional
        The log probability function.
    batch_size : int, optional
        The batch size for the log probability function, by default 1000
    """
    if labels is None:
        labels = [f"x{i}" for i in range(sampler.n_dim)]
    if logpdf is None:
        raise ValueError("logpdf must be provided")

    os.makedirs(out_dir, exist_ok=True)

    header = " ".join(labels)

    out_train = sampler.get_sampler_state(training=True)

    train_chains = np.array(out_train["chains"])
    train_global_accs = np.array(out_train["global_accs"])
    train_local_accs = np.array(out_train["local_accs"])
    train_loss_vals = np.array(out_train["loss_vals"])
    train_log_prob = np.array(out_train["log_prob"])

    out_prod = sampler.get_sampler_state(training=False)

    prod_chains = np.array(out_prod["chains"])
    prod_global_accs = np.array(out_prod["global_accs"])
    prod_local_accs = np.array(out_prod["local_accs"])
    prod_log_prob = np.array(out_prod["log_prob"])

    n_chains = sampler.n_chains

    np.savetxt(
        rf"{out_dir}/global_accs.dat",
        np.column_stack(
            _same_length_arrays(
                max(train_global_accs.shape[1], prod_global_accs.shape[1]),
                train_global_accs.mean(0),
                prod_global_accs.mean(0),
            )
        ),
        header="train prod",
        comments="#",
    )
    np.savetxt(
        rf"{out_dir}/local_accs.dat",
        np.column_stack(
            _same_length_arrays(
                max(train_local_accs.shape[1], prod_local_accs.shape[1]),
                train_local_accs.mean(0),
                prod_local_accs.mean(0),
            )
        ),
        header="train prod",
        comments="#",
    )

    np.savetxt(rf"{out_dir}/loss.dat", train_loss_vals.reshape(-1), header="loss")

    for i in range(n_chains):
        np.savetxt(
            rf"{out_dir}/train_chains_{i}.dat",
            train_chains[i, :, :],
            header=header,
        )
        np.savetxt(
            rf"{out_dir}/prod_chains_{i}.dat",
            prod_chains[i, :, :],
            header=header,
        )
        np.savetxt(
            rf"{out_dir}/log_prob_{i}.dat",
            np.column_stack(
                _same_length_arrays(
                    max(train_log_prob[i].shape[0], prod_log_prob[i].shape[0]),
                    train_log_prob[i],
                    prod_log_prob[i],
                )
            ),
            header="train prod",
            comments="#",
        )

    gc.collect()

    key_unweighted, key_weighted = jrd.split(
        jrd.PRNGKey(np.random.randint(1, 2**32 - 1))
    )

    samples = np.asarray(
        sampler.sample_flow(n_samples=n_samples, rng_key=key_unweighted)
    )
    np.savetxt(rf"{out_dir}/nf_samples_unweighted.dat", samples, header=header)

    gc.collect()

    samples = sampler.sample_flow(n_samples=n_samples + 50_000, rng_key=key_weighted)
    weights = np.asarray(
        jnn.softmax(
            jax.lax.map(lambda s: logpdf(s, None), samples, batch_size=batch_size)
            - sampler.nf_model.log_prob(samples)
        )
    )
    samples = np.asarray(samples)
    samples = samples[np.random.choice(n_samples + 50_000, size=n_samples, p=weights)]
    np.savetxt(rf"{out_dir}/nf_samples_weighted.dat", samples, header=header)

    gc.collect()


class _flowMCResourceBundle(ResourceStrategyBundle):
    def __repr__(self):
        return "Local Global NF Sampling"

    def __init__(
        self,
        rng_key: PRNGKeyArray,
        n_chains: int,
        n_dims: int,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float[Array, ""]],
        local_sampler_kwargs: Dict[str, Any],
        nf_model_kwargs: Dict[str, Any],
        n_local_steps: int,
        n_global_steps: int,
        n_loop_training: int,
        n_loop_production: int,
        n_epochs: int,
        n_flow_sample: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 10000,
        verbose: bool = False,
    ):
        n_training_steps = (
            n_local_steps * n_loop_training + n_global_steps * n_loop_training
        )
        n_production_steps = (
            n_local_steps * n_loop_production + n_global_steps * n_loop_production
        )
        n_total_epochs = n_loop_training * n_epochs

        positions_training = Buffer(
            "positions_training", (n_chains, n_training_steps, n_dims), 1
        )
        log_prob_training = Buffer("log_prob_training", (n_chains, n_training_steps), 1)
        local_accs_training = Buffer(
            "local_accs_training", (n_chains, n_training_steps), 1
        )
        global_accs_training = Buffer(
            "global_accs_training", (n_chains, n_training_steps), 1
        )
        loss_buffer = Buffer("loss_buffer", (n_total_epochs,), 0)

        position_production = Buffer(
            "positions_production", (n_chains, n_production_steps, n_dims), 1
        )
        log_prob_production = Buffer(
            "log_prob_production", (n_chains, n_production_steps), 1
        )
        local_accs_production = Buffer(
            "local_accs_production", (n_chains, n_production_steps), 1
        )
        global_accs_production = Buffer(
            "global_accs_production", (n_chains, n_production_steps), 1
        )

        local_sampler = self._get_local_sampler(**local_sampler_kwargs)
        rng_key, subkey = jax.random.split(rng_key)
        model = self._get_nf_model(key=subkey, n_features=n_dims, **nf_model_kwargs)
        global_sampler = NFProposal(model, n_flow_sample=n_flow_sample)
        optimizer = Optimizer(model=model, learning_rate=learning_rate)
        logpdf = LogPDF(logpdf, n_dims=n_dims)

        self.resources = {
            "logpdf": logpdf,
            "positions_training": positions_training,
            "log_prob_training": log_prob_training,
            "local_accs_training": local_accs_training,
            "global_accs_training": global_accs_training,
            "loss_buffer": loss_buffer,
            "positions_production": position_production,
            "log_prob_production": log_prob_production,
            "local_accs_production": local_accs_production,
            "global_accs_production": global_accs_production,
            "local_sampler": local_sampler,
            "global_sampler": global_sampler,
            "model": model,
            "optimizer": optimizer,
        }

        self.strategies = {
            "training_sampler": LocalGlobalNFSample(
                "logpdf",
                "local_sampler",
                "global_sampler",
                ["positions_training", "log_prob_training", "local_accs_training"],
                ["model", "positions_training", "optimizer"],
                ["positions_training", "log_prob_training", "global_accs_training"],
                n_local_steps,
                n_global_steps,
                n_loop_training,
                n_epochs,
                loss_buffer_name="loss_buffer",
                batch_size=batch_size,
                n_max_examples=n_max_examples,
                training=True,
                verbose=verbose,
            ),
            "production_sampler": LocalGlobalNFSample(
                "logpdf",
                "local_sampler",
                "global_sampler",
                [
                    "positions_production",
                    "log_prob_production",
                    "local_accs_production",
                ],
                ["model", "positions_production", "optimizer"],
                [
                    "positions_production",
                    "log_prob_production",
                    "global_accs_production",
                ],
                n_local_steps,
                n_global_steps,
                n_loop_production,
                n_epochs,
                batch_size=batch_size,
                n_max_examples=n_max_examples,
                training=False,
                verbose=verbose,
            ),
        }
        self.strategy_order = ["training_sampler", "production_sampler"]

    @staticmethod
    def _get_local_sampler(sampler: str = "MALA", **kwargs) -> ProposalBase:
        """Make a local sampler based on the given arguments.

        Parameters
        ----------
        sampler : str, optional
            The name of the local sampler, by default "MALA"

        Returns
        -------
        ProposalBase
            A local sampler.
        """
        assert isinstance(sampler, str), "Local Sampler name must be a string"
        assert len(sampler) > 0, "Local Sampler name must be a non-empty string"
        assert sampler in [
            "GaussianRandomWalk",
            "HMC",
            "MALA",
        ], f"{sampler} is not a valid local sampler"

        if sampler == "MALA":
            step_size = kwargs.get("step_size")

            assert step_size is not None, "step_size must be provided for MALA"

            return MALA(step_size=step_size)

        if sampler == "HMC":
            condition_matrix = kwargs.get("condition_matrix")
            step_size = kwargs.get("step_size")
            n_leapfrog = kwargs.get("n_leapfrog")

            assert condition_matrix is not None, (
                "condition_matrix must be provided for HMC"
            )
            assert step_size is not None, "step_size must be provided for HMC"
            assert n_leapfrog is not None, "n_leapfrog must be provided for HMC"

            return HMC(
                condition_matrix=condition_matrix,
                step_size=step_size,
                n_leapfrog=n_leapfrog,
            )

        if sampler == "GaussianRandomWalk":
            step_size = kwargs.get("step_size")

            assert step_size is not None, (
                "step_size must be provided for GaussianRandomWalk"
            )

            return GaussianRandomWalk(step_size=step_size)

    @staticmethod
    def _get_nf_model(
        key: PRNGKeyArray, model: str = "MaskedCouplingRQSpline", **kwargs
    ) -> NFModel:
        """Make a normalizing flow model based on the given arguments.

        Parameters
        ----------
        key : PRNGKeyArray
            Pseudo-random number generator key.
        model : str, optional
            The name of the normalizing flow model, by default "MaskedCouplingRQSpline"

        Returns
        -------
        NFModel
            A normalizing flow model.
        """
        assert isinstance(model, str), "NFModel name must be a string"
        assert len(model) > 0, "NFModel name must be a non-empty string"
        assert model in [
            "RealNVP",
            "MaskedCouplingRQSpline",
        ], f"{model} is not a valid normalizing flow model"

        if model == "RealNVP":
            n_features = kwargs.get("n_features")
            n_hidden = kwargs.get("n_hidden")
            n_layers = kwargs.get("n_layers")

            assert n_features is not None, "n_features must be provided for RealNVP"
            assert n_hidden is not None, "n_hidden must be provided for RealNVP"
            assert n_layers is not None, "n_layers must be provided for RealNVP"

            assert isinstance(n_features, int), "n_features must be an integer"
            assert isinstance(n_hidden, int), "n_hidden must be an integer"
            assert isinstance(n_layers, int), "n_layers must be an integer"

            assert n_features > 0, "n_features must be greater than 0"
            assert n_hidden > 0, "n_hidden must be greater than 0"
            assert n_layers > 0, "n_layers must be greater than 0"

            del kwargs["n_features"]
            del kwargs["n_hidden"]
            del kwargs["n_layers"]

            return RealNVP(
                n_features=n_features,
                n_layers=n_layers,
                n_hidden=n_hidden,
                key=key,
                **kwargs,
            )

        if model == "MaskedCouplingRQSpline":
            num_bins = kwargs.get("num_bins")
            n_layers = kwargs.get("n_layers")
            hidden_size = kwargs.get("hidden_size")
            n_features = kwargs.get("n_features")

            assert n_features is not None, (
                "n_features must be provided for MaskedCouplingRQSpline"
            )
            assert num_bins is not None, (
                "num_bins must be provided for MaskedCouplingRQSpline"
            )
            assert n_layers is not None, (
                "n_layers must be provided for MaskedCouplingRQSpline"
            )
            assert hidden_size is not None, (
                "hidden_units must be provided for MaskedCouplingRQSpline"
            )

            assert isinstance(n_features, int), "n_features must be an integer"
            assert isinstance(num_bins, int), "num_bins must be an integer"
            assert isinstance(n_layers, int), "n_layers must be an integer"
            assert isinstance(hidden_size, list), "hidden_units must be a list"
            assert all(isinstance(i, int) for i in hidden_size), (
                "hidden_units must be a list of integers"
            )

            assert n_features > 0, "n_features must be greater than 0"
            assert num_bins > 0, "num_bins must be greater than 0"
            assert n_layers > 0, "n_layers must be greater than 0"

            assert len(hidden_size) > 0, "hidden_units must be a list of integers"
            assert all(i > 0 for i in hidden_size), (
                "hidden_units must be a list of integers greater than 0"
            )

            del kwargs["num_bins"]
            del kwargs["n_layers"]
            del kwargs["hidden_units"]
            del kwargs["n_features"]

            return MaskedCouplingRQSpline(
                n_features=n_features,
                n_layers=n_layers,
                hidden_size=hidden_size,
                num_bins=num_bins,
                key=key,
                **kwargs,
            )


def run_flowMC(
    rng_key: PRNGKeyArray,
    logpdf: Callable,
    local_sampler_kwargs: dict[str, Any],
    nf_model_kwargs: dict[str, Any],
    sampler_kwargs: dict[str, Any],
    data_dump_kwargs: dict[str, Any],
    initial_position: Array,
    data: Optional[dict] = None,
    apply_gradient_checkpoint: bool = False,
    gradient_checkpoint_policy: Optional[Callable[..., bool]] = None,
    debug_nans: bool = False,
    profile_memory: bool = False,
    check_leaks: bool = False,
    file_prefix: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Convenience function to run flowMC with the given parameters. This function
    initializes the flowMC resource bundle and sampler, and runs the sampler with the
    given initial position and data. The function also handles gradient checkpointing,
    debugging, and profiling options.

    Parameters
    ----------
    rng_key : PRNGKeyArray
        Pseudo-random number generator key.
    logpdf : Callable
        The log probability function.
    local_sampler_kwargs : dict[str, Any]
        Keyword arguments for the local sampler.
    nf_model_kwargs : dict[str, Any]
        Keyword arguments for the normalizing flow model.
    sampler_kwargs : dict[str, Any]
        Keyword arguments for the sampler.
    data_dump_kwargs : dict[str, Any]
        Keyword arguments for the data dump.
    initial_position : Array
        The initial position for the sampler.
    data : Optional[dict], optional
        Additional data to pass to the log probability function, by default None
    apply_gradient_checkpoint : bool, optional
        Whether to apply gradient checkpointing, by default False
    gradient_checkpoint_policy : Optional[Callable[..., bool]], optional
        The policy for gradient checkpointing, by default None
    debug_nans : bool, optional
        Whether to enable JAX debug mode for NaNs, by default False
    profile_memory : bool, optional
        Whether to profile memory usage, by default False
    check_leaks : bool, optional
        Whether to check for memory leaks, by default False
    file_prefix : Optional[str], optional
        Prefix for the output files, by default None
    verbose : bool, optional
        Whether to enable verbose output, by default False
    """
    if apply_gradient_checkpoint:
        logpdf = eqx.filter_checkpoint(logpdf, policy=gradient_checkpoint_policy)  # type: ignore

    n_dim = initial_position.shape[1]
    n_chains = sampler_kwargs.pop("n_chains")

    resource_bundle = _flowMCResourceBundle(
        rng_key=rng_key,
        n_dims=n_dim,
        n_chains=n_chains,
        logpdf=logpdf,
        local_sampler_kwargs=local_sampler_kwargs,
        nf_model_kwargs=nf_model_kwargs,
        verbose=verbose,
        **sampler_kwargs,
    )

    rng_key, subkey = jax.random.split(rng_key)

    sampler = Sampler(
        rng_key=subkey,
        n_dim=n_dim,
        n_chains=n_chains,
        resource_strategy_bundles=resource_bundle,
    )

    if debug_nans:
        with jax.debug_nans(True):
            sampler.sample(initial_position, data)
    elif profile_memory:
        sampler.sample(initial_position, data)

        import datetime

        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"memory_{time}.prof"
        if file_prefix:
            filename = f"{file_prefix}_{filename}"
        jax.profiler.save_device_memory_profile(filename)
    elif check_leaks:
        with jax.checking_leaks():
            sampler.sample(initial_position, data)
    else:
        sampler.sample(initial_position, data)
    _save_data_from_sampler(sampler, logpdf=logpdf, **data_dump_kwargs)
