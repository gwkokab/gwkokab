# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import gc
import os
from collections.abc import Callable
from typing import Any, Optional

import equinox as eqx
import jax
import numpy as np
from flowMC.nfmodel.base import NFModel
from flowMC.nfmodel.realNVP import RealNVP  # noqa F401
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline  # noqa F401
from flowMC.proposal.base import ProposalBase
from flowMC.proposal.flowHMC import flowHMC  # noqa F401
from flowMC.proposal.Gaussian_random_walk import GaussianRandomWalk  # noqa F401
from flowMC.proposal.HMC import HMC  # noqa F401
from flowMC.proposal.MALA import MALA  # noqa F401
from flowMC.Sampler import Sampler  # noqa F401
from jax import nn as jnn, random as jrd
from jaxtyping import Array


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


class flowMChandler(object):
    r"""Handler class for running flowMC."""

    def __init__(
        self,
        logpdf: Callable,
        local_sampler_kwargs: dict[str, Any],
        nf_model_kwargs: dict[str, Any],
        sampler_kwargs: dict[str, Any],
        data_dump_kwargs: dict[str, Any],
        initial_position: Array,
        data: Optional[dict] = None,
        apply_gradient_checkpoint: bool = False,
        gradient_checkpoint_policy: Optional[Callable[..., bool]] = None,
    ) -> None:
        if apply_gradient_checkpoint:
            self.logpdf = eqx.filter_checkpoint(
                logpdf, policy=gradient_checkpoint_policy
            )
        else:
            self.logpdf = logpdf
        self.local_sampler_kwargs = local_sampler_kwargs
        self.nf_model_kwargs = nf_model_kwargs
        self.sampler_kwargs = sampler_kwargs
        self.data_dump_kwargs = data_dump_kwargs
        self.initial_position = initial_position
        self.data = data

    def make_local_sampler(self) -> ProposalBase:
        """Make a local sampler based on the given arguments.

        Returns
        -------
        ProposalBase
            A local sampler.

        Raises
        ------
        ValueError
            If the sampler is not recognized.
        """
        sampler_name = self.local_sampler_kwargs["sampler"]
        if sampler_name not in ["flowHMC", "GaussianRandomWalk", "HMC", "MALA"]:
            raise ValueError("Invalid sampler")
        del self.local_sampler_kwargs["sampler"]
        return eval(sampler_name)(self.logpdf, **self.local_sampler_kwargs)

    def make_nf_model(self) -> NFModel:
        """Make a normalizing flow model based on the given arguments.

        Returns
        -------
        NFModel
            A normalizing flow model.

        Raises
        ------
        ValueError
            If the model is not recognized
        """
        model_name = self.nf_model_kwargs["model"]
        if model_name not in ["RealNVP", "MaskedCouplingRQSpline"]:
            raise ValueError("Invalid model")
        del self.nf_model_kwargs["model"]
        return eval(model_name)(**self.nf_model_kwargs)

    def make_sampler(self) -> Sampler:
        """Make a sampler based on the given arguments.

        Returns
        -------
        Sampler
            A sampler.
        """
        return Sampler(
            local_sampler=self.make_local_sampler(),
            nf_model=self.make_nf_model(),
            **self.sampler_kwargs,
        )

    def run(
        self,
        debug_nans: bool = False,
        profile_memory: bool = False,
        check_leaks: bool = False,
        file_prefix: Optional[str] = None,
    ) -> None:
        """Run the flowMC sampler and save the data.

        Parameters
        ----------
        debug_nans : bool, optional
            Whether to debug NaNs, by default False
        profile_memory : bool, optional
            Whether to profile memory, by default False
        file_prefix : Optional[str], optional
            Prefix for the file name, by default None
        """
        sampler = self.make_sampler()
        if debug_nans:
            with jax.debug_nans(True):
                sampler.sample(self.initial_position, self.data)
        elif profile_memory:
            sampler.sample(self.initial_position, self.data)

            import datetime

            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"memory_{time}.prof"
            if file_prefix:
                filename = f"{file_prefix}_{filename}"
            jax.profiler.save_device_memory_profile(filename)
        elif check_leaks:
            with jax.checking_leaks():
                sampler.sample(self.initial_position, self.data)
        else:
            sampler.sample(self.initial_position, self.data)
        _save_data_from_sampler(sampler, logpdf=self.logpdf, **self.data_dump_kwargs)
