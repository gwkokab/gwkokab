# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
import gc
import os
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import jax
import numpy as np
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
from flowMC.Sampler import Sampler
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, PRNGKeyArray
from loguru import logger

from gwkokab.models.utils import JointDistribution
from gwkokab.utils.tools import error_if
from kokab.utils.common import read_json
from kokab.utils.guru import Guru, guru_arg_parser as guru_parser


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
    rng_key: PRNGKeyArray,
    out_dir: str,
    labels: Optional[list[str]] = None,
    n_samples: int = 5000,
    logpdf: Optional[Callable] = None,
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
    """
    logger.info("Saving data from sampler to directory: {out_dir}", out_dir=out_dir)
    if labels is None:
        labels = [f"x{i}" for i in range(sampler.n_dim)]
    if logpdf is None:
        raise ValueError("logpdf must be provided")

    os.makedirs(out_dir, exist_ok=True)

    header = " ".join(labels)

    sampler_resources = sampler.resources

    train_chains = np.array(sampler_resources["positions_training"].data)
    train_global_accs = np.array(sampler_resources["global_accs_training"].data)
    train_local_accs = np.array(sampler_resources["local_accs_training"].data)
    train_loss_vals = np.array(sampler_resources["loss_buffer"].data)
    train_log_prob = np.array(sampler_resources["log_prob_training"].data)

    prod_chains = np.array(sampler_resources["positions_production"].data)
    prod_global_accs = np.array(sampler_resources["global_accs_production"].data)
    prod_local_accs = np.array(sampler_resources["local_accs_production"].data)
    prod_log_prob = np.array(sampler_resources["log_prob_production"].data)

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
    logger.info(
        "Global acceptance rates saved to {out_dir}/global_accs.dat", out_dir=out_dir
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
    logger.info(
        "Local acceptance rates saved to {out_dir}/local_accs.dat", out_dir=out_dir
    )

    for n_chain in range(n_chains):
        np.savetxt(
            rf"{out_dir}/global_accs_{n_chain}.dat",
            np.column_stack(
                _same_length_arrays(
                    max(
                        train_global_accs[n_chain, :].shape[0],
                        prod_global_accs[n_chain, :].shape[0],
                    ),
                    train_global_accs[n_chain, :],
                    prod_global_accs[n_chain, :],
                )
            ),
            header="train prod",
            comments="#",
        )
        logger.info(
            "Saving chain {n_chain} global acceptance rates to {out_dir}/global_accs_{n_chain}.dat",
            n_chain=n_chain,
            out_dir=out_dir,
        )
        np.savetxt(
            rf"{out_dir}/local_accs_{n_chain}.dat",
            np.column_stack(
                _same_length_arrays(
                    max(
                        train_local_accs[n_chain, :].shape[0],
                        prod_local_accs[n_chain, :].shape[0],
                    ),
                    train_local_accs[n_chain, :],
                    prod_local_accs[n_chain, :],
                )
            ),
            header="train prod",
            comments="#",
        )
        logger.info(
            "Saving chain {n_chain} local acceptance rates to {out_dir}/local_accs_{n_chain}.dat",
            n_chain=n_chain,
            out_dir=out_dir,
        )

    np.savetxt(rf"{out_dir}/loss.dat", train_loss_vals.reshape(-1), header="loss")
    logger.info("Loss values saved to {out_dir}/loss.dat", out_dir=out_dir)

    for i in range(n_chains):
        logger.info(
            "Saving chain {i} data to {out_dir}/train_chains_{i}.dat and {out_dir}/prod_chains_{i}.dat",
            i=i,
            out_dir=out_dir,
        )
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
    logger.debug(
        "Garbage collection completed after saving chains and log probabilities."
    )

    nf_model = sampler_resources["model"]
    _, subkey = jrd.split(rng_key)
    unweighted_samples = np.asarray(
        jax.block_until_ready(nf_model.sample(n_samples=n_samples, rng_key=subkey))
    )
    np.savetxt(
        rf"{out_dir}/nf_samples_unweighted.dat", unweighted_samples, header=header
    )
    logger.info(
        "Unweighted samples saved to {out_dir}/nf_samples_unweighted.dat",
        out_dir=out_dir,
    )

    gc.collect()
    logger.debug("Garbage collection completed after unweighted samples.")

    logpdf_val = jax.block_until_ready(
        jax.lax.map(lambda s: logpdf(s), unweighted_samples)
    )
    logger.debug(
        "Nan count in logpdf values: {nan_count}", nan_count=jnp.isnan(logpdf_val).sum()
    )
    logger.debug(
        "Neginf count in logpdf values: {inf_count}",
        inf_count=jnp.isneginf(logpdf_val).sum(),
    )
    logger.debug(
        "Posinf count in logpdf values: {inf_count}",
        inf_count=jnp.isposinf(logpdf_val).sum(),
    )

    nf_model_log_prob_val = jax.block_until_ready(
        jax.lax.map(nf_model.log_prob, unweighted_samples)
    )
    logger.debug(
        "Nan count in nf_model_log_prob values: {nan_count}",
        nan_count=jnp.isnan(nf_model_log_prob_val).sum(),
    )
    logger.debug(
        "Neginf count in nf_model_log_prob values: {inf_count}",
        inf_count=jnp.isneginf(nf_model_log_prob_val).sum(),
    )
    logger.debug(
        "Posinf count in nf_model_log_prob values: {inf_count}",
        inf_count=jnp.isposinf(nf_model_log_prob_val).sum(),
    )

    weights = jax.nn.softmax(logpdf_val - nf_model_log_prob_val)
    logger.debug(
        "Nan count in weights values: {nan_count}", nan_count=jnp.isnan(weights).sum()
    )
    logger.debug(
        "Neginf count in weights values: {inf_count}",
        inf_count=jnp.isneginf(weights).sum(),
    )
    logger.debug(
        "Posinf count in weights values: {inf_count}",
        inf_count=jnp.isposinf(weights).sum(),
    )

    weights = np.asarray(weights)  # type: ignore
    ess = int(1.0 / np.sum(np.square(weights)))
    logger.debug("Effective sample size is {} out of {}".format(ess, n_samples))
    if ess == 0:
        raise ValueError("Effective sample size is zero, cannot proceed with sampling.")
    weighted_samples = unweighted_samples[
        np.random.choice(n_samples, size=ess, p=weights)
    ]
    np.savetxt(rf"{out_dir}/nf_samples_weighted.dat", weighted_samples, header=header)
    logger.info(
        "Weighted samples saved to {out_dir}/nf_samples_weighted.dat", out_dir=out_dir
    )

    gc.collect()
    logger.debug("Garbage collection completed after weighted samples.")


class FlowMCBased(Guru):
    def driver(
        self,
        *,
        logpdf: Callable[[Array, Dict[str, Any]], Array],
        priors: JointDistribution,
        data: Dict[str, Any],
        labels: List[str],
    ) -> None:
        sampler_config = read_json(self.sampler_settings_filename)

        RQSpline_MALA_Bundle_value = sampler_config.pop("RQSpline_MALA_Bundle", {})

        batch_size = RQSpline_MALA_Bundle_value["batch_size"]
        chain_batch_size = RQSpline_MALA_Bundle_value["chain_batch_size"]
        global_thinning = RQSpline_MALA_Bundle_value["global_thinning"]
        learning_rate = RQSpline_MALA_Bundle_value["learning_rate"]
        local_thinning = RQSpline_MALA_Bundle_value["local_thinning"]
        mala_step_size = RQSpline_MALA_Bundle_value["mala_step_size"]
        n_chains = RQSpline_MALA_Bundle_value["n_chains"]
        n_epochs = RQSpline_MALA_Bundle_value["n_epochs"]
        n_global_steps = RQSpline_MALA_Bundle_value["n_global_steps"]
        n_local_steps = RQSpline_MALA_Bundle_value["n_local_steps"]
        n_max_examples = RQSpline_MALA_Bundle_value["n_max_examples"]
        n_NFproposal_batch_size = RQSpline_MALA_Bundle_value["n_NFproposal_batch_size"]
        n_production_loops = RQSpline_MALA_Bundle_value["n_production_loops"]
        n_training_loops = RQSpline_MALA_Bundle_value["n_training_loops"]
        rq_spline_hidden_units = RQSpline_MALA_Bundle_value["rq_spline_hidden_units"]
        rq_spline_n_bins = RQSpline_MALA_Bundle_value["rq_spline_n_bins"]
        rq_spline_n_layers = RQSpline_MALA_Bundle_value["rq_spline_n_layers"]
        verbose = RQSpline_MALA_Bundle_value["verbose"]

        logger.debug("Validation for Sampler parameters starting")

        for var, var_name, expected_type in (
            (batch_size, "batch_size", int),
            (chain_batch_size, "chain_batch_size", int),
            (global_thinning, "global_thinning", int),
            (learning_rate, "learning_rate", (int, float)),
            (local_thinning, "local_thinning", int),
            (mala_step_size, "mala_step_size", (int, float)),
            (n_chains, "n_chains", int),
            (n_epochs, "n_epochs", int),
            (n_global_steps, "n_global_steps", int),
            (n_local_steps, "n_local_steps", int),
            (n_max_examples, "n_max_examples", int),
            (n_NFproposal_batch_size, "n_NFproposal_batch_size", int),
            (n_production_loops, "n_production_loops", int),
            (n_training_loops, "n_training_loops", int),
            (rq_spline_hidden_units, "rq_spline_hidden_units", list),
            (rq_spline_n_bins, "rq_spline_n_bins", int),
            (rq_spline_n_layers, "rq_spline_n_layers", int),
            (verbose, "verbose", bool),
        ):
            error_if(
                not isinstance(var, expected_type),
                err=TypeError,
                msg=f"expected a {expected_type}, got {type(var)} for {var_name}",
            )
            if expected_type is int:
                error_if(
                    var < 1 and var_name != "chain_batch_size",
                    err=ValueError,
                    msg=f"expected a positive integer, got {var} for {var_name}",
                )
            logger.debug(f"{var_name}: {var}")
        error_if(
            not all(isinstance(x, int) and x > 0 for x in rq_spline_hidden_units),
            msg=f"expected a list of positive integers, got {rq_spline_hidden_units} for rq_spline_hidden_units",
        )

        initial_position = priors.sample(self.rng_key, (n_chains,))

        n_dims = initial_position.shape[1]

        bundle = RQSpline_MALA_Bundle(
            rng_key=self.rng_key,
            n_chains=n_chains,
            n_dims=n_dims,
            logpdf=logpdf,
            n_local_steps=n_local_steps,
            n_global_steps=n_global_steps,
            n_training_loops=n_training_loops,
            n_production_loops=n_production_loops,
            n_epochs=n_epochs,
            mala_step_size=mala_step_size,
            chain_batch_size=chain_batch_size,
            rq_spline_hidden_units=rq_spline_hidden_units,
            rq_spline_n_bins=rq_spline_n_bins,
            rq_spline_n_layers=rq_spline_n_layers,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            local_thinning=local_thinning,
            global_thinning=global_thinning,
            n_NFproposal_batch_size=n_NFproposal_batch_size,
            verbose=verbose,
        )
        logger.info("RQSpline_MALA_Bundle created successfully.")

        sampler = Sampler(
            n_dims,
            n_chains,
            self.rng_key,
            resource_strategy_bundles=bundle,
        )

        logger.debug("Sampler initialized, starting sampling.")

        if self.debug_nans:
            with jax.debug_nans(True):
                sampler.sample(initial_position, data)
        elif self.profile_memory:
            sampler.sample(initial_position, data)
            import datetime

            time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            filename = f"{self.analysis_name}_memory_{time}.prof"
            jax.profiler.save_device_memory_profile(filename)
            logger.debug("Memory profile saved as {filename}", filename=filename)
        elif self.check_leaks:
            with jax.checking_leaks():
                sampler.sample(initial_position, data)
        else:
            sampler.sample(initial_position, data)

        _save_data_from_sampler(
            sampler,
            rng_key=self.rng_key,
            logpdf=ft.partial(logpdf, data=data),  # type: ignore
            out_dir="sampler_data",
            labels=labels,
            n_samples=sampler_config["data_dump"]["n_samples"],
        )

        logger.info("Sampling and data saving complete.")


flowMC_arg_parser = guru_parser
