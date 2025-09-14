# Copyright 2023 The GWKokab Authors
# SPDX-License-Identifier: Apache-2.0


import functools as ft
import gc
import os
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import numpy as np
import tqdm
from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.resource.states import State
from flowMC.resource_strategy_bundle.base import ResourceStrategyBundle
from flowMC.strategy.base import Strategy
from flowMC.strategy.lambda_function import Lambda
from flowMC.strategy.take_steps import TakeGroupSteps, TakeSerialSteps
from flowMC.strategy.train_model import TrainModel
from flowMC.strategy.update_state import UpdateState
from jax import numpy as jnp, random as jrd
from jaxtyping import Array, Float, PRNGKeyArray
from loguru import logger

from gwkokab.models.utils import JointDistribution
from gwkokab.utils.tools import error_if
from kokab.utils.common import read_json
from kokab.utils.guru import Guru, guru_arg_parser
from kokab.utils.literals import INFERENCE_DIRECTORY


# WARNING: do not change anything in this class


# Copyright (c) 2022 Kaze Wong & contributor
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class RQSpline_MALA_Bundle(ResourceStrategyBundle):
    """A bundle that uses a Rational Quadratic Spline as a normalizing flow model and
    the Metropolis Adjusted Langevin Algorithm as a local sampler.

    This is the base algorithm described in
    https://www.pnas.org/doi/full/10.1073/pnas.2109420119
    """

    def __repr__(self):
        return "RQSpline_MALA Bundle"

    def __init__(
        self,
        rng_key: PRNGKeyArray,
        n_chains: int,
        n_dims: int,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        n_local_steps: int,
        n_global_steps: int,
        n_training_loops: int,
        n_production_loops: int,
        n_epochs: int,
        mala_step_size: float = 1e-1,
        chain_batch_size: int = 0,
        rq_spline_hidden_units: list[int] = [32, 32],
        rq_spline_n_bins: int = 8,
        rq_spline_n_layers: int = 4,
        rq_spline_range: tuple[float, float] = (-10.0, 10.0),
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 10000,
        history_window: int = 100,
        local_thinning: int = 1,
        global_thinning: int = 1,
        n_NFproposal_batch_size: int = 10000,
        verbose: bool = False,
    ):
        n_training_steps = (
            n_local_steps // local_thinning * n_training_loops
            + n_global_steps // global_thinning * n_training_loops
        )
        n_production_steps = (
            n_local_steps // local_thinning * n_production_loops
            + n_global_steps // global_thinning * n_production_loops
        )
        n_total_epochs = n_training_loops * n_epochs

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

        local_sampler = MALA(step_size=mala_step_size)
        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(
            n_dims,
            rq_spline_n_layers,
            rq_spline_hidden_units,
            rq_spline_n_bins,
            subkey,
            rq_spline_range,
        )
        global_sampler = NFProposal(
            model, n_NFproposal_batch_size=n_NFproposal_batch_size
        )
        optimizer = Optimizer(model=model, learning_rate=learning_rate)
        logpdf = LogPDF(logpdf, n_dims=n_dims)

        sampler_state = State(
            {
                "target_positions": "positions_training",
                "target_log_prob": "log_prob_training",
                "target_local_accs": "local_accs_training",
                "target_global_accs": "global_accs_training",
                "training": True,
            },
            name="sampler_state",
        )

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
            "sampler_state": sampler_state,
        }

        local_stepper = TakeSerialSteps(
            "logpdf",
            "local_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_local_accs"],
            n_local_steps,
            thinning=local_thinning,
            chain_batch_size=chain_batch_size,
            verbose=verbose,
        )

        global_stepper = TakeGroupSteps(
            "logpdf",
            "global_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_global_accs"],
            n_global_steps,
            thinning=global_thinning,
            chain_batch_size=chain_batch_size,
            verbose=verbose,
        )

        model_trainer = TrainModel(
            "model",
            "positions_training",
            "optimizer",
            loss_buffer_name="loss_buffer",
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            history_window=history_window,
            verbose=verbose,
        )

        update_state = UpdateState(
            "sampler_state",
            [
                "target_positions",
                "target_log_prob",
                "target_local_accs",
                "target_global_accs",
                "training",
            ],
            [
                "positions_production",
                "log_prob_production",
                "local_accs_production",
                "global_accs_production",
                False,
            ],
        )

        def reset_steppers(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Reset the steppers to the initial position."""
            local_stepper.set_current_position(0)
            global_stepper.set_current_position(0)
            return rng_key, resources, initial_position

        reset_steppers_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: reset_steppers(
                rng_key, resources, initial_position, data
            )
        )

        update_global_step = Lambda(
            lambda rng_key,
            resources,
            initial_position,
            data: global_stepper.set_current_position(local_stepper.current_position)
        )
        update_local_step = Lambda(
            lambda rng_key,
            resources,
            initial_position,
            data: local_stepper.set_current_position(global_stepper.current_position)
        )

        def update_model(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Update the model."""
            model = resources["model"]
            resources["global_sampler"] = eqx.tree_at(
                lambda x: x.model,
                resources["global_sampler"],
                model,
            )
            return rng_key, resources, initial_position

        update_model_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: update_model(
                rng_key, resources, initial_position, data
            )
        )

        self.strategies = {
            "local_stepper": local_stepper,
            "global_stepper": global_stepper,
            "model_trainer": model_trainer,
            "update_state": update_state,
            "update_global_step": update_global_step,
            "update_local_step": update_local_step,
            "reset_steppers": reset_steppers_lambda,
            "update_model": update_model_lambda,
        }

        training_phase = [
            "local_stepper",
            "update_global_step",
            "model_trainer",
            "update_model",
            "global_stepper",
            "update_local_step",
        ]
        production_phase = [
            "local_stepper",
            "update_global_step",
            "global_stepper",
            "update_local_step",
        ]
        strategy_order = []
        for _ in range(n_training_loops):
            strategy_order.extend(training_phase)

        strategy_order.append("reset_steppers")
        strategy_order.append("update_state")
        for _ in range(n_production_loops):
            strategy_order.extend(production_phase)

        self.strategy_order = strategy_order


# WARNING: do not change anything in this class


# Copyright (c) 2022 Kaze Wong & contributor
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class Sampler:
    """Top level API that the users primarily interact with.

    Args:
        n_dim (int): Dimension of the parameter space.
        n_chains (int): Number of chains to sample.
        rng_key (PRNGKeyArray): Jax PRNGKey.
        logpdf (Callable[[Float[Array, "n_dim"], dict], Float):
            Log probability function.
        resources (dict[str, Resource]): Resources to be used by the sampler.
        strategies (dict[str, Strategy]): Strategies to be used by the sampler.
        verbose (bool): Whether to print out progress. Defaults to False.
        logging (bool): Whether to log the progress. Defaults to True.
        outdir (str): Directory to save the logs. Defaults to "./outdir/".
    """

    # Essential parameters
    n_dim: int
    n_chains: int
    rng_key: PRNGKeyArray
    resources: dict[str, Resource]
    strategies: dict[str, Strategy]
    strategy_order: Optional[list[str]]

    # Logging hyperparameters
    verbose: bool = False
    logging: bool = True
    outdir: str = "./outdir/"

    def __init__(
        self,
        n_dim: int,
        n_chains: int,
        rng_key: PRNGKeyArray,
        resources: None | dict[str, Resource] = None,
        strategies: None | dict[str, Strategy] = None,
        strategy_order: None | list[str] = None,
        resource_strategy_bundles: None | ResourceStrategyBundle = None,
        **kwargs,
    ):
        # Copying input into the model

        self.n_dim = n_dim
        self.n_chains = n_chains
        self.rng_key = rng_key

        if resources is not None and strategies is not None:
            print(
                "Resources and strategies provided. Ignoring resource strategy bundles."
            )
            self.resources = resources
            self.strategies = strategies
            self.strategy_order = strategy_order

        else:
            print(
                "Resources or strategies not provided. Using resource strategy bundles."
            )
            if resource_strategy_bundles is None:
                raise ValueError(
                    "Resource strategy bundles not provided."
                    "Please provide either resources and strategies or resource strategy bundles."
                )
            self.resources = resource_strategy_bundles.resources
            self.strategies = resource_strategy_bundles.strategies
            self.strategy_order = resource_strategy_bundles.strategy_order

        # Set and override any given hyperparameters
        class_keys = list(self.__class__.__dict__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def sample(self, initial_position: Float[Array, "n_chains n_dim"], data: dict):
        """Sample from the posterior using the local sampler.

        Args:
            initial_position (Device Array): Initial position.
            data (dict): Data to be used by the likelihood functions
        """

        initial_position = jnp.atleast_2d(initial_position)  # type: ignore
        rng_key = self.rng_key
        last_step = initial_position
        assert isinstance(self.strategy_order, list)
        for strategy in self.strategy_order:
            if strategy not in self.strategies:
                raise ValueError(
                    f"Invalid strategy name '{strategy}' provided. "
                    f"Available strategies are: {list(self.strategies.keys())}."
                )

        n_total_strategy = len(self.strategy_order)
        n_global_steps = self.strategy_order.count("model_trainer")
        n_local_steps = (n_total_strategy - 6 * n_global_steps - 2) // 4

        with tqdm.tqdm(range(n_global_steps), total=n_global_steps) as pbar:
            pbar.set_description("Global Tuning")
            for i in pbar:
                for strategy in self.strategy_order[6 * i : 6 * (i + 1)]:
                    (
                        rng_key,
                        self.resources,
                        last_step,
                    ) = self.strategies[strategy](
                        rng_key, self.resources, last_step, data
                    )

        print("Resetting steppers and updating state")
        for strategy in self.strategy_order[
            6 * n_global_steps : 6 * n_global_steps + 2
        ]:
            (
                rng_key,
                self.resources,
                last_step,
            ) = self.strategies[strategy](rng_key, self.resources, last_step, data)

        with tqdm.tqdm(range(n_local_steps), total=n_local_steps) as pbar:
            pbar.set_description("Global Sampling")
            offset = 6 * n_global_steps + 2
            for i in pbar:
                for strategy in self.strategy_order[
                    offset + 4 * i : offset + 4 * (i + 1)
                ]:
                    (
                        rng_key,
                        self.resources,
                        last_step,
                    ) = self.strategies[strategy](
                        rng_key, self.resources, last_step, data
                    )

    # TODO: Implement quick access and summary functions that operates on buffer

    def serialize(self):
        """Serialize the sampler object."""
        raise NotImplementedError

    def deserialize(self):
        """Deserialize the sampler object."""
        raise NotImplementedError


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

    np.savetxt(rf"{out_dir}/loss.dat", train_loss_vals.reshape(-1), header="loss")
    logger.info("Loss values saved to {out_dir}/loss.dat", out_dir=out_dir)

    for n_chain in tqdm.trange(
        n_chains, total=n_chains, unit="chain", desc="Saving data from chains"
    ):
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

        np.savetxt(
            rf"{out_dir}/train_chains_{n_chain}.dat",
            train_chains[n_chain, :, :],
            header=header,
        )
        np.savetxt(
            rf"{out_dir}/prod_chains_{n_chain}.dat",
            prod_chains[n_chain, :, :],
            header=header,
        )
        np.savetxt(
            rf"{out_dir}/log_prob_{n_chain}.dat",
            np.column_stack(
                _same_length_arrays(
                    max(
                        train_log_prob[n_chain].shape[0],
                        prod_log_prob[n_chain].shape[0],
                    ),
                    train_log_prob[n_chain],
                    prod_log_prob[n_chain],
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

        RQSpline_MALA_Bundle_value: dict = sampler_config.pop(
            "RQSpline_MALA_Bundle", {}
        )

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
        history_window = RQSpline_MALA_Bundle_value.get("history_window", 100)
        n_NFproposal_batch_size = RQSpline_MALA_Bundle_value["n_NFproposal_batch_size"]
        n_production_loops = RQSpline_MALA_Bundle_value["n_production_loops"]
        n_training_loops = RQSpline_MALA_Bundle_value["n_training_loops"]
        rq_spline_hidden_units = RQSpline_MALA_Bundle_value["rq_spline_hidden_units"]
        rq_spline_n_bins = RQSpline_MALA_Bundle_value["rq_spline_n_bins"]
        rq_spline_n_layers = RQSpline_MALA_Bundle_value["rq_spline_n_layers"]
        rq_spline_range = RQSpline_MALA_Bundle_value.get(
            "rq_spline_range", (-10.0, 10.0)
        )
        rq_spline_range = tuple(rq_spline_range)
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
            (history_window, "history_window", int),
            (n_NFproposal_batch_size, "n_NFproposal_batch_size", int),
            (n_production_loops, "n_production_loops", int),
            (n_training_loops, "n_training_loops", int),
            (rq_spline_hidden_units, "rq_spline_hidden_units", list),
            (rq_spline_n_bins, "rq_spline_n_bins", int),
            (rq_spline_n_layers, "rq_spline_n_layers", int),
            (rq_spline_range, "rq_spline_range", tuple),
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
            rq_spline_range=rq_spline_range,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            history_window=history_window,
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
            out_dir=INFERENCE_DIRECTORY,
            labels=labels,
            n_samples=sampler_config["data_dump"]["n_samples"],
        )

        logger.info("Sampling and data saving complete.")


flowMC_arg_parser = guru_arg_parser
